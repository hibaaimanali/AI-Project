"""Stream per-spectrum CASMI evaluation with resumable CUDA inference."""

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, nullcontext
from functools import lru_cache
from itertools import batched
import json
import multiprocessing
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from rdkit import rdBase
import torch

from SMILES_decoder_module.fingerprint import MORGAN, achiral_molecule
from SMILES_decoder_module.generate import (
    MODEL_DIR, MSGPTDecoder, normalize_sample, rank_candidates, resolve_device, supports_formula,
)

from .data import FORMULA_COLUMN, LABEL_COLUMNS, PEAK_COLUMNS
from .evaluate import score_candidates
from .metrics import fingerprint_metrics


COLUMNS = LABEL_COLUMNS + PEAK_COLUMNS + (FORMULA_COLUMN, "precursor_mz", "adduct", "instrument_type")
ARMS = ("predicted", "oracle")
ORACLE_COLUMNS = LABEL_COLUMNS + (FORMULA_COLUMN,)


def _batches(source, start, stop, batch_size, columns=COLUMNS):
    group_start = 0
    for group in range(source.num_row_groups):
        group_end = group_start + source.metadata.row_group(group).num_rows
        if group_start >= stop:
            break
        if group_end > start:
            offset = group_start
            for batch in source.iter_batches(batch_size=batch_size, row_groups=[group], columns=columns):
                end = offset + batch.num_rows
                first, last = max(offset, start), min(end, stop)
                if first < last:
                    yield first, batch.slice(first - offset, last - first).to_pylist()
                if end >= stop:
                    break
                offset = end
        group_start = group_end


class CandidateCache:
    """Store one candidate pool for each exact MS-GPT conditioning input."""

    def __init__(self, connection, decoder, samples, batch_size):
        self.connection = connection
        self.decoder = decoder
        self.samples = samples
        self.batch_size = batch_size
        self.generation_seconds = 0.0
        self.unique_inputs = 0

    def generate(self, fingerprints, formulas):
        packed = np.packbits(fingerprints, axis=1)
        keys = [(formula, bits.tobytes()) for formula, bits in zip(formulas, packed, strict=True)]
        pools, missing = {}, {}
        for index, key in enumerate(keys):
            if key in pools or key in missing:
                continue
            found = self.connection.execute(
                "SELECT candidates FROM decoder_cache WHERE formula = ? AND fingerprint = ?", key,
            ).fetchone()
            if found is None:
                missing[key] = index
            else:
                pools[key] = json.loads(found[0])

        for chunk in batched(missing, self.batch_size):
            indices = [missing[key] for key in chunk]
            started = perf_counter()
            candidates = self.decoder.generate_batch(
                np.stack([fingerprints[index] for index in indices]),
                [formulas[index] for index in indices], self.samples,
            )
            self.generation_seconds += perf_counter() - started
            pools.update(zip(chunk, candidates, strict=True))
            self.connection.executemany(
                "INSERT INTO decoder_cache VALUES (?, ?, ?)",
                ((formula, bits, json.dumps(pools[formula, bits])) for formula, bits in chunk),
            )
            self.unique_inputs += len(chunk)
        return [pools[key] for key in keys]


def _encode_batch(rows, encoder, reference_fingerprint, threshold, formula_supported):
    from .mist_encoder import spectrum_to_mist

    results, features, accepted = [], [], []
    for index, row in enumerate(rows):
        key, smiles = (row[name] for name in LABEL_COLUMNS)
        formula = row[FORMULA_COLUMN]
        result = {"inchikey14": key, "provided_formula": formula}
        results.append(result)
        target = reference_fingerprint(smiles) if key and smiles else None
        if target is None:
            result["status"] = "invalid_reference" if key and smiles else "missing_reference"
            continue
        feature, reason = spectrum_to_mist(row)
        if reason:
            result["status"] = reason
            continue
        features.append(feature)
        accepted.append((index, target))

    if not features:
        return results, [], [], [], []

    predicted_fingerprints = encoder.predict(features) >= threshold
    eligible, fingerprints, targets, formulas = [], [], [], []
    for (index, target), predicted in zip(accepted, predicted_fingerprints, strict=True):
        result = results[index]
        formula = result["provided_formula"]
        result["fingerprint"] = fingerprint_metrics(predicted, target)
        if not formula_supported(formula):
            result["status"] = "decoder_unsupported_formula"
            continue
        result["status"] = "evaluated"
        eligible.append(index)
        fingerprints.append(predicted)
        targets.append(target)
        formulas.append(formula)

    return results, eligible, fingerprints, targets, formulas


def _oracle_batch(rows):
    results, eligible, targets, formulas = [], [], [], []
    for index, row in enumerate(rows):
        key, smiles = (row[name] for name in LABEL_COLUMNS)
        formula = row[FORMULA_COLUMN]
        result = {"inchikey14": key, "provided_formula": formula}
        results.append(result)
        target = _reference_fingerprint(smiles) if key and smiles else None
        if target is None:
            result["status"] = "invalid_reference" if key and smiles else "missing_reference"
        elif not supports_formula(formula):
            result["status"] = "decoder_unsupported_formula"
        else:
            result["status"] = "evaluated"
            eligible.append(index)
            targets.append(target)
            formulas.append(formula)
    return results, eligible, None, targets, formulas


def _evaluate_batch(rows, encoder, decoder, reference_fingerprint, threshold, samples, seed,
                    candidate_cache=None):
    results, eligible, fingerprints, targets, formulas = _encode_batch(
        rows, encoder, reference_fingerprint, threshold, decoder.supports_formula,
    )
    if eligible:
        if candidate_cache is not None:
            torch.manual_seed(seed)
            pools = candidate_cache.generate(fingerprints + targets, formulas + formulas)
            arm_candidates = (pools[:len(eligible)], pools[len(eligible):])
        else:
            arm_candidates = []
            for inputs in (fingerprints, targets):
                torch.manual_seed(seed)
                arm_candidates.append(decoder.generate_batch(np.stack(inputs), formulas, samples))
        for arm, candidates in zip(ARMS, arm_candidates, strict=True):
            for index, ranked in zip(eligible, candidates, strict=True):
                result = results[index]
                result[arm] = score_candidates(ranked, result["inchikey14"], result["provided_formula"])
    return results


def _accumulate(totals, results, arms=ARMS):
    for result in results:
        totals[f"status.{result['status']}"] += 1
        if "fingerprint" in result:
            totals["encoded"] += 1
            for name, value in result["fingerprint"].items():
                totals[f"fingerprint.{name}"] += value
        if result["status"] == "evaluated":
            for arm in arms:
                for metric in ("top1", "recovered", "has_formula_match", "valid_samples"):
                    totals[f"{arm}.{metric}"] += result[arm][metric]


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


@lru_cache(maxsize=8192)
def _reference_fingerprint(smiles):
    try:
        return MORGAN.GetFingerprintAsNumPy(achiral_molecule(smiles))
    except ValueError:
        return None


def _load_decoder(config, graph_batch_size):
    decoder = MSGPTDecoder(config["device"])
    backend = config["compile_backend"]
    if "generation" in config:
        decoder.enable_cached_generation(
            backend, cuda_graph=config["generation"] in {"cuda_graph", "cuda_graph_prefix"},
            graph_batch_size=graph_batch_size, prefix_attention=config["generation"] == "cuda_graph_prefix",
        )
    elif backend != "eager":
        for layer in decoder.model.cross_attn_layers.values():
            layer.compile(backend=backend, dynamic=True, fullgraph=True)
    return decoder


def _summary(config, progress):
    totals = Counter(progress["totals"])
    scanned = progress["next_row"]
    encoded, evaluated = totals["encoded"], totals["status.evaluated"]
    elapsed = progress["elapsed_seconds"]
    return {
        **config,
        "rows_scanned": scanned,
        "encoded": encoded,
        "evaluated": evaluated,
        "status_counts": {key.split(".", 1)[1]: value for key, value in totals.items()
                          if key.startswith("status.")},
        "fingerprint": {key.split(".", 1)[1]: value / encoded for key, value in totals.items()
                        if key.startswith("fingerprint.")},
        "arms": {
            arm: {
                "top1_rate": totals[f"{arm}.top1"] / evaluated if evaluated else None,
                "top_k_recovery_rate": totals[f"{arm}.recovered"] / evaluated if evaluated else None,
                "formula_match_rate": totals[f"{arm}.has_formula_match"] / evaluated if evaluated else None,
                "valid_sample_rate": totals[f"{arm}.valid_samples"] / (evaluated * config["samples"])
                if evaluated else None,
                "recovery_rate_all_rows": totals[f"{arm}.recovered"] / scanned if scanned else None,
            }
            for arm in (("oracle",) if config.get("oracle_only") else ARMS)
        },
        "elapsed_seconds": elapsed,
        "rows_per_second": scanned / elapsed if elapsed else None,
        "peak_cuda_allocated_mib": progress["peak_cuda_bytes"] / (1024 ** 2),
        **({key: progress.get(key, 0) for key in ("unique_decoder_inputs", "candidate_generation_seconds")}
           if config.get("reuse_candidates") else {}),
        **({"phase": progress["phase"], "decoded_inputs": progress["decoded_inputs"],
            ("oracle_seconds" if config.get("oracle_only") else "encoder_seconds"): progress["encoder_seconds"]}
           if config.get("staged") else {}),
    }


def _save_batch(connection, progress, previous, began):
    progress["elapsed_seconds"] += perf_counter() - began
    progress["peak_cuda_bytes"] = max(progress["peak_cuda_bytes"], torch.cuda.max_memory_allocated())
    current = json.dumps(progress)
    changed = connection.execute(
        "UPDATE run SET progress = ? WHERE id = 1 AND progress = ?", (current, previous),
    )
    if changed.rowcount != 1:
        raise RuntimeError("Another process advanced this database. Reopen it to continue.")
    return current


def _stage_inputs(connection, first, results, eligible, fingerprints, targets, formulas):
    def input_ids(vectors):
        ids = []
        for formula, packed in zip(formulas, np.packbits(vectors, axis=1), strict=True):
            key = (formula, packed.tobytes())
            connection.execute("INSERT OR IGNORE INTO decoder_inputs (formula, fingerprint) VALUES (?, ?)", key)
            ids.append(connection.execute(
                "SELECT id FROM decoder_inputs WHERE formula = ? AND fingerprint = ?", key,
            ).fetchone()[0])
        return ids

    if eligible:
        predicted_ids = [None] * len(eligible) if fingerprints is None else input_ids(fingerprints)
        connection.executemany(
            "INSERT INTO spectrum_inputs VALUES (?, ?, ?)",
            zip((first + index for index in eligible), predicted_ids, input_ids(targets), strict=True),
        )
    connection.executemany(
        "INSERT INTO spectra VALUES (?, ?, ?)",
        ((first + index, result["status"], json.dumps(result)) for index, result in enumerate(results)),
    )


def _decode_batches(connection, config, progress, decoder, workers):
    """Yield ranked batches in commit order while overlapping GPU and CPU work."""
    total = progress["unique_decoder_inputs"]
    remaining = total - progress["decoded_inputs"]
    if remaining == 0:
        return
    last_id = progress["last_input_id"]
    batch_size = config["decoder_batch_size"]
    samples = config["samples"]
    pending = deque()
    in_flight = {}
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers else nullcontext()
    )
    with executor as pool:
        while remaining:
            batch = connection.execute(
                "SELECT id, formula, fingerprint FROM decoder_inputs WHERE id > ? ORDER BY id LIMIT ?",
                (last_id, batch_size),
            ).fetchall()
            ids, formulas, packed = zip(*batch)
            last_id = ids[-1]
            remaining -= len(ids)
            fingerprints = np.unpackbits(np.stack([np.frombuffer(bits, dtype=np.uint8) for bits in packed]), axis=1)
            torch.manual_seed((config["seed"] + ids[0]) % 2**64)
            if pool is None:
                with rdBase.BlockLogs():
                    candidates = decoder.generate_batch(fingerprints, formulas, samples)
                yield ids, candidates
                continue
            tokens = decoder.sample_batch(fingerprints, formulas, samples)
            sequences = [decoder.sample_sequences(group) for group in tokens]
            unique = dict.fromkeys(sequence for group in sequences for sequence in group if sequence)
            futures = {}
            for sequence in unique:
                if sequence not in in_flight:
                    in_flight[sequence] = pool.submit(normalize_sample, sequence)
                futures[sequence] = in_flight[sequence]
            pending.append((ids, formulas, sequences, futures))
            # Bound the backlog while allowing a slow molecule to overlap several GPU batches.
            if len(pending) >= 2 * workers:
                yield _rank_decoded_batch(*pending.popleft())
                in_flight = {
                    sequence: future
                    for _, _, _, batch_futures in pending
                    for sequence, future in batch_futures.items()
                }
        for batch in pending:
            yield _rank_decoded_batch(*batch)


def _rank_decoded_batch(ids, formulas, sequences, futures):
    normalized = {sequence: future.result() for sequence, future in futures.items()}
    return ids, [
        rank_candidates((normalized.get(sequence) for sequence in group), formula)
        for group, formula in zip(sequences, formulas, strict=True)
    ]


def _run_staged(source, connection, config, progress, previous, encode_only, ranking_workers):
    stop, batch_size = config["target_rows"], config["batch_size"]
    decoder_batch_size = config["decoder_batch_size"]
    oracle_only = config.get("oracle_only", False)
    arms = ("oracle",) if oracle_only else ARMS
    columns = ORACLE_COLUMNS if oracle_only else COLUMNS
    stage_name = "Oracle fingerprints" if oracle_only else "MIST"
    torch.cuda.reset_peak_memory_stats()
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS decoder_inputs (id INTEGER PRIMARY KEY, formula TEXT NOT NULL, "
            "fingerprint BLOB NOT NULL, candidates TEXT, UNIQUE (formula, fingerprint))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS spectrum_inputs (row_index INTEGER PRIMARY KEY, "
            "predicted_id INTEGER, oracle_id INTEGER NOT NULL)"
        )

    if progress["phase"] == "encode":
        start = progress["next_row"]
        print(f"{stage_name} | {start:,}/{stop:,} rows checked", flush=True)
        if start < stop:
            encoder = None
            if not oracle_only:
                from .mist_encoder import MistEncoder

                encoder = MistEncoder(config["mist_checkpoint"]["path"], config["device"])
                if config["compile_backend"] != "eager":
                    encoder.model.compile(backend=config["compile_backend"], dynamic=True)
            for first, rows in _batches(source, start, stop, batch_size, columns):
                began = perf_counter()
                with rdBase.BlockLogs():
                    if oracle_only:
                        encoded = _oracle_batch(rows)
                    else:
                        encoded = _encode_batch(
                            rows, encoder, _reference_fingerprint, config["threshold"], supports_formula,
                        )
                last = first + len(rows)
                with connection:
                    _stage_inputs(connection, first, *encoded)
                    progress["next_row"] = last
                    progress["encoder_seconds"] += perf_counter() - began
                    previous = _save_batch(connection, progress, previous, began)
                if first == start or last == stop or first // 4096 != last // 4096:
                    print(f"{stage_name} | {last:,}/{stop:,} rows checked", flush=True)
            del encoder
            torch.cuda.empty_cache()
        with connection:
            began = perf_counter()
            progress["unique_decoder_inputs"] = connection.execute("SELECT COUNT(*) FROM decoder_inputs").fetchone()[0]
            progress["phase"] = "decode"
            previous = _save_batch(connection, progress, previous, began)
        print(f"{stage_name} complete | {progress['unique_decoder_inputs']:,} unique inputs", flush=True)

    if encode_only:
        return _summary(config, progress)

    if progress["phase"] == "decode":
        total = progress["unique_decoder_inputs"]
        decoder = _load_decoder(config, decoder_batch_size) if progress["decoded_inputs"] < total else None
        print(f"MS-GPT | {progress['decoded_inputs']:,}/{total:,} unique inputs decoded", flush=True)
        began = perf_counter()
        for ids, candidates in _decode_batches(connection, config, progress, decoder, ranking_workers):
            progress["candidate_generation_seconds"] = (
                progress.get("candidate_generation_seconds", 0.0) + perf_counter() - began
            )
            before = progress["decoded_inputs"]
            with connection:
                connection.executemany(
                    "UPDATE decoder_inputs SET candidates = ? WHERE id = ?",
                    ((json.dumps(pool), index) for index, pool in zip(ids, candidates, strict=True)),
                )
                progress["decoded_inputs"] += len(ids)
                progress["last_input_id"] = ids[-1]
                previous = _save_batch(connection, progress, previous, began)
            done = progress["decoded_inputs"]
            if before == 0 or done == total or before // 1024 != done // 1024:
                seconds = progress["candidate_generation_seconds"]
                print(
                    f"MS-GPT | {done:,}/{total:,} unique inputs decoded and ranked | {done / seconds:.1f} inputs/s",
                    flush=True,
                )
            began = perf_counter()
        del decoder
        with connection:
            progress["phase"] = "score"
            previous = _save_batch(connection, progress, previous, perf_counter())

    if progress["phase"] == "score":
        totals = Counter(progress["totals"])
        while progress["scored_row"] < stop:
            began = perf_counter()
            first = progress["scored_row"]
            last = min(first + 4096, stop)
            rows = connection.execute(
                "SELECT s.row_index, s.result, p.candidates, o.candidates FROM spectra s "
                "LEFT JOIN spectrum_inputs i ON i.row_index = s.row_index "
                "LEFT JOIN decoder_inputs p ON p.id = i.predicted_id "
                "LEFT JOIN decoder_inputs o ON o.id = i.oracle_id "
                "WHERE s.row_index >= ? AND s.row_index < ? ORDER BY s.row_index",
                (first, last),
            )
            results, updates = [], []
            for index, payload, predicted, oracle in rows:
                result = json.loads(payload)
                if result["status"] == "evaluated":
                    pools = (oracle,) if oracle_only else (predicted, oracle)
                    for arm, pool in zip(arms, pools, strict=True):
                        result[arm] = score_candidates(
                            json.loads(pool), result["inchikey14"], result["provided_formula"],
                        )
                    updates.append((json.dumps(result), index))
                results.append(result)
            _accumulate(totals, results, arms)
            with connection:
                connection.executemany("UPDATE spectra SET result = ? WHERE row_index = ?", updates)
                progress.update(scored_row=last, totals=totals, phase="done" if last == stop else "score")
                previous = _save_batch(connection, progress, previous, began)
        if stop == 0:
            with connection:
                progress["phase"] = "done"
                _save_batch(connection, progress, previous, perf_counter())
    return _summary(config, progress)


def run_pipeline(data, checkpoint, database, *, max_rows=None, threshold=0.20, samples=4,
                 batch_size=16, seed=42, compile_backend="inductor", cached_generation=True,
                 reuse_candidates=False, decoder_batch_size=32, cuda_graph=False,
                 staged=False, encode_only=False, oracle_only=False, prefix_attention=False, ranking_workers=0):
    """Evaluate selected rows with resumable batches. Oracle-only runs accept checkpoint=None."""
    device = resolve_device("cuda")
    staged = staged or oracle_only
    if ((not oracle_only and not 0 < threshold < 1)
            or min(samples, batch_size, decoder_batch_size) < 1 or seed < 0
            or (max_rows is not None and max_rows < 1)):
        raise ValueError("Use positive batch/sample sizes, a nonnegative seed, and a threshold in (0, 1).")
    if cuda_graph and not cached_generation:
        raise ValueError("CUDA graph decoding requires cached generation.")
    if prefix_attention and not cuda_graph:
        raise ValueError("Prefix attention requires CUDA graph decoding.")
    if encode_only and not staged:
        raise ValueError("encode_only requires staged evaluation.")
    if ranking_workers < 0 or (ranking_workers and not staged):
        raise ValueError("ranking_workers requires staged evaluation and a nonnegative count.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetFile(data) as source, closing(sqlite3.connect(database)) as connection:
        missing = set(ORACLE_COLUMNS if oracle_only else COLUMNS) - set(source.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing CASMI columns: {', '.join(sorted(missing))}")
        total = source.metadata.num_rows
        stop = total if max_rows is None else min(max_rows, total)
        config = {
            "data": _file_identity(data),
            "decoder_checkpoint": _file_identity(MODEL_DIR / "ms_gpt_mlm.pt"),
            "source_rows": total, "target_rows": stop, "samples": samples,
            "batch_size": batch_size, "seed": seed, "device": device,
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "compile_backend": compile_backend,
        }
        if oracle_only:
            config["oracle_only"] = True
        else:
            config["mist_checkpoint"] = _file_identity(checkpoint)
            config["threshold"] = threshold
        if cached_generation:
            config["generation"] = "cuda_graph" if cuda_graph else "cached_cross_attention"
            if prefix_attention:
                config["generation"] = "cuda_graph_prefix"
        if reuse_candidates or staged:
            config["reuse_candidates"] = "exact_conditioning"
            config["decoder_batch_size"] = decoder_batch_size
        initial = {"next_row": 0, "totals": {}, "elapsed_seconds": 0.0, "peak_cuda_bytes": 0}
        if staged:
            config["staged"] = True
            initial.update(phase="encode", encoder_seconds=0.0, decoded_inputs=0, last_input_id=0, scored_row=0)
        config_json = json.dumps(config, sort_keys=True)
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY, config TEXT, progress TEXT)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS spectra (row_index INTEGER PRIMARY KEY, status TEXT, result TEXT)"
            )
            connection.execute("INSERT OR IGNORE INTO run VALUES (1, ?, ?)", (config_json, json.dumps(initial)))
        saved_config, saved_progress = connection.execute("SELECT config, progress FROM run WHERE id = 1").fetchone()
        if saved_config != config_json:
            raise ValueError(f"{database} belongs to a different run. Choose a new database path.")
        progress = json.loads(saved_progress)
        if staged:
            return _run_staged(source, connection, config, progress, saved_progress, encode_only, ranking_workers)
        start = progress["next_row"]
        if start == stop:
            return _summary(config, progress)

        print(f"{config['gpu']} | {start:,}/{stop:,} spectra already processed", flush=True)
        torch.cuda.reset_peak_memory_stats()
        from .mist_encoder import MistEncoder

        encoder = MistEncoder(checkpoint, device)
        if compile_backend != "eager":
            encoder.model.compile(backend=compile_backend, dynamic=True)
        decoder = _load_decoder(config, decoder_batch_size if reuse_candidates else batch_size)
        candidate_cache = None
        if reuse_candidates:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS decoder_cache (formula TEXT, fingerprint BLOB, "
                    "candidates TEXT NOT NULL, PRIMARY KEY (formula, fingerprint)) WITHOUT ROWID"
                )
            candidate_cache = CandidateCache(connection, decoder, samples, decoder_batch_size)
            candidate_cache.generation_seconds = progress.get("candidate_generation_seconds", 0.0)
            candidate_cache.unique_inputs = connection.execute("SELECT COUNT(*) FROM decoder_cache").fetchone()[0]
        totals = Counter(progress["totals"])
        prior_elapsed = progress["elapsed_seconds"]
        began = perf_counter()
        for first, rows in _batches(source, start, stop, batch_size):
            # Keep repeated chemistry diagnostics from flooding notebook output.
            with connection:
                if candidate_cache is not None:
                    # Reserve the batch before reading or filling the persistent candidate cache.
                    connection.execute("BEGIN IMMEDIATE")
                    current_progress = connection.execute("SELECT progress FROM run WHERE id = 1").fetchone()[0]
                    if current_progress != saved_progress:
                        raise RuntimeError("Another process advanced this database. Reopen it to continue.")
                with rdBase.BlockLogs():
                    results = _evaluate_batch(
                        rows, encoder, decoder, _reference_fingerprint, threshold, samples,
                        (seed + first) % 2**64, candidate_cache,
                    )
                _accumulate(totals, results)
                last = first + len(rows)
                progress.update(
                    next_row=last, totals=totals, elapsed_seconds=prior_elapsed + perf_counter() - began,
                    peak_cuda_bytes=max(progress["peak_cuda_bytes"], torch.cuda.max_memory_allocated()),
                )
                if candidate_cache is not None:
                    progress["candidate_generation_seconds"] = candidate_cache.generation_seconds
                    progress["unique_decoder_inputs"] = candidate_cache.unique_inputs
                next_progress = json.dumps(progress)
                updated = connection.execute(
                    "UPDATE run SET progress = ? WHERE id = 1 AND progress = ?", (next_progress, saved_progress),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("Another process advanced this database. Reopen it to continue.")
                connection.executemany(
                    "INSERT INTO spectra VALUES (?, ?, ?)",
                    ((index, result["status"], json.dumps(result)) for index, result in enumerate(results, first)),
                )
            saved_progress = next_progress
            interval = 256 if last < 1024 else 4096
            if first == start or last == stop or first // interval != last // interval:
                rate = last / progress["elapsed_seconds"]
                print(f"{last:,}/{stop:,} | {totals['status.evaluated']:,} decoded | {rate:.1f} rows/s", flush=True)
        return _summary(config, progress)
