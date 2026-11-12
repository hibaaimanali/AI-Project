"""Stream per-spectrum CASMI evaluation with resumable CUDA inference."""

from collections import Counter
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from rdkit import rdBase
import torch

from SMILES_decoder_module.fingerprint import MORGAN, achiral_molecule
from SMILES_decoder_module.generate import MODEL_DIR, MSGPTDecoder, resolve_device

from .data import FORMULA_COLUMN, LABEL_COLUMNS, PEAK_COLUMNS
from .evaluate import score_candidates
from .metrics import fingerprint_metrics
from .mist_encoder import MistEncoder, spectrum_to_mist


COLUMNS = LABEL_COLUMNS + PEAK_COLUMNS + (FORMULA_COLUMN, "precursor_mz", "adduct", "instrument_type")
ARMS = ("predicted", "oracle")


def _batches(source, start, stop, batch_size):
    group_start = 0
    for group in range(source.num_row_groups):
        group_end = group_start + source.metadata.row_group(group).num_rows
        if group_start >= stop:
            break
        if group_end > start:
            offset = group_start
            for batch in source.iter_batches(batch_size=batch_size, row_groups=[group], columns=COLUMNS):
                end = offset + batch.num_rows
                first, last = max(offset, start), min(end, stop)
                if first < last:
                    yield first, batch.slice(first - offset, last - first).to_pylist()
                if end >= stop:
                    break
                offset = end
        group_start = group_end


def _evaluate_batch(rows, encoder, decoder, reference_fingerprint, threshold, samples, seed):
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
        return results

    predicted_fingerprints = encoder.predict(features) >= threshold
    eligible, fingerprints, targets, formulas = [], [], [], []
    for (index, target), predicted in zip(accepted, predicted_fingerprints, strict=True):
        result = results[index]
        formula = result["provided_formula"]
        result["fingerprint"] = fingerprint_metrics(predicted, target)
        if not decoder.supports_formula(formula):
            result["status"] = "decoder_unsupported_formula"
            continue
        result["status"] = "evaluated"
        eligible.append(result)
        fingerprints.append(predicted)
        targets.append(target)
        formulas.append(formula)

    if eligible:
        for arm, inputs in zip(ARMS, (fingerprints, targets), strict=True):
            torch.manual_seed(seed)
            candidates = decoder.generate_batch(np.stack(inputs), formulas, samples)
            for result, ranked in zip(eligible, candidates, strict=True):
                result[arm] = score_candidates(ranked, result["inchikey14"], result["provided_formula"])
    return results


def _accumulate(totals, results):
    for result in results:
        totals[f"status.{result['status']}"] += 1
        if "fingerprint" in result:
            totals["encoded"] += 1
            for name, value in result["fingerprint"].items():
                totals[f"fingerprint.{name}"] += value
        if result["status"] == "evaluated":
            for arm in ARMS:
                for metric in ("top1", "recovered", "has_formula_match", "valid_samples"):
                    totals[f"{arm}.{metric}"] += result[arm][metric]


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


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
            for arm in ARMS
        },
        "elapsed_seconds": elapsed,
        "rows_per_second": scanned / elapsed if elapsed else None,
        "peak_cuda_allocated_mib": progress["peak_cuda_bytes"] / (1024 ** 2),
    }


def run_pipeline(data, checkpoint, database, *, max_rows=None, threshold=0.20, samples=4,
                 batch_size=16, seed=42, compile_backend="inductor", cached_generation=True):
    """Evaluate each selected Parquet row and commit every completed batch to SQLite."""
    device = resolve_device("cuda")
    if (not 0 < threshold < 1 or min(samples, batch_size) < 1 or seed < 0
            or (max_rows is not None and max_rows < 1)):
        raise ValueError("Use positive batch/sample sizes, a nonnegative seed, and a threshold in (0, 1).")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetFile(data) as source, closing(sqlite3.connect(database)) as connection:
        missing = set(COLUMNS) - set(source.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing CASMI columns: {', '.join(sorted(missing))}")
        total = source.metadata.num_rows
        stop = total if max_rows is None else min(max_rows, total)
        config = {
            "data": _file_identity(data), "mist_checkpoint": _file_identity(checkpoint),
            "decoder_checkpoint": _file_identity(MODEL_DIR / "ms_gpt_mlm.pt"),
            "source_rows": total, "target_rows": stop, "threshold": threshold, "samples": samples,
            "batch_size": batch_size, "seed": seed, "device": device,
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "compile_backend": compile_backend,
        }
        if cached_generation:
            config["generation"] = "cached_cross_attention"
        config_json = json.dumps(config, sort_keys=True)
        initial = {"next_row": 0, "totals": {}, "elapsed_seconds": 0.0, "peak_cuda_bytes": 0}
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
        start = progress["next_row"]
        if start == stop:
            return _summary(config, progress)

        print(f"{config['gpu']} | {start:,}/{stop:,} spectra already processed", flush=True)
        torch.cuda.reset_peak_memory_stats()
        encoder, decoder = MistEncoder(checkpoint, device), MSGPTDecoder(device)
        encoder.model.compile(backend=compile_backend, dynamic=True)
        if cached_generation:
            decoder.enable_cached_generation(compile_backend)
        else:
            for layer in decoder.model.cross_attn_layers.values():
                layer.compile(backend=compile_backend, dynamic=True, fullgraph=True)
        print(f"Compiling MIST and MS-GPT cross-attention with {compile_backend} on first use.", flush=True)

        @lru_cache(maxsize=8192)
        def reference_fingerprint(smiles):
            try:
                return MORGAN.GetFingerprintAsNumPy(achiral_molecule(smiles))
            except ValueError:
                return None

        totals = Counter(progress["totals"])
        prior_elapsed = progress["elapsed_seconds"]
        began = perf_counter()
        for first, rows in _batches(source, start, stop, batch_size):
            # Keep repeated chemistry diagnostics from flooding notebook output.
            with rdBase.BlockLogs():
                results = _evaluate_batch(
                    rows, encoder, decoder, reference_fingerprint, threshold, samples, (seed + first) % 2**64,
                )
            _accumulate(totals, results)
            last = first + len(rows)
            progress.update(
                next_row=last, totals=totals, elapsed_seconds=prior_elapsed + perf_counter() - began,
                peak_cuda_bytes=max(progress["peak_cuda_bytes"], torch.cuda.max_memory_allocated()),
            )
            next_progress = json.dumps(progress)
            with connection:
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
