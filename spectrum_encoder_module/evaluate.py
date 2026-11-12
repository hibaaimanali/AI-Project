"""Evaluate pretrained MIST fingerprints and MS-GPT decoding on labeled CASMI spectra."""

import argparse
from collections import Counter
from itertools import batched
import json
from pathlib import Path
import random

import numpy as np
import torch

from SMILES_decoder_module.generate import MSGPTDecoder, resolve_device

from .data import ARTIFACTS, DEFAULT_DATA, FORMULA_COLUMN, collect_molecules, split_molecules
from .metrics import THRESHOLDS, fingerprint_metrics


MIST_FILENAME = "mist_msg_noaug.ckpt"


def predict_split(encoder, spectra, keys, batch_size):
    from .mist_encoder import spectrum_to_mist

    skipped = Counter()
    records = []
    for batch in batched(keys, batch_size):
        features, accepted = [], []
        for key in batch:
            position, row, _, target = spectra[key][0]
            feature, reason = spectrum_to_mist(row)
            if reason:
                skipped[reason] += 1
                continue
            features.append(feature)
            accepted.append((position, row, target))
        if features:
            records.extend(
                (*record, probability)
                for record, probability in zip(accepted, encoder.predict(features), strict=True)
            )
    return records, dict(skipped)


def score_candidates(candidates, key, formula):
    return {
        "top1": bool(candidates) and candidates[0]["key"] == key,
        "recovered": any(candidate["key"] == key for candidate in candidates),
        "has_formula_match": any(candidate["formula"] == formula for candidate in candidates),
        "valid_samples": sum(candidate["count"] for candidate in candidates),
        "candidates": candidates,
    }


def evaluate(args):
    from .mist_encoder import MistEncoder

    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    spectra, fingerprints = collect_molecules(
        args.data, max_rows=args.max_rows, max_molecules=args.max_molecules,
        spectra_per_molecule=3, max_peaks=32, seed=args.seed,
        extra_columns=(FORMULA_COLUMN, "instrument_type"),
    )
    # Use the same structure-grouped partition as the original CASMI smoke run.
    split, group_count, _ = split_molecules(fingerprints, 0.65, 0.15, 0.15, args.seed)
    print(f"Loaded {len(spectra)} molecules in {group_count} structural groups. Loading MIST on {device}.",
          flush=True)
    encoder = MistEncoder(args.checkpoint, device)
    validation, validation_skips = predict_split(encoder, spectra, split["validation"], args.batch_size)
    if not validation:
        raise ValueError(f"No validation spectra were usable by MIST: {validation_skips}")
    validation_targets = np.stack([record[2] for record in validation])
    validation_probabilities = np.stack([record[3] for record in validation])
    sweep = [
        {"threshold": threshold, **fingerprint_metrics(validation_probabilities >= threshold, validation_targets)}
        for threshold in THRESHOLDS
    ]
    threshold = max(sweep, key=lambda row: (row["tanimoto"], row["threshold"]))["threshold"]
    print(f"Selected threshold {threshold:g} on {len(validation)} validation molecules.", flush=True)
    test, test_skips = predict_split(encoder, spectra, split["test"], args.batch_size)
    if not test:
        raise ValueError(f"No test spectra were usable by MIST: {test_skips}")
    test_metrics = fingerprint_metrics(
        np.stack([record[3] >= threshold for record in test]), np.stack([record[2] for record in test]),
    )
    parameter_count = encoder.parameter_count
    del encoder
    print(f"Scored fingerprints for {len(test)}/{len(split['test'])} test molecules "
          f"(mean Tanimoto {test_metrics['tanimoto']:.3f}).", flush=True)

    print(f"Loading the existing MS-GPT decoder on {device}.", flush=True)
    decoder = MSGPTDecoder(device)
    eligible = [record for record in test if decoder.supports_formula(record[1][FORMULA_COLUMN])]
    if not eligible:
        raise ValueError("No held-out spectra have an MS-GPT-compatible provided formula")
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[:args.molecules] if args.molecules else eligible
    results, predicted_bits, true_bits = [], [], []
    for index, (position, row, oracle, probabilities) in enumerate(selected, start=1):
        key = row["inchikey14"]
        formula = row[FORMULA_COLUMN]
        predicted = probabilities >= threshold

        pair_seed = args.seed + index - 1
        torch.manual_seed(pair_seed)
        predicted_candidates = decoder.generate(predicted, formula, args.samples)

        torch.manual_seed(pair_seed)
        oracle_candidates = decoder.generate(oracle, formula, args.samples)
        predicted_bits.append(predicted)
        true_bits.append(oracle)
        result = {
            "inchikey14": key, "parquet_position": list(position), "provided_formula": formula, "seed": pair_seed,
            "fingerprint": fingerprint_metrics(predicted, oracle),
            "predicted": score_candidates(predicted_candidates, key, formula),
            "oracle": score_candidates(oracle_candidates, key, formula),
        }
        results.append(result)
        print(f"[{index}/{len(selected)}] {key} ({formula}) | "
              f"Tanimoto {result['fingerprint']['tanimoto']:.3f} | "
              f"predicted recovered {result['predicted']['recovered']} | "
              f"oracle recovered {result['oracle']['recovered']}", flush=True)

    summary = {
        "encoder": "MIST", "checkpoint": str(args.checkpoint.resolve()), "parameters": parameter_count,
        "seed": args.seed, "max_rows": args.max_rows, "max_molecules": args.max_molecules,
        "validation_molecules": len(split["validation"]), "validation_encoded": len(validation),
        "validation_skips": validation_skips, "validation_threshold_sweep": sweep,
        "test_molecules": len(split["test"]), "test_encoded": len(test), "test_skips": test_skips,
        "test_fingerprint": test_metrics, "evaluated_molecules": len(results),
        "decoder_unsupported_formulas": len(test) - len(eligible),
        "samples_per_molecule_per_arm": args.samples, "threshold": threshold,
        "fingerprint": fingerprint_metrics(np.stack(predicted_bits), np.stack(true_bits)),
    }
    for arm in ("predicted", "oracle"):
        summary[arm] = {
            name: sum(result[arm][name] for result in results)
            for name in ("top1", "recovered", "has_formula_match", "valid_samples")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "molecules": results}, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Saved candidate SMILES to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Extracted CASMI train.parquet")
    parser.add_argument("--checkpoint", type=Path, help=f"Path to {MIST_FILENAME} (artifacts or project root)")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "mist_smoke_candidates.json")
    parser.add_argument("--max-rows", type=int, default=8192, help="Maximum Parquet rows scanned from the start")
    parser.add_argument("--max-molecules", type=int, default=512, help="Maximum molecules used for partitioning")
    parser.add_argument("--batch-size", type=int, default=8, help="MIST inference batch size")
    parser.add_argument("--molecules", type=int, default=4, help="Number of eligible test molecules, 0 for all")
    parser.add_argument("--samples", type=int, default=4, help="Generation attempts per molecule in each arm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.molecules < 0 or min(args.samples, args.threads, args.batch_size, args.max_rows,
                                args.max_molecules) < 1:
        parser.error("--molecules must be nonnegative. The other limits must be positive.")
    if args.checkpoint is None:
        args.checkpoint = ARTIFACTS / MIST_FILENAME
        if not args.checkpoint.is_file():
            args.checkpoint = ARTIFACTS.parent.parent / MIST_FILENAME
    try:
        evaluate(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
