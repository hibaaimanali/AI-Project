"""Compare MS-GPT decoding from predicted and oracle fingerprints on held-out CASMI molecules."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from SMILES_decoder_module.fingerprint import MORGAN, achiral_molecule
from SMILES_decoder_module.generate import MSGPTDecoder, resolve_device

from . import CHECKPOINT_VERSION
from .data import (
    ARTIFACTS, FORMULA_COLUMN, LABEL_COLUMNS, SPECTRUM_COLUMNS, feature_batch,
    parquet_rows, spectrum_features,
)
from .metrics import fingerprint_metrics
from .model import SpectrumEncoder


def load_heldout(path, saved):
    positions = defaultdict(set)
    expected = {}
    for key, (group, offset) in saved.items():
        positions[group].add(offset)
        expected[group, offset] = key
    columns = SPECTRUM_COLUMNS + LABEL_COLUMNS + (FORMULA_COLUMN,)
    rows = dict(parquet_rows(path, columns, positions=positions))
    if rows.keys() != expected.keys() or any(rows[pos]["inchikey14"] != key for pos, key in expected.items()):
        raise ValueError("Hold-out rows do not match. Use the same train.parquet used for training.")
    return list(rows.items())


def score_candidates(candidates, key, formula):
    return {
        "top1": bool(candidates) and candidates[0]["key"] == key,
        "recovered": any(candidate["key"] == key for candidate in candidates),
        "has_formula_match": any(candidate["formula"] == formula for candidate in candidates),
        "valid_samples": sum(candidate["count"] for candidate in candidates),
        "candidates": candidates,
    }


def evaluate(args):
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported encoder checkpoint version {checkpoint['version']}")
    if args.data is None:
        args.data = Path(checkpoint["training"]["data"])
    heldout = load_heldout(args.data, checkpoint["test_spectra"])
    model = SpectrumEncoder(**checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print(f"Loading the existing MS-GPT decoder on {device}.", flush=True)
    decoder = MSGPTDecoder(device)
    eligible = [(pos, row) for pos, row in heldout if decoder.supports_formula(row[FORMULA_COLUMN])]
    if not eligible:
        raise ValueError("No held-out spectra have an MS-GPT-compatible provided formula")
    random.Random(args.seed).shuffle(eligible)
    selected = eligible if args.molecules is None else eligible[:args.molecules]
    threshold = checkpoint["threshold"]
    results, predicted_bits, true_bits = [], [], []
    for index, (position, row) in enumerate(selected, start=1):
        key = row["inchikey14"]
        formula = row[FORMULA_COLUMN]
        features = spectrum_features(row, checkpoint["preprocessing"])
        with torch.inference_mode():
            logits = model(*(value.to(device) for value in feature_batch([features])))
            predicted = (logits.sigmoid().squeeze(0) >= threshold).cpu().numpy()

        pair_seed = args.seed + index - 1
        torch.manual_seed(pair_seed)
        predicted_candidates = decoder.generate(predicted, formula, args.samples)

        oracle = MORGAN.GetFingerprintAsNumPy(achiral_molecule(row["normalized_smiles"]))
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
        "evaluated_molecules": len(results), "test_molecules": len(heldout),
        "unsupported_or_missing_formulas": len(heldout) - len(eligible),
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
    print(json.dumps(summary, indent=2))
    print(f"Saved candidate SMILES to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Same train.parquet used to train (default: saved data path)")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "encoder.pt")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "candidates.json")
    parser.add_argument("--molecules", type=int, default=4, help="Number of eligible test molecules, 0 for all")
    parser.add_argument("--samples", type=int, default=4, help="Generation attempts per molecule in each arm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.molecules < 0 or args.samples < 1 or args.threads < 1:
        parser.error("--molecules must be nonnegative, and --samples and --threads must be positive")
    if args.molecules == 0:
        args.molecules = None
    try:
        evaluate(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
