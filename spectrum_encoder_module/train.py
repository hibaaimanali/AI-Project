"""Train a spectrum-to-fingerprint model on a bounded labeled CASMI subset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from SMILES_decoder_module.generate import resolve_device

from . import CHECKPOINT_VERSION
from .data import (
    ARTIFACTS, DEFAULT_DATA, CATEGORY_COLUMNS, collect_molecules, feature_batch,
    preprocessing, spectrum_features, split_molecules,
)
from .metrics import fingerprint_metrics
from .model import SpectrumEncoder


THRESHOLDS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7)


def collate(examples):
    features, targets = zip(*examples)
    return feature_batch(features), torch.from_numpy(np.stack(targets)).float()


@torch.inference_mode()
def predict(model, loader, loss, device):
    model.eval()
    probabilities, targets = [], []
    total_loss = 0.0
    for features, labels in loader:
        logits = model(*(value.to(device) for value in features))
        total_loss += loss(logits, labels.to(device)).item() * len(labels)
        probabilities.append(logits.sigmoid().cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets), total_loss / len(loader.dataset)


def train(args):
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    print(f"Reading at most {args.max_rows:,} labeled rows and training on {device}.", flush=True)
    spectra, fingerprints = collect_molecules(
        args.data, max_rows=args.max_rows, max_molecules=args.max_molecules,
        spectra_per_molecule=args.spectra_per_molecule, max_peaks=args.max_peaks, seed=args.seed,
    )
    split, group_count, largest_group = split_molecules(
        fingerprints, args.similarity, args.validation_fraction, args.test_fraction, args.seed,
    )
    settings = preprocessing(
        (row for key in split["train"] for _, row, _, _ in spectra[key]), args.max_peaks,
    )
    loaders, counts = {}, {}
    for name, keys in split.items():
        examples = []
        for key in keys:
            rows = spectra[key] if name == "train" else spectra[key][:1]
            examples.extend((spectrum_features(row, settings, peaks), target) for _, row, peaks, target in rows)
        counts[name] = {"molecules": len(keys), "spectra": len(examples)}
        loaders[name] = DataLoader(
            examples, batch_size=args.batch_size, shuffle=name == "train", collate_fn=collate,
        )
    print(f"{len(spectra)} molecules in {group_count} structural groups. {counts}", flush=True)

    adducts, modes = (len(settings["vocab"][name]) + 2 for name in CATEGORY_COLUMNS)
    config = dict(adducts=adducts, modes=modes, layers=args.layers, width=args.width, heads=args.heads)
    model = SpectrumEncoder(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(value.to(device) for value in features))
            batch_loss = loss(logits, targets.to(device))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() * len(targets)

        probabilities, target, validation_loss = predict(model, loaders["validation"], loss, device)
        print(f"Epoch {epoch} | train BCE {total_loss / len(loaders['train'].dataset):.5f} | "
              f"validation BCE {validation_loss:.5f}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation = [
                {"threshold": threshold, **fingerprint_metrics(probabilities >= threshold, target)}
                for threshold in THRESHOLDS
            ]
    if not np.isfinite(best_loss):
        raise ValueError("No epoch produced a finite validation loss")
    # Ties go to the higher threshold, which produces the sparser conditioning vector.
    threshold = max(best_validation, key=lambda row: (row["tanimoto"], row["threshold"]))["threshold"]
    model.load_state_dict(best_weights)
    probabilities, target, _ = predict(model, loaders["test"], loss, device)
    report = {
        "unique_molecules": len(spectra), "structural_groups": group_count, "largest_group": largest_group,
        "splits": counts, "group_similarity_cutoff": args.similarity,
        "best_epoch": best_epoch, "validation_bce": best_loss, "selected_threshold": threshold,
        "validation_threshold_sweep": best_validation,
        "test": fingerprint_metrics(probabilities >= threshold, target),
        "test_at_0.5": fingerprint_metrics(probabilities >= 0.5, target),
    }
    checkpoint = {
        "version": CHECKPOINT_VERSION, "model": config, "state_dict": best_weights,
        "preprocessing": settings, "threshold": threshold, "split_keys": split,
        "test_spectra": {key: list(spectra[key][0][0]) for key in split["test"]},
        "training": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "metrics": report,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    metrics_file = args.checkpoint.with_suffix(".metrics.json")
    metrics_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected threshold {threshold}. Test fingerprint metrics {report['test']}")
    print(f"Saved {args.checkpoint} and {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Extracted CASMI train.parquet")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS / "encoder.pt")
    parser.add_argument("--max-rows", type=int, default=8192, help="Maximum Parquet rows scanned from the start")
    parser.add_argument("--max-molecules", type=int, default=768)
    parser.add_argument("--spectra-per-molecule", type=int, default=4, help="Maximum repeats used for training")
    parser.add_argument("--max-peaks", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--similarity", type=float, default=0.65, help="Morgan Tanimoto cutoff for connecting analogs")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for name in ("max_rows", "max_molecules", "spectra_per_molecule", "max_peaks", "layers",
                 "width", "heads", "epochs", "batch_size", "threads", "learning_rate"):
        if not getattr(args, name) > 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.width % args.heads:
        parser.error("--width must be divisible by --heads")
    if not (0 < args.similarity <= 1):
        parser.error("--similarity must be in (0, 1]")
    if not (min(args.validation_fraction, args.test_fraction) > 0
            and args.validation_fraction + args.test_fraction < 1):
        parser.error("Hold-out fractions must be positive and sum to less than one")
    if args.checkpoint.suffix != ".pt":
        parser.error("--checkpoint must use the .pt extension")
    try:
        train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
