"""Projected CASMI reads, spectral features, and molecule-level partitioning."""

from collections import defaultdict
from itertools import islice
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs
import torch
from torch.nn.utils.rnn import pad_sequence

from SMILES_decoder_module.fingerprint import FP_BITS, MORGAN, achiral_molecule


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_DATA = ARTIFACTS / "train.parquet"
PEAK_COLUMNS = ("ms2_mzs", "ms2_normalized_intensities")
NUMERIC_COLUMNS = ("precursor_mz", "collision_energy_ev")
CATEGORY_COLUMNS = ("adduct", "ionization_mode")
LABEL_COLUMNS = ("inchikey14", "normalized_smiles")
SPECTRUM_COLUMNS = PEAK_COLUMNS + NUMERIC_COLUMNS + CATEGORY_COLUMNS
FORMULA_COLUMN = "molecular_formula"


def parquet_rows(path, columns, *, max_rows=None, positions=None):
    """Yield ((row group, offset), row) for a bounded scan or saved positions."""
    with pq.ParquetFile(path) as source:
        names = set(source.schema_arrow.names)
        missing = set(columns) - names - set(CATEGORY_COLUMNS + NUMERIC_COLUMNS)
        if missing:
            raise ValueError(f"Missing CASMI columns: {', '.join(sorted(missing))}")
        available = [column for column in columns if column in names]
        groups = sorted(positions) if positions is not None else range(source.num_row_groups)
        remaining = max_rows
        for group in groups:
            limit = max(positions[group]) + 1 if positions is not None else remaining
            offset = 0
            batches = source.iter_batches(row_groups=[group], columns=available, batch_size=512)
            for batch in batches:
                if limit is not None:
                    batch = batch.slice(0, limit - offset)
                if positions is None:
                    for index, row in enumerate(batch.to_pylist(), start=offset):
                        yield (group, index), row
                else:
                    indices = sorted(index for index in positions[group] if offset <= index < offset + len(batch))
                    if indices:
                        selected = batch.take([index - offset for index in indices])
                        for index, row in zip(indices, selected.to_pylist()):
                            yield (group, index), row
                offset += len(batch)
                if limit is not None and offset == limit:
                    break
            if remaining is not None:
                remaining -= offset
                if remaining == 0:
                    break


def clean_peaks(row, max_peaks):
    mzs = np.asarray(row[PEAK_COLUMNS[0]], dtype=np.float64)
    intensity = np.asarray(row[PEAK_COLUMNS[1]], dtype=np.float64)
    if mzs.ndim != 1 or mzs.shape != intensity.shape:
        return None
    valid = np.isfinite(mzs) & (mzs > 0) & np.isfinite(intensity) & (intensity > 0)
    mzs, intensity = mzs[valid], intensity[valid]
    if not len(mzs):
        return None
    keep = np.lexsort((mzs, -intensity))[:max_peaks]
    return mzs[keep], intensity[keep] / intensity[keep[0]]


def category(row, name):
    return (row.get(name) or "").strip()


def preprocessing(train_rows, max_peaks):
    categories = {name: set() for name in CATEGORY_COLUMNS}
    for row in train_rows:
        for name, values in categories.items():
            values.add(category(row, name))
    vocab = {
        name: {value: index for index, value in enumerate(sorted(values - {""}), start=2)}
        for name, values in categories.items()
    }
    return {"max_peaks": max_peaks, "mz_scale": 1000.0, "energy_scale": 100.0, "vocab": vocab}


def spectrum_features(row, settings, peaks=None):
    if peaks is None:
        peaks = clean_peaks(row, settings["max_peaks"])
    if peaks is None:
        raise ValueError("Expected a spectrum with at least one valid peak")
    mzs, intensity = peaks
    precursor = row.get("precursor_mz")
    has_precursor = precursor is not None and np.isfinite(precursor) and precursor > 0
    mz_scale = settings["mz_scale"]
    differences = (precursor - mzs) / mz_scale if has_precursor else np.zeros_like(mzs)
    peak_features = np.column_stack((mzs / mz_scale, intensity, differences)).astype(np.float32)

    energies = [value for value in (row.get("collision_energy_ev") or [])
                if value is not None and np.isfinite(value) and value >= 0]
    numeric = np.array([
        precursor / mz_scale if has_precursor else 0.0, has_precursor,
        np.mean(energies) / settings["energy_scale"] if energies else 0.0, bool(energies),
    ], dtype=np.float32)
    categorical = []
    for name in CATEGORY_COLUMNS:
        value = category(row, name)
        categorical.append(settings["vocab"][name].get(value, 1) if value else 0)
    return peak_features, numeric, np.array(categorical, dtype=np.int64)


def feature_batch(features):
    peaks, numeric, categorical = zip(*features)
    lengths = torch.tensor([len(peak) for peak in peaks])
    peaks = pad_sequence([torch.from_numpy(peak) for peak in peaks], batch_first=True)
    padding = torch.arange(peaks.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    return peaks, padding, torch.from_numpy(np.stack(numeric)), torch.from_numpy(np.stack(categorical))


def fingerprint_array(bit_vector):
    result = np.empty(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, result)
    return result


def collect_molecules(path, *, max_rows, max_molecules, spectra_per_molecule, max_peaks, seed):
    rows = defaultdict(list)
    for position, row in parquet_rows(path, SPECTRUM_COLUMNS + LABEL_COLUMNS, max_rows=max_rows):
        key, smiles = (row[name] for name in LABEL_COLUMNS)
        if key and smiles:
            rows[key].append((position, row))

    rng = random.Random(seed)
    keys = sorted(rows)
    rng.shuffle(keys)
    spectra, fingerprints = {}, {}
    for key in keys:
        variants = {}
        for smiles in sorted({row["normalized_smiles"] for _, row in rows[key]}):
            try:
                variants[smiles] = MORGAN.GetFingerprint(achiral_molecule(smiles))
            except ValueError:
                continue

        rng.shuffle(rows[key])
        usable = (
            (position, row, peaks, fingerprint_array(variants[row["normalized_smiles"]]))
            for position, row in rows[key]
            if row["normalized_smiles"] in variants and (peaks := clean_peaks(row, max_peaks)) is not None
        )
        selected = list(islice(usable, spectra_per_molecule))
        if selected:
            spectra[key] = selected
            fingerprints[key] = list(variants.values())
        if len(spectra) == max_molecules:
            break
    return spectra, fingerprints


def split_molecules(fingerprints, similarity, validation_fraction, test_fraction, seed):
    """Connect any pair at or above the cutoff, then split whole components."""
    parents = {key: key for key in fingerprints}

    def root(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    previous_keys, previous_fps = [], []
    for key, variants in fingerprints.items():
        for fingerprint in variants:
            scores = DataStructs.BulkTanimotoSimilarity(fingerprint, previous_fps)
            for index in np.flatnonzero(np.asarray(scores) >= similarity):
                parents[root(previous_keys[index])] = root(key)
        previous_keys.extend([key] * len(variants))
        previous_fps.extend(variants)

    components = defaultdict(list)
    for key in fingerprints:
        components[root(key)].append(key)
    groups = list(components.values())
    if len(groups) < 3:
        raise ValueError("Need at least three structural groups. Scan more rows or raise --similarity.")
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    fractions = {"train": 1 - validation_fraction - test_fraction, "validation": validation_fraction,
                 "test": test_fraction}
    split = {name: [] for name in fractions}
    for index, group in enumerate(groups):
        empty = [name for name in split if not split[name]]
        choices = empty if len(groups) - index == len(empty) else split
        name = max(choices, key=lambda part: len(fingerprints) * fractions[part] - len(split[part]))
        split[name].extend(group)
    return split, len(groups), len(groups[0])
