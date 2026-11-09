"""Fingerprint statistics with equal weight per representative molecule."""

import numpy as np


def fingerprint_metrics(predicted, target):
    predicted = np.atleast_2d(predicted).astype(bool)
    target = np.atleast_2d(target).astype(bool)
    overlap = (predicted & target).sum(axis=1)
    union = (predicted | target).sum(axis=1)
    active = predicted.sum(axis=1)
    true_active = target.sum(axis=1)
    return {
        "tanimoto": float((overlap / union.clip(min=1)).mean()),
        "precision": float((overlap / active.clip(min=1)).mean()),
        "recall": float((overlap / true_active.clip(min=1)).mean()),
        "predicted_bits": float(active.mean()),
        "true_bits": float(true_active.mean()),
    }
