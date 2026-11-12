"""Adapt CASMI peak arrays to the Coley Group MIST inference model."""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from mist.data.featurizers import PeakFormula, assign_subformulae
from mist.models.scaled_mist import ScaledMIST
from mist.utils import chem_utils

from SMILES_decoder_module.fingerprint import FP_BITS
from .data import FORMULA_COLUMN, PEAK_COLUMNS


MAX_ASSIGNMENT_PEAKS = 60
MAX_MODEL_FRAGMENTS = 10
SUBFORMULA_PPM = 20.0
ADDUCT_ALIASES = {"[M+NH4]+": "[M+H3N+H]+"}


def spectrum_to_mist(row):
    """Return MIST tensors and a skip reason when preprocessing cannot produce them."""
    adduct = row.get("adduct")
    adduct = ADDUCT_ALIASES.get(adduct, adduct)
    if adduct not in chem_utils.ion_to_idx:
        return None, "unsupported_adduct"

    formula = row.get(FORMULA_COLUMN)
    try:
        precursor = chem_utils.formula_to_dense(formula or "")
    except ValueError:
        return None, "unsupported_formula"
    if not precursor.any():
        return None, "unsupported_formula"

    mzs, intensities = (np.asarray(row[column], dtype=np.float64) for column in PEAK_COLUMNS)
    if mzs.ndim != 1 or mzs.shape != intensities.shape:
        return None, "invalid_peaks"
    keep = np.isfinite(mzs) & (mzs > 0) & np.isfinite(intensities) & (intensities > 0)
    precursor_mz = row.get("precursor_mz")
    if precursor_mz is not None and np.isfinite(precursor_mz) and precursor_mz > 0:
        keep &= mzs <= precursor_mz + 1
    mzs, intensities = mzs[keep], intensities[keep]
    if not len(mzs):
        return None, "invalid_peaks"

    order = np.argsort(-intensities, kind="stable")[:MAX_ASSIGNMENT_PEAKS]
    intensities = np.sqrt(intensities[order] / intensities[order[0]])
    keep = intensities > 0.001
    assigned = assign_subformulae({
        "cand_form": formula,
        "cand_ion": adduct,
        "output_tbl": {"mz": mzs[order][keep], "ms2_inten": intensities[keep]},
    }, SUBFORMULA_PPM)["output_tbl"]
    if assigned is None:
        return None, "no_assigned_fragments"

    fragments = [
        (fragment, intensity)
        for fragment, intensity in zip(assigned["formula"], assigned["ms2_inten"], strict=True)
        if fragment
    ][:MAX_MODEL_FRAGMENTS]
    if not fragments:
        return None, "no_assigned_fragments"

    forms = [chem_utils.formula_to_dense(fragment) for fragment, _ in fragments]
    forms.append(precursor)
    types = [PeakFormula.PEAK_TYPES["fragment"]] * len(fragments) + [PeakFormula.PEAK_TYPES["cls"]]
    instrument = row.get("instrument_type")
    if instrument == "timsTOF":
        instrument = "QTOF"
    return {
        "form_vec": torch.from_numpy(np.asarray(forms, dtype=np.float32)),
        "ion_vec": torch.full((len(forms),), chem_utils.get_ion_idx(adduct), dtype=torch.long),
        "intens": torch.tensor([value for _, value in fragments] + [1.0], dtype=torch.float32),
        "types": torch.tensor(types),
        "num_peaks": torch.tensor(len(forms)),
        "instruments": torch.tensor(chem_utils.get_instr_idx(instrument)),
    }, None


class MistEncoder:
    def __init__(self, checkpoint, device):
        released = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "ema_model" not in released or "model" not in released.get("config", {}):
            raise ValueError("Expected a Coley Group MIST checkpoint with model config and EMA weights")
        model = ScaledMIST(**released["config"]["model"])
        model.load_state_dict(released["ema_model"], strict=True)
        self.model = model.to(device).eval()
        self.device = device
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())

    @torch.inference_mode()
    def predict(self, features):
        """Return one vector of fingerprint probabilities per featurized spectrum."""
        batch = {
            name: pad_sequence([row[name] for row in features], batch_first=True)
            for name in ("form_vec", "ion_vec", "intens", "types")
        }
        for name in ("num_peaks", "instruments"):
            batch[name] = torch.stack([row[name] for row in features])
        probabilities, _ = self.model({name: value.to(self.device) for name, value in batch.items()})
        probabilities = probabilities.float().cpu().numpy()
        if probabilities.shape != (len(features), FP_BITS) or not np.isfinite(probabilities).all():
            raise RuntimeError("MIST returned invalid fingerprint predictions")
        return probabilities
