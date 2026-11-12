import argparse
from collections import Counter
from itertools import batched
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

if __package__:
    from .fingerprint import FP_BITS, MORGAN, achiral_molecule
else:
    from fingerprint import FP_BITS, MORGAN, achiral_molecule


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "MS-GPT"
EXAMPLE_SMILES = "CC(=O)Nc1ccc(O)cc1"


def resolve_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class MSGPTDecoder:
    """Load the existing MS-GPT assets once and decode binary fingerprints."""

    def __init__(self, device):
        source = HERE / ".upstream" / "src"
        tokenizer_dir = HERE / ".tokenizer"
        checkpoint = MODEL_DIR / "ms_gpt_mlm.pt"
        config_path = MODEL_DIR / "mlm_pretraining.yaml"
        for required in (source / "ms_gpt", tokenizer_dir / "tokenizer.json", checkpoint, config_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing {required}. See SMILES_decoder_module/README.md.")

        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import yaml
        from ms_gpt.data.mlm_pretraining_dataset import ELEMENTS, formula_str_to_vec, load_molecule_tokenizer
        from ms_gpt.models.conditional_mlm import (
            ConditionalMoleculeLanguageModel,
            normalize_mlm_checkpoint_state_dict,
        )
        from ms_gpt.pipeline.postprocessing import decode_safe
        from ms_gpt.utils.mol_repair import formula_counts, molecule_data

        self.device = device
        self.formula_counts = formula_counts
        self.elements = set(ELEMENTS)
        self.formula_str_to_vec = formula_str_to_vec
        self.decode_safe = decode_safe
        self.molecule_data = molecule_data
        model_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
        if model_config["fp_bits"] != FP_BITS:
            raise ValueError(f"Expected MS-GPT to use {FP_BITS}-bit fingerprints")
        self.max_len = model_config["max_seq_len"]
        self.tokenizer = load_molecule_tokenizer(str(tokenizer_dir))
        model = ConditionalMoleculeLanguageModel(vocab_size=len(self.tokenizer), **model_config)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(normalize_mlm_checkpoint_state_dict(weights))
        del weights
        self.model = model.to(device).eval()
        self.generate_tokens = self.model.generate

    def enable_cached_generation(self, backend):
        from .cached_generation import CachedGeneration

        self.generate_tokens = CachedGeneration(self.model, backend)

    def supports_formula(self, formula):
        counts = self.formula_counts(formula) if formula else None
        return bool(counts) and counts.keys() <= self.elements

    @torch.inference_mode()
    def generate(self, fingerprint, formula, samples):
        return self.generate_batch(torch.as_tensor(fingerprint).unsqueeze(0), [formula], samples)[0]

    @torch.inference_mode()
    def generate_batch(self, fingerprints, formulas, samples):
        """Sample each fingerprint/formula pair and rank its candidates independently."""
        if samples < 1:
            raise ValueError("samples must be positive")
        for formula in formulas:
            if not self.supports_formula(formula):
                raise ValueError(f"The model does not support this molecular formula: {formula}")
        fp_tensor = torch.as_tensor(fingerprints, dtype=torch.float32, device=self.device)
        if (not formulas or fp_tensor.shape != (len(formulas), FP_BITS)
                or not ((fp_tensor == 0) | (fp_tensor == 1)).all()):
            raise ValueError(f"Expected one binary {FP_BITS}-bit fingerprint per formula")
        formula_tensor = torch.as_tensor(
            np.stack([self.formula_str_to_vec(formula) for formula in formulas]), device=self.device,
        )

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            token_batches = self.generate_tokens(
                fingerprint=fp_tensor,
                formula_vec=formula_tensor,
                max_len=self.max_len,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_samples=samples,
            ).cpu().tolist()

        return [
            self._rank_samples(token_ids, formula)
            for token_ids, formula in zip(batched(token_batches, samples, strict=True), formulas, strict=True)
        ]

    def _rank_samples(self, token_batches, formula):
        frequencies = Counter()
        candidates = {}
        for token_ids in token_batches:
            if self.tokenizer.eos_token_id not in token_ids:
                continue
            tokens = token_ids[:token_ids.index(self.tokenizer.eos_token_id)]
            sequence = self.tokenizer.decode(tokens, skip_special_tokens=True)
            smiles = self.decode_safe(sequence) if sequence else None
            candidate = self.molecule_data(smiles) if smiles else None
            if candidate is None:
                continue
            key = candidate["key"]
            frequencies[key] += 1
            if key not in candidates or candidate["formula"] == formula:
                candidates[key] = {field: candidate[field] for field in ("key", "formula", "smiles")}

        ranked = sorted(candidates, key=lambda key: (candidates[key]["formula"] != formula, -frequencies[key], key))
        return [dict(candidates[key], count=frequencies[key]) for key in ranked]


def main():
    parser = argparse.ArgumentParser(
        description="Generate SMILES with MS-GPT using a fingerprint and formula from a known structure."
    )
    parser.add_argument("--smiles", default=EXAMPLE_SMILES, help="Reference SMILES (default: acetaminophen)")
    parser.add_argument("--samples", type=int, default=8, help="Number of generation attempts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    try:
        reference_mol = achiral_molecule(args.smiles)
        device = resolve_device(args.device)
        if device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.manual_seed(args.seed)
        decoder = MSGPTDecoder(device)
        reference = decoder.molecule_data(args.smiles)
        if reference is None:
            parser.error("Could not normalize the reference structure")
        formula = rdMolDescriptors.CalcMolFormula(reference_mol)
        if not decoder.supports_formula(formula):
            parser.error(f"The model does not support this molecular formula: {formula}")
        fingerprint = MORGAN.GetFingerprintAsNumPy(reference_mol)
        print(f"Generating on {device} from {Chem.MolToSmiles(reference_mol)} ({formula})", flush=True)
        candidates = decoder.generate(fingerprint, formula, args.samples)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    decoded = sum(candidate["count"] for candidate in candidates)
    print(f"Decoded {decoded}/{args.samples} samples into {len(candidates)} unique structures.")
    recovered = any(candidate["key"] == reference["key"] for candidate in candidates)
    print(f"Reference connectivity recovered: {'yes' if recovered else 'no'}")
    for rank, candidate in enumerate(candidates, start=1):
        print(f"{rank:>2}. {candidate['smiles']}  ({candidate['formula']}, {candidate['count']} samples)")
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
