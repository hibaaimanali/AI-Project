import argparse
from collections import Counter
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "MS-GPT"
EXAMPLE_SMILES = "CC(=O)Nc1ccc(O)cc1"


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

    source = HERE / ".upstream" / "src"
    tokenizer_dir = HERE / ".tokenizer"
    checkpoint = MODEL_DIR / "ms_gpt_mlm.pt"
    config_path = MODEL_DIR / "mlm_pretraining.yaml"
    for required in (source / "ms_gpt", tokenizer_dir / "tokenizer.json", checkpoint, config_path):
        if not required.exists():
            parser.error(f"Missing {required}. See the setup steps in README.md.")

    sys.path.insert(0, str(source))
    import torch
    import yaml
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    from ms_gpt.data.mlm_pretraining_dataset import ELEMENTS, formula_str_to_vec, load_molecule_tokenizer
    from ms_gpt.models.conditional_mlm import (
        ConditionalMoleculeLanguageModel,
        normalize_mlm_checkpoint_state_dict,
    )
    from ms_gpt.pipeline.postprocessing import decode_safe
    from ms_gpt.utils.mol_repair import formula_counts, molecule_data

    reference_mol = Chem.MolFromSmiles(args.smiles)
    if reference_mol is None or reference_mol.GetNumAtoms() == 0:
        parser.error("--smiles must be a valid nonempty SMILES")
    Chem.RemoveStereochemistry(reference_mol)
    reference = molecule_data(args.smiles)
    if reference is None:
        parser.error("Could not normalize the reference structure")
    formula = rdMolDescriptors.CalcMolFormula(reference_mol)
    counts = formula_counts(formula)
    if counts is None or counts.keys() - set(ELEMENTS):
        parser.error(f"The model does not support this molecular formula: {formula}")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)

    model_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    tokenizer = load_molecule_tokenizer(str(tokenizer_dir))
    model = ConditionalMoleculeLanguageModel(vocab_size=len(tokenizer), **model_config)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(normalize_mlm_checkpoint_state_dict(weights))
    del weights
    model.to(device).eval()

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=model_config["fp_bits"])
    fingerprint = fingerprint_generator.GetFingerprintAsNumPy(reference_mol)
    fp_tensor = torch.as_tensor(fingerprint, dtype=torch.float32, device=device).unsqueeze(0)
    formula_tensor = torch.as_tensor(formula_str_to_vec(formula), device=device).unsqueeze(0)

    print(f"Generating on {device} from {Chem.MolToSmiles(reference_mol)} ({formula})", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        samples = model.generate(
            fingerprint=fp_tensor,
            formula_vec=formula_tensor,
            max_len=model_config["max_seq_len"],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_samples=args.samples,
        ).cpu().tolist()

    frequencies = Counter()
    candidates = {}
    for token_ids in samples:
        if tokenizer.eos_token_id not in token_ids:
            continue
        tokens = token_ids[:token_ids.index(tokenizer.eos_token_id)]
        sequence = tokenizer.decode(tokens, skip_special_tokens=True)
        smiles = decode_safe(sequence) if sequence else None
        candidate = molecule_data(smiles) if smiles else None
        if candidate is None:
            continue
        key = candidate["key"]
        frequencies[key] += 1
        if key not in candidates or candidate["formula"] == formula:
            candidates[key] = candidate

    print(f"Decoded {frequencies.total()}/{args.samples} samples into {len(candidates)} unique structures.")
    print(f"Reference connectivity recovered: {'yes' if reference['key'] in candidates else 'no'}")
    ranked = sorted(candidates, key=lambda key: (candidates[key]["formula"] != formula, -frequencies[key], key))
    for rank, key in enumerate(ranked, start=1):
        candidate = candidates[key]
        print(f"{rank:>2}. {candidate['smiles']}  ({candidate['formula']}, {frequencies[key]} samples)")
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
