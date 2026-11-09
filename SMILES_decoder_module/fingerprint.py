"""Fingerprint convention shared by the MS-GPT smoke test and spectrum encoder."""

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


FP_BITS = 4096
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def achiral_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Expected a valid nonempty SMILES")
    Chem.RemoveStereochemistry(mol)
    return mol
