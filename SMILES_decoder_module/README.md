# MS-GPT decoder smoke test

This runs the downloaded MS-GPT decoder on a known molecule. It computes a radius-2 Morgan fingerprint and molecular formula with RDKit, samples SAFE sequences from the checkpoint, and decodes them into candidate SMILES. It reports the number of valid samples and whether any candidate recovers the reference connectivity. The formula and fingerprint come from the reference structure, so this measures decoder behavior in isolation.

## Run in PowerShell

From the project root, use the shared `.venv`. The first command uses acetaminophen. The second uses a labeled structure from CASMI's `train.parquet`.

```powershell
& .\.venv\Scripts\python.exe -X utf8 .\SMILES_decoder_module\generate.py
& .\.venv\Scripts\python.exe -X utf8 .\SMILES_decoder_module\generate.py --smiles 'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21' --samples 8
```

## Recreate the environment

From the project root, use Python 3.14 and Git. Place the downloaded `ms_gpt_mlm.pt` and its `mlm_pretraining.yaml` in `SMILES_decoder_module/MS-GPT/`.

```powershell
git clone https://github.com/VIKI623/MS-GPT.git .\SMILES_decoder_module\.upstream
git -C .\SMILES_decoder_module\.upstream checkout ec032f87dcaa54c6380176c2f959e11335440ae5
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r .\SMILES_decoder_module\requirements.txt
New-Item -ItemType Directory -Path .\SMILES_decoder_module\.tokenizer -Force | Out-Null
Invoke-WebRequest -Uri 'https://huggingface.co/datamol-io/safe-gpt/resolve/3d5fa09/tokenizer.json' -OutFile .\SMILES_decoder_module\.tokenizer\tokenizer.json
```

The CPU environment is sufficient for this test. With a CUDA-enabled PyTorch installation, `--device auto` uses the GPU when available. The script uses the upstream architecture and tokenizer loader and loads the checkpoint with `weights_only=True`.
