# CASMI spectrum encoder

The default evaluation pipeline uses the pretrained Coley Group MIST MassSpecGym encoder. MIST predicts 4,096 radius-2 Morgan fingerprint probabilities from an MS/MS spectrum and its provided CASMI molecular formula. The existing MS-GPT decoder receives the thresholded fingerprint and the same formula. The evaluation also decodes from the reference structure's fingerprint for comparison.

## Set up MIST

Run from the project root in PowerShell with the [shared Python environment and decoder assets](../SMILES_decoder_module/README.md) set up.

```powershell
git clone https://github.com/coleygroup/MIST.git .\spectrum_encoder_module\artifacts\mist-upstream
git -C .\spectrum_encoder_module\artifacts\mist-upstream checkout 48e339be6535546c87ba4f6b81a0f787c2e4ac40
& .\.venv\Scripts\python.exe -m pip install -e .\spectrum_encoder_module\artifacts\mist-upstream
```

Download [mist_msg_noaug.ckpt](https://zenodo.org/records/23303289/files/mist_msg_noaug.ckpt?download=1) into the project root or `spectrum_encoder_module/artifacts/`. Use `--checkpoint` for another location.

If the labeled CASMI Parquet has yet to be extracted, extract it once. The file is about 3 GB.

```powershell
& .\.venv\Scripts\python.exe -X utf8 -c "from zipfile import ZipFile; ZipFile('enveda-CASMI26-molecule-id-mass-spectra.zip').extract('train.parquet', 'spectrum_encoder_module/artifacts')"
```

## Run the small CPU check

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m spectrum_encoder_module.evaluate --device cpu --molecules 4 --samples 4
```

The defaults scan the first 8,192 Parquet rows and sample up to 512 unique molecules. Groups connected by Morgan Tanimoto of at least 0.65 stay together across partitions. The run uses one seeded representative spectrum per validation or test molecule. It selects the fingerprint threshold on the validation partition by mean Tanimoto, then applies it to the test partition. The pretrained weights are used as released.

The adapter filters peaks above precursor m/z + 1, square-root scales base-peak-normalized intensities, and uses MIST's subformula assignment at 20 ppm on up to 60 peaks. It passes up to 10 assigned fragments and the precursor formula token to MIST. Unsupported adducts, unsupported formulas, and spectra without usable fragment assignments are counted in the report. The checkpoint's adduct vocabulary covers a subset of CASMI's positive ions.

The run writes `spectrum_encoder_module/artifacts/mist_smoke_candidates.json` with encoder coverage, threshold sweep, test fingerprint metrics, and ranked decoder candidates. The test partition is held out from CASMI threshold calibration. Compound overlap with MassSpecGym pretraining has not been checked.

The earlier peak-transformer training entry point remains available as `python -m spectrum_encoder_module.train` for baseline experiments. The evaluation entry point now takes a released MIST checkpoint.
