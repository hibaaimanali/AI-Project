# CASMI spectrum encoder

The spectrum encoder evaluation uses the pretrained Coley Group MIST MassSpecGym encoder. MIST predicts 4,096 radius-2 Morgan fingerprint probabilities from an MS/MS spectrum and its provided CASMI molecular formula. MS-GPT decodes from the thresholded fingerprint and the supplied formula. The separate notebook evaluation uses reference fingerprints directly.

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

## Oracle fingerprint notebook evaluation

`Pipeline.ipynb` evaluates the first 0.1% of labeled rows, rounded down. It derives Morgan fingerprints from the reference SMILES and passes them to MS-GPT with the supplied formulas. The oracle run requires the CASMI data and MS-GPT assets. It does not load MIST or read spectrum peaks.

SQLite stores a single candidate pool per exact fingerprint and formula pair, so rows with identical inputs share the same draws. Each stage resumes from its last committed batch. `DECODER_BATCH_SIZE` controls the number of unique inputs in one CUDA graph batch. Changing the batch size changes which draws each input receives.

The decoder replays CUDA graphs with a shared key/value cache. It starts by attending to a 32-token prefix of the cache and grows that prefix as needed, up to the model's original sequence limit. It also reuses the projected fingerprint and formula context across samples and tokens.

For staged runs with many samples per input, `ranking_workers` overlaps CPU chemistry normalization with GPU generation. It dispatches each distinct SAFE string once per batch and counts all its occurrences when ranking. Run from a Python script with the call under `if __name__ == "__main__":` when using workers. The default of zero runs normalization in the calling process.

For a two-arm MIST and oracle comparison, call `run_pipeline` with the MIST checkpoint and `staged=True`. Pass `encode_only=True` to stage the inputs and count unique pairs, then rerun with `encode_only=False` to decode and score.
