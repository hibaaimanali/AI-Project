# CASMI spectrum encoder

The encoder predicts a 4,096-bit molecular fingerprint from an MS/MS spectrum. The integration uses the provided CASMI molecular formula to condition the existing MS-GPT checkpoint. This milestone uses labeled `train.parquet`. Formula prediction and the unlabeled competition data are deferred.

## Run the small CPU check

Run these commands in PowerShell from the project root. They use the shared `.venv` and the [existing decoder assets](../SMILES_decoder_module/README.md).

Extract the labeled member once. The extracted Parquet is about 3 GB, and both commands project the columns they use.

```powershell
& .\.venv\Scripts\python.exe -X utf8 -c "from zipfile import ZipFile; ZipFile('enveda-CASMI26-molecule-id-mass-spectra.zip').extract('train.parquet', 'spectrum_encoder_module/artifacts')"
```

Train a two-layer, width-64 model on up to 512 molecules.

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m spectrum_encoder_module.train `
  --device cpu --max-rows 8192 --max-molecules 512 --spectra-per-molecule 3 `
  --max-peaks 32 --layers 2 --width 64 --heads 4 --epochs 8 `
  --checkpoint .\spectrum_encoder_module\artifacts\smoke_encoder.pt
```

The default architecture has four layers, width 256 and eight heads. Run `-m spectrum_encoder_module.train --help` to see the row, molecule, peak and batch limits. The checkpoint stores the model weights, architecture, preprocessing settings and vocabularies, selected threshold, split membership, and Parquet positions of the test spectra. The threshold sweep and fingerprint metrics are also written to `smoke_encoder.metrics.json`.

Run four generation attempts per molecule per arm on four held-out molecules.

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m spectrum_encoder_module.evaluate `
  --device cpu --checkpoint .\spectrum_encoder_module\artifacts\smoke_encoder.pt `
  --molecules 4 --samples 4 --output .\spectrum_encoder_module\artifacts\smoke_candidates.json
```

The evaluator reloads the same Parquet file using the saved test positions. Use `--data` if the file moved. The JSON includes ranked SMILES for both arms, their sample frequencies and the per-molecule scores. The decoder receives the same provided formula, seed and sampling budget for each pair. Each arm reports how many molecules it recovers by CASMI's `inchikey14`, at rank one and anywhere in its ranked candidates. Test molecules without a supported formula are counted and skipped. The artifacts directory is ignored by Git.

## Data and model

- Each peak contributes fragment m/z, base-peak-normalized intensity, and signed precursor-minus-fragment m/z. The model keeps the most intense peaks and masks padding. A learned query pools the peak set with precursor m/z, adduct and ionization mode embeddings, and the mean available collision energy in eV. Numeric fields have presence indicators. Categorical fields distinguish missing values from categories unseen in training.
- The model emits 4,096 logits and trains with binary cross-entropy. Target fingerprints come from `normalized_smiles` using the decoder's RDKit radius-2 Morgan bit vector after removing stereochemistry. During inference, the encoder sees spectrum and acquisition fields. Its logits go through sigmoid and the saved validation threshold before decoding. Both decoder arms use upstream `formula_str_to_vec` on the provided `molecular_formula`. The oracle arm uses the reference fingerprint.
- The row limit selects a prefix of the Parquet file, and a seeded sample of `inchikey14` groups is taken from that prefix. Each group stays together. Groups connected by Morgan Tanimoto of at least 0.65 are assigned together, including transitive links and all observed SMILES variants of the selected keys. The exact pairwise check is quadratic in the number of distinct fingerprints. Training can use several spectra per key. Validation and test use one seeded representative spectrum per key.

## Observed CPU results

The small run used 512 unique molecules from the first 8,192 rows. The 501 structural groups produced 358 train molecules (532 spectra), 77 validation molecules, and 77 test molecules. The maximum cross-split Morgan Tanimoto was 0.64865.

Epoch 4 had the lowest validation BCE (0.04313). Threshold 0.20 maximized mean validation Tanimoto over the saved sweep. The validation targets averaged 42.1 active bits.

| Threshold | Mean predicted bits | Mean bit recall | Mean Tanimoto |
| ---: | ---: | ---: | ---: |
| 0.05 | 133.1 | 0.483 | 0.128 |
| 0.10 | 60.3 | 0.369 | 0.174 |
| 0.15 | 36.4 | 0.322 | 0.200 |
| **0.20** | **25.4** | **0.280** | **0.204** |
| 0.30 | 20.4 | 0.249 | 0.194 |
| 0.50 | 10.5 | 0.175 | 0.159 |

On the 77 test molecules, the selected threshold gave **0.202 mean Tanimoto**, 0.457 mean bit precision, and 0.278 mean bit recall. It produced 25.3 active bits on average against 44.3 target bits. At 0.50, mean test Tanimoto was 0.150 and the mean predicted fingerprint had 10.4 active bits.

The paired decoder run used four test molecules and four attempts per molecule in each arm. All 77 test molecules had supported formulas. The four evaluated encoder fingerprints averaged 0.147 Tanimoto.

| Fingerprint input | Valid decoded samples | Molecules with a formula match | Rank-one recovery | Recovery anywhere |
| --- | ---: | ---: | ---: | ---: |
| Oracle | 16/16 | 3/4 | 3/4 | 3/4 |
| Predicted | 4/16 | 0/4 | 0/4 | 0/4 |

The original acetaminophen smoke test recovered the reference in 8/8 attempts. These results cover a capped row prefix and a four-molecule decoder check. A broader evaluation would be needed to estimate recovery on the labeled dataset.
