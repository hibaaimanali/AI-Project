---
license: apache-2.0
tags:
- chemistry
- mass-spectrometry
- molecule-generation
- conditional-language-model
- ms-gpt
- arxiv:2607.23607
base_model: nielsr/MS-GPT-SAFE-GPT-Base
---

# MS-GPT conditional molecule-language model

This repository contains the 150.1M-parameter conditional molecule-language
model checkpoint released with **MS-GPT: Rethinking MS/MS De Novo Structure
Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language
Model**.

The model is pretrained on a molecule-only corpus with oracle fingerprints and
formulas and is shared across the NPLIB1 and MassSpecGym experiments. The
released training configuration is included as `mlm_pretraining.yaml`.

Use this checkpoint with the implementation in the
[original MS-GPT repository](https://github.com/VIKI623/MS-GPT).

## Provenance and attribution

These files are mirrored from the authors' released
[MS-GPT asset bundle](https://pan.sjtu.edu.cn/web/share/8f63823e4d82280636e537d1c789128d)
with their explicit authorization.

- Original repository: https://github.com/VIKI623/MS-GPT
- Author's Hugging Face profile: https://huggingface.co/viki623
- Paper: https://huggingface.co/papers/2607.23607
- arXiv: https://arxiv.org/abs/2607.23607

The authors confirmed that this MS-GPT checkpoint may be released under
Apache-2.0 in
[GitHub issue #1](https://github.com/VIKI623/MS-GPT/issues/1#issuecomment-5130759988).
