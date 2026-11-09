"""Set Transformer with metadata-conditioned pooling by learned attention."""

import torch
from torch import nn

from SMILES_decoder_module.fingerprint import FP_BITS


class SpectrumEncoder(nn.Module):
    def __init__(self, *, adducts, modes, layers=4, width=256, heads=8, dropout=0.1):
        super().__init__()
        self.peaks = nn.Linear(3, width)
        self.adduct = nn.Embedding(adducts, 16)
        self.mode = nn.Embedding(modes, 8)
        self.metadata = nn.Sequential(nn.Linear(4 + 16 + 8, width), nn.GELU())
        block = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.seed = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.seed, std=0.02)
        self.pool = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, FP_BITS))
        nn.init.constant_(self.output[-1].bias, -4.0)

    def forward(self, peaks, padding, numeric, categorical):
        meta = self.metadata(torch.cat((
            numeric, self.adduct(categorical[:, 0]), self.mode(categorical[:, 1]),
        ), dim=-1))
        tokens = self.blocks(self.peaks(peaks), src_key_padding_mask=padding)
        query = self.seed + meta.unsqueeze(1)
        pooled, _ = self.pool(query, tokens, tokens, key_padding_mask=padding, need_weights=False)
        return self.output((query + pooled).squeeze(1))
