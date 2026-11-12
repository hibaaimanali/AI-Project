"""MS-GPT sampling with shared conditioning keys and values."""

import torch
from transformers import DynamicCache


def _condition_token(layer, hidden, keys, values, context_mask):
    attention = layer.cross_attn
    molecules = keys.shape[0]
    query = attention.q_proj(layer.norm(hidden))
    query = query.reshape(molecules, -1, attention.num_heads, attention.head_dim).transpose(1, 2)
    scores = torch.matmul(query * attention.scale, keys.transpose(-2, -1))
    scores = scores.masked_fill(~context_mask[:, None, None, :], float("-inf"))
    attended = torch.matmul(scores.softmax(dim=-1), values)
    attended = attended.transpose(1, 2).reshape_as(hidden)
    conditioned = hidden + attention.out_proj(attended)
    return conditioned + layer.ffn(layer.norm2(conditioned))


class CachedGeneration:
    def __init__(self, model, backend):
        self.model = model
        self.condition_token = torch.compile(_condition_token, backend=backend, dynamic=True, fullgraph=True)

    def __call__(self, fingerprint, formula_vec, max_len, bos_token_id, eos_token_id, num_samples):
        model = self.model
        with torch.autocast("cuda", enabled=False):
            context, context_mask = model.encode_condition(fingerprint.float(), formula_vec.float())

        # Every token and sample for a molecule uses the same conditioning projections.
        context_cache = {}
        for index, layer in model.cross_attn_layers.items():
            attention = layer.cross_attn
            shape = (*context.shape[:2], attention.num_heads, attention.head_dim)
            keys = attention.k_proj(context).reshape(shape).transpose(1, 2)
            values = attention.v_proj(context).reshape(shape).transpose(1, 2)
            context_cache[int(index)] = layer, keys, values

        device = fingerprint.device
        batch_size = fingerprint.shape[0] * num_samples
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        positions = torch.arange(max_len, device=device)
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        cache = DynamicCache()

        for position in positions[:-1].split(1):
            hidden = model.transformer.wte(input_ids[:, -1:]) + model.transformer.wpe(position.unsqueeze(0))
            for index, block in enumerate(model.transformer.h):
                hidden = block(
                    hidden, past_key_values=cache, cache_position=position, use_cache=True,
                )[0]
                if index in context_cache:
                    layer, keys, values = context_cache[index]
                    hidden = self.condition_token(layer, hidden, keys, values, context_mask)

            logits = model.gpt2.lm_head(model.transformer.ln_f(hidden[:, -1])).float()
            # Match the upstream top-k and nucleus sampling policy, including ties at the cutoff.
            cutoff = torch.topk(logits, 50).values[..., -1:]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
            sorted_logits, sorted_indices = logits.sort(descending=True)
            remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > 0.95
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            logits = logits.masked_fill(remove.scatter(1, sorted_indices, remove), float("-inf"))
            next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            unfinished &= next_token[:, 0] != eos_token_id
            if not unfinished.any():
                break

        return input_ids
