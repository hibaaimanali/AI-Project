"""MS-GPT sampling with shared conditioning keys and values."""

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers import DynamicCache, StaticCache


def _condition_token(layer, hidden, keys, values, context_mask, *, fused_attention=False):
    attention = layer.cross_attn
    molecules = keys.shape[0]
    query = attention.q_proj(layer.norm(hidden))
    query = query.reshape(molecules, -1, attention.num_heads, attention.head_dim).transpose(1, 2)
    mask = context_mask[:, None, None, :]
    if fused_attention:
        attended = scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=attention.scale)
    else:
        scores = torch.matmul(query * attention.scale, keys.transpose(-2, -1))
        scores = scores.masked_fill(~mask, float("-inf"))
        attended = torch.matmul(scores.softmax(dim=-1), values)
    attended = attended.transpose(1, 2).reshape_as(hidden)
    conditioned = hidden + attention.out_proj(attended)
    return conditioned + layer.ffn(layer.norm2(conditioned))


def _encode_compact_condition(model, fingerprint, formula_vec):
    encoder = model.fp_encoder
    active = fingerprint.sum(dim=-1, dtype=torch.long).clamp_(1, encoder.max_active_bits)
    # All padding is masked, so short fingerprints can share a smaller context.
    maximum = int(active.max())
    width = min(64, encoder.max_active_bits)
    while width < maximum:
        width = min(2 * width, encoder.max_active_bits)
    bits = torch.arange(fingerprint.shape[-1], device=fingerprint.device)
    indices = torch.where(fingerprint.bool(), bits, fingerprint.shape[-1]).topk(
        width, largest=False, sorted=True,
    ).values
    indices = indices.masked_fill(indices == fingerprint.shape[-1], 0)
    mask = torch.arange(width, device=fingerprint.device) < active[:, None]
    values = encoder.value_proj(fingerprint.gather(1, indices).unsqueeze(-1))
    embedded = encoder.combine(torch.cat((encoder.bit_embedding(indices), values), dim=-1))
    embedded = embedded.masked_fill(~mask.unsqueeze(-1), 0)
    embedded = encoder.norm(encoder.self_attn(embedded, src_key_padding_mask=~mask))
    formula = model.formula_encoder(formula_vec)
    context = model.context_proj(torch.cat((embedded, formula), dim=1))
    return context, torch.cat((mask, torch.ones_like(formula[:, :, 0], dtype=torch.bool)), dim=1)


def _sample_full_vocabulary(logits):
    cutoff = torch.topk(logits, 50).values[..., -1:]
    logits = logits.masked_fill(logits < cutoff, float("-inf"))
    sorted_logits, sorted_indices = logits.sort(descending=True)
    remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > 0.95
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    logits = logits.masked_fill(remove.scatter(1, sorted_indices, remove), float("-inf"))
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


class WindowedCache(StaticCache):
    """Share the full cache across graphs that attend to different prefix lengths."""

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return keys[:, :, :self.attention_length], values[:, :, :self.attention_length]


class GraphedDecodeStep:
    """Replay one fixed-shape token step with an in-place, masked KV cache."""

    def __init__(self, model, context_cache, context_mask, max_len, samples, prefix_attention):
        self.model = model
        self.prefix_attention = prefix_attention
        self.initial_length = min(32, max_len) if prefix_attention else max_len
        self.context_cache = {
            index: (layer, keys.clone(), values.clone())
            for index, (layer, keys, values) in context_cache.items()
        }
        self.context_mask = context_mask.clone()
        self.cache = WindowedCache(config=model.gpt2.config, max_cache_len=max_len)
        device = context_mask.device
        self.positions = torch.arange(max_len, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.tokens = torch.zeros((context_mask.shape[0] * samples, max_len), dtype=torch.long, device=device)
        self.unfinished = torch.ones(self.tokens.shape[0], dtype=torch.bool, device=device)
        self.warmup_position = torch.zeros_like(self.position)
        self.warmup_tokens = torch.zeros_like(self.tokens)
        self.warmup_unfinished = torch.ones_like(self.unfinished)
        self.eos = torch.zeros((), dtype=torch.long, device=device)
        self.graphs = {}

        # Initialize library workspaces and allocate the static cache before capturing.
        self._capture(self.initial_length)

    def _capture(self, attention_length):
        self.cache.attention_length = attention_length
        device = self.tokens.device
        # Warmup may write the current slot, which the next replay overwrites.
        # Earlier slots belong to the active batch and must stay intact.
        self.warmup_position.copy_(self.position)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.random.fork_rng(devices=[device]), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            with torch.cuda.stream(stream):
                self._step(self.warmup_position, self.warmup_tokens, self.warmup_unfinished)
                stream.synchronize()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._step(self.position, self.tokens, self.unfinished)
        self.graphs[attention_length] = graph

    def _step(self, position, tokens, unfinished):
        model = self.model
        positions = self.positions[:self.cache.attention_length]
        mask = positions.new_zeros(positions.shape, dtype=torch.float16)
        mask = mask.masked_fill(positions > position, float("-inf"))[None, None, None, :]
        token = tokens.index_select(1, position)
        hidden = model.transformer.wte(token) + model.transformer.wpe(position.unsqueeze(0))
        for index, block in enumerate(model.transformer.h):
            hidden = block(
                hidden, past_key_values=self.cache, cache_position=position, attention_mask=mask,
                use_cache=True,
            )[0]
            if index in self.context_cache:
                layer, keys, values = self.context_cache[index]
                hidden = _condition_token(
                    layer, hidden, keys, values, self.context_mask, fused_attention=self.prefix_attention,
                )
        logits = model.gpt2.lm_head(model.transformer.ln_f(hidden[:, -1])).float()
        next_token = _sample_full_vocabulary(logits)
        position.add_(1)
        tokens.index_copy_(1, position, next_token)
        unfinished.logical_and_(next_token[:, 0] != self.eos)

    def run(self, context_cache, context_mask, max_len, bos_token_id, eos_token_id, candidates):
        for index, (_, keys, values) in context_cache.items():
            _, cached_keys, cached_values = self.context_cache[index]
            cached_keys.copy_(keys)
            cached_values.copy_(values)
        self.context_mask.copy_(context_mask)
        self.position.zero_()
        self.tokens[:, 0].fill_(bos_token_id)
        self.unfinished[:candidates].fill_(True)
        self.unfinished[candidates:].fill_(False)
        self.eos.fill_(eos_token_id)
        attention_length = self.initial_length
        eos_check_interval = 4 if self.prefix_attention else 8
        for step in range(1, max_len):
            if step > attention_length:
                attention_length = min(2 * attention_length, max_len)
                if attention_length not in self.graphs:
                    self._capture(attention_length)
            self.graphs[attention_length].replay()
            if step % eos_check_interval == 0 and not self.unfinished.any():
                return self.tokens[:candidates, :step + 1].clone()
        return self.tokens[:candidates].clone()


class CachedGeneration:
    def __init__(self, model, backend, *, cuda_graph=False, graph_batch_size=None, prefix_attention=False):
        self.model = model
        self.cuda_graph = cuda_graph
        self.prefix_attention = prefix_attention
        self.graph_batch_size = graph_batch_size
        self.graphed_step = None
        self.graph_shape = None
        if cuda_graph:
            # Keep the matrix weights in FP16 so graph replay skips autocast copies.
            # The positional embedding keeps the residual stream in FP32.
            for module in (model.gpt2, model.cross_attn_layers):
                for parameter in module.parameters():
                    if parameter.ndim == 2 and parameter is not model.transformer.wpe.weight:
                        parameter.data = parameter.data.half()
                if prefix_attention:
                    for layer in module.modules():
                        weight = getattr(layer, "weight", None)
                        bias = getattr(layer, "bias", None)
                        if weight is not None and weight.ndim == 2 and bias is not None:
                            bias.data = bias.data.half()
        elif backend != "eager":
            self.condition_token = torch.compile(_condition_token, backend=backend, dynamic=True, fullgraph=True)
        else:
            self.condition_token = _condition_token

    def __call__(self, fingerprint, formula_vec, max_len, bos_token_id, eos_token_id, num_samples):
        model = self.model
        molecules = fingerprint.shape[0]
        if max_len == 1:
            return torch.full((molecules * num_samples, 1), bos_token_id, device=fingerprint.device)
        if self.cuda_graph and self.graph_batch_size and molecules < self.graph_batch_size:
            pad = self.graph_batch_size - molecules
            fingerprint = torch.cat((fingerprint, fingerprint[:1].expand(pad, -1)))
            formula_vec = torch.cat((formula_vec, formula_vec[:1].expand(pad, -1)))
        with torch.autocast("cuda", enabled=False):
            if self.cuda_graph:
                context, context_mask = _encode_compact_condition(model, fingerprint.float(), formula_vec.float())
            else:
                context, context_mask = model.encode_condition(fingerprint.float(), formula_vec.float())

        # Every token and sample for a molecule uses the same conditioning projections.
        context_cache = {}
        for index, layer in model.cross_attn_layers.items():
            attention = layer.cross_attn
            shape = (*context.shape[:2], attention.num_heads, attention.head_dim)
            keys = attention.k_proj(context).reshape(shape).transpose(1, 2)
            values = attention.v_proj(context).reshape(shape).transpose(1, 2)
            context_cache[int(index)] = layer, keys, values

        if self.cuda_graph:
            shape = (context.shape, max_len, num_samples)
            if shape != self.graph_shape:
                self.graphed_step = None
                self.graph_shape = None
                self.graphed_step = GraphedDecodeStep(
                    model, context_cache, context_mask, max_len, num_samples, self.prefix_attention,
                )
                self.graph_shape = shape
            return self.graphed_step.run(
                context_cache, context_mask, max_len, bos_token_id, eos_token_id, molecules * num_samples,
            )

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
            next_token = _sample_full_vocabulary(logits)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            unfinished &= next_token[:, 0] != eos_token_id
            if not unfinished.any():
                break

        return input_ids
