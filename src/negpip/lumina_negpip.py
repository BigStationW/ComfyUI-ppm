import torch
import re

from comfy.ldm.lumina.model import JointAttention
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked
import comfy.utils
import logging

NEGPIP_ZIMAGE_KEY = "negpip_zimage"
NEGPIP_ZIMAGE_STRENGTH_KEY = "negpip_zimage_strength"

ZIMAGE_CHAT_PREFIX = "<|im_start|>user\n"
ZIMAGE_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
ZIMAGE_SUFFIX_TOKEN_COUNT = 5

_negpip_state: dict = {}
_encoder_tokenizer_cache: dict = {}
_negated_spans_cache: list =[]


# ===========================================================================
#  CONDNegpipZImage
# ===========================================================================

class CONDNegpipZImage:
    def __init__(self, cond: torch.Tensor):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, **kwargs):
        repeated = comfy.utils.repeat_to_batch_size(self.cond, batch_size)
        return self._copy_with(repeated)

    def can_concat(self, other):
        shape_ok  = self.cond.shape == other.cond.shape
        device_ok = self.cond.device == other.cond.device
        if not device_ok:
            logging.warning("[CONDNegpipZImage] conds not on same device.")
        return shape_ok and device_ok

    def concat(self, others):
        conds  = [self.cond] + [x.cond for x in others]
        result = torch.cat(conds)
        return result

    def size(self):
        return list(self.cond.size())


def _make_negpip_cond(tensor: torch.Tensor) -> CONDNegpipZImage:
    try:
        result = CONDNegpipZImage(tensor)
        assert hasattr(result, "process_cond"), "MISSING process_cond!"
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


# ===========================================================================
#  STEP 0 — Tokenizer patch
# ===========================================================================

def make_zimage_tokenize_with_weights(original_tokenize_with_weights, inner_sd_tokenizer):
    KNOWN_KEYS = ["qwen3_4b", "gemma3_4b"]
    NEG_BLOCK_RE = re.compile(r"\(([^()]*?):\s*(-?\d+(?:\.\d+)?)\)")

    def _find_weighted_spans(text):
        spans =[]
        for m in re.finditer(r"\(([^()]*?):(-?\d+(?:\.\d+)?)\)", text):
            weight = float(m.group(2))
            if weight != 1.0:
                spans.append((m.start(1), m.end(1), weight))
        return spans

    def _strip_negative_blocks_keep_cache(raw_text: str):
        global _negated_spans_cache
        _negated_spans_cache =[]

        def repl(m):
            span_text = m.group(1)
            w = float(m.group(2))
            if w < 0:
                _negated_spans_cache.append((span_text, w))
                return ""
            return m.group(0)

        return re.sub(NEG_BLOCK_RE, repl, raw_text)

    def _get_hf_tokenizer(inner_tok):
        for attr in ("tokenizer", "_tokenizer", "hf_tokenizer"):
            t = getattr(inner_tok, attr, None)
            if t is not None and callable(getattr(t, "encode", None)):
                return t
        return None

    def patched_tokenize_with_weights(text, return_word_ids=False, **kwargs):
        global _negated_spans_cache

        clean_text = _strip_negative_blocks_keep_cache(text)
        result = original_tokenize_with_weights(
            clean_text, return_word_ids=return_word_ids, **kwargs
        )

        detected_key = next((k for k in KNOWN_KEYS if k in result), None)
        if detected_key is None:
            return result

        wrapped_chunks      = result[detected_key]
        weight_spans        = _find_weighted_spans(clean_text)

        if not weight_spans:
            return result

        hf_tok = _get_hf_tokenizer(inner_sd_tokenizer)
        if hf_tok is None:
            return result

        flat_wrapped   = [item for chunk in wrapped_chunks for item in chunk]
        flat_injected  = list(flat_wrapped)

        for orig_start, orig_end, w in weight_spans:
            if w < 0:
                continue
            try:
                ids_before = hf_tok.encode(ZIMAGE_CHAT_PREFIX + clean_text[:orig_start], add_special_tokens=False)
                ids_after  = hf_tok.encode(ZIMAGE_CHAT_PREFIX + clean_text[:orig_end],   add_special_tokens=False)
            except Exception as e:
                continue
            for pos in range(len(ids_before), len(ids_after)):
                if pos >= len(flat_injected):
                    continue
                old  = flat_injected[pos]
                rest = tuple(old[2:]) if len(old) > 2 else ()
                flat_injected[pos] = (old[0], w) + rest

        new_chunks, idx =[], 0
        for chunk in wrapped_chunks:
            new_chunks.append(flat_injected[idx: idx + len(chunk)])
            idx += len(chunk)
        result[detected_key] = new_chunks

        return result

    return patched_tokenize_with_weights


# ===========================================================================
#  STEP 1 — encode_token_weights replacement
# ===========================================================================

def zimage_encode_token_weights_negpip(real_encoder, token_weight_pairs):
    from comfy import model_management
    self = real_encoder

    has_non_unit_weights = any(item[1] != 1.0 for chunk in token_weight_pairs for item in chunk)
    sections             = len(token_weight_pairs)
    has_cached_negated   = bool(_negated_spans_cache)

    to_encode_tokens = [[item[0] for item in chunk] for chunk in token_weight_pairs]
    max_token_len    = max((len(t) for t in to_encode_tokens), default=0)

    if has_non_unit_weights or sections == 0:
        if hasattr(self, "gen_empty_tokens"):
            to_encode_tokens.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
        else:
            from comfy.sd1_clip import gen_empty_tokens
            to_encode_tokens.append(gen_empty_tokens(self.special_tokens, max_token_len))
        has_empty_baseline     = True
        token_weight_pairs_abs = [
            [(tid, abs(w)) for tid, w in chunk]
            for chunk in token_weight_pairs
        ]
    else:
        has_empty_baseline     = False
        token_weight_pairs_abs = token_weight_pairs

    o           = self.encode(to_encode_tokens)
    out, pooled = o[:2]

    first_pooled = (
        pooled[0:1].to(device=model_management.intermediate_device())
        if pooled is not None else pooled
    )

    output_sections =[]
    for k in range(sections):
        z = out[k:k+1].clone()
        if has_non_unit_weights and has_empty_baseline:
            z_empty = out[-1]
            for j in range(min(z.shape[1], len(token_weight_pairs_abs[k]))):
                w = token_weight_pairs_abs[k][j][1]
                if w != 1.0:
                    z[0, j] = (z[0, j] - z_empty[j]) * w + z_empty[j]
        output_sections.append(z)

    normal_out = (
        torch.cat(output_sections, dim=-2)
        if output_sections else out[-1:].clone()
    )
    normal_out = normal_out.to(device=model_management.intermediate_device())

    extra = {}
    if len(o) > 2 and o[2]:
        raw_extra = o[2]
        for ek, ev in raw_extra.items():
            if ek == "attention_mask":
                ev = ev[:sections].flatten().unsqueeze(0).to(
                    device=model_management.intermediate_device()
                )
            extra[ek] = ev

    if has_cached_negated:
        hf_tok = _encoder_tokenizer_cache.get(id(self), None)

        if hf_tok is not None:
            neg_emb_list      =[]
            neg_strength_list = []

            for si, (span_text, w) in enumerate(_negated_spans_cache):
                full_neg_text           = ZIMAGE_CHAT_PREFIX + span_text + ZIMAGE_CHAT_SUFFIX
                neg_token_ids_templated = hf_tok.encode(full_neg_text, add_special_tokens=False)

                span_only_ids = hf_tok.encode(span_text, add_special_tokens=False)
                span_only_len = len(span_only_ids)

                prefix_ids = hf_tok.encode(ZIMAGE_CHAT_PREFIX, add_special_tokens=False)
                suffix_ids = hf_tok.encode(ZIMAGE_CHAT_SUFFIX,  add_special_tokens=False)
                prefix_len = len(prefix_ids)
                suffix_len = len(suffix_ids)

                neg_full_tokens = [neg_token_ids_templated]
                if hasattr(self, "gen_empty_tokens"):
                    neg_full_tokens.append(
                        self.gen_empty_tokens(self.special_tokens, len(neg_token_ids_templated))
                    )
                else:
                    from comfy.sd1_clip import gen_empty_tokens
                    neg_full_tokens.append(
                        gen_empty_tokens(self.special_tokens, len(neg_token_ids_templated))
                    )

                neg_o         = self.encode(neg_full_tokens)
                neg_out_batch = neg_o[0]
                neg_out       = neg_out_batch[0:1]

                actual_seq_len = neg_out.shape[1]

                if actual_seq_len > prefix_len + suffix_len:
                    span_emb = neg_out[:, prefix_len:-suffix_len, :]
                else:
                    span_emb = neg_out

                span_len = span_emb.shape[1]
                neg_strength_list.extend([float(w)] * span_len)
                neg_emb_list.append(span_emb)

            neg_embeddings = torch.cat(neg_emb_list, dim=1) if neg_emb_list else None

            if neg_embeddings is not None:
                neg_embeddings = neg_embeddings.to(device=model_management.intermediate_device())
                neg_count_new  = neg_embeddings.shape[1]

                extra[NEGPIP_ZIMAGE_KEY] = neg_embeddings

                strength = torch.tensor(
                    neg_strength_list,
                    dtype=torch.float32,
                    device=model_management.intermediate_device(),
                ).unsqueeze(0)
                extra[NEGPIP_ZIMAGE_STRENGTH_KEY] = strength

                if "attention_mask" in extra:
                    am          = extra["attention_mask"]
                    orig_am_len = am.shape[1]
                    am_ones     = torch.ones(1, neg_count_new, dtype=am.dtype, device=am.device)
                    if orig_am_len >= ZIMAGE_SUFFIX_TOKEN_COUNT:
                        extra["attention_mask"] = torch.cat(
                            [am[:, :-ZIMAGE_SUFFIX_TOKEN_COUNT], am_ones, am[:, -ZIMAGE_SUFFIX_TOKEN_COUNT:]], dim=1
                        )
                    else:
                        extra["attention_mask"] = torch.cat([am, am_ones], dim=1)

    r = (normal_out, first_pooled)
    if extra:
        r = r + (extra,)
    return r


# ===========================================================================
#  STEP 2 — DIFFUSION_MODEL wrapper
# ===========================================================================

def lumina_diffusion_negpip_wrapper(executor, *args, **kwargs):
    global _negpip_state

    transformer_options: dict = kwargs.get("transformer_options", {})
    context: torch.Tensor = args[2]
    bsz = context.shape[0]

    negated_embeds = kwargs.pop(NEGPIP_ZIMAGE_KEY, None)
    neg_strength   = kwargs.pop(NEGPIP_ZIMAGE_STRENGTH_KEY, None)

    if negated_embeds is None:
        negated_embeds = transformer_options.get(NEGPIP_ZIMAGE_KEY, None)
    if neg_strength is None:
        neg_strength = transformer_options.get(NEGPIP_ZIMAGE_STRENGTH_KEY, None)

    if negated_embeds is None:
        _negpip_state = {}
        result = executor(*args, **kwargs)
        return result

    orig_tokens = context.shape[1]

    if orig_tokens <= ZIMAGE_SUFFIX_TOKEN_COUNT:
        _negpip_state = {}
        result = executor(*args, **kwargs)
        return result

    cond_or_uncond = transformer_options.get("cond_or_uncond", None)

    if negated_embeds.dim() == 2:
        negated_embeds = negated_embeds.unsqueeze(0)

    neg_embeds_single = negated_embeds[0:1].to(context)
    neg_count         = neg_embeds_single.shape[1]

    strength_single = None
    if neg_strength is not None:
        if neg_strength.dim() == 1:
            neg_strength = neg_strength.unsqueeze(0)
        strength_single = neg_strength[0:1]
    else:
        strength_single = torch.full((1, neg_count), -1.0, dtype=torch.float32, device=context.device)

    zero_pad = torch.zeros(
        1, neg_count, context.shape[2],
        dtype=context.dtype, device=context.device
    )

    new_context_parts =[]
    for b in range(bsz):
        is_uncond = (
            cond_or_uncond is not None and
            b < len(cond_or_uncond) and
            cond_or_uncond[b] == 1
        )
        prefix = context[b:b+1, :-ZIMAGE_SUFFIX_TOKEN_COUNT, :]
        suffix = context[b:b+1, -ZIMAGE_SUFFIX_TOKEN_COUNT:, :]
        insert = zero_pad if is_uncond else neg_embeds_single
        part = torch.cat([prefix, insert, suffix], dim=1)
        new_context_parts.append(part)

    context_new     = torch.cat(new_context_parts, dim=0)
    context_new_len = context_new.shape[1]

    args = list(args)
    args[2] = context_new

    if len(args) > 3 and args[3] is not None:
        if isinstance(args[3], int):
            args[3] = context_new_len
        elif isinstance(args[3], (list, tuple)):
            args[3] = type(args[3])([context_new_len] * len(args[3]))
        elif hasattr(args[3], "shape"):
            args[3] = torch.full_like(args[3], context_new_len)

    if len(args) > 4 and args[4] is not None:
        am = args[4]
        if am.dim() == 2:
            am_ones = torch.ones(am.shape[0], neg_count, dtype=am.dtype, device=am.device)
            if am.shape[1] >= ZIMAGE_SUFFIX_TOKEN_COUNT:
                args[4] = torch.cat(
                    [am[:, :-ZIMAGE_SUFFIX_TOKEN_COUNT], am_ones, am[:, -ZIMAGE_SUFFIX_TOKEN_COUNT:]], dim=1
                )
            else:
                args[4] = torch.cat([am, am_ones], dim=1)

    args = tuple(args)

    if "num_tokens" in kwargs:
        nt = kwargs["num_tokens"]
        if isinstance(nt, int):
            kwargs["num_tokens"] = context_new_len
        elif hasattr(nt, "shape"):
            kwargs["num_tokens"] = torch.full_like(nt, context_new_len)

    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
        am = kwargs["attention_mask"]
        if am.dim() == 2:
            am_ones = torch.ones(am.shape[0], neg_count, dtype=am.dtype, device=am.device)
            if am.shape[1] >= ZIMAGE_SUFFIX_TOKEN_COUNT:
                kwargs["attention_mask"] = torch.cat(
                    [am[:, :-ZIMAGE_SUFFIX_TOKEN_COUNT], am_ones, am[:, -ZIMAGE_SUFFIX_TOKEN_COUNT:]], dim=1
                )
            else:
                kwargs["attention_mask"] = torch.cat([am, am_ones], dim=1)

    _negpip_state = {
        "orig_tokens":    orig_tokens,
        "neg_count":      neg_count,
        "cond_or_uncond": cond_or_uncond,
        "strength":       strength_single,
        "logged":         False,
        "joint_logged":   False,
    }
    
    transformer_options[NEGPIP_ZIMAGE_KEY + "_orig_tokens"]    = orig_tokens
    transformer_options[NEGPIP_ZIMAGE_KEY + "_neg_count"]      = neg_count
    transformer_options[NEGPIP_ZIMAGE_KEY + "_cond_or_uncond"] = cond_or_uncond
    transformer_options[NEGPIP_ZIMAGE_STRENGTH_KEY]            = strength_single
    kwargs["transformer_options"] = transformer_options

    try:
        result = executor(*args, **kwargs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    finally:
        _negpip_state = {}
        for key in [
            NEGPIP_ZIMAGE_KEY + "_orig_tokens",
            NEGPIP_ZIMAGE_KEY + "_neg_count",
            NEGPIP_ZIMAGE_KEY + "_cond_or_uncond",
            NEGPIP_ZIMAGE_STRENGTH_KEY,
        ]:
            transformer_options.pop(key, None)

    return result


# ===========================================================================
#  STEP 3 — Full JointAttention forward replacement
# ===========================================================================

def make_joint_attention_forward_negpip(block: JointAttention):
    q_dim = block.n_local_heads     * block.head_dim
    k_dim = block.n_local_kv_heads  * block.head_dim
    v_dim = block.n_local_kv_heads  * block.head_dim

    def patched_forward(x, x_mask, freqs_cis, transformer_options={}):
        orig_tokens    = _negpip_state.get("orig_tokens",    None)
        neg_count      = _negpip_state.get("neg_count",      None)
        cond_or_uncond = _negpip_state.get("cond_or_uncond", None)
        negpip_active  = orig_tokens is not None and neg_count is not None

        bsz, seqlen, _ = x.shape

        if negpip_active:
            text_start        = 0 
            text_insert_local = orig_tokens - ZIMAGE_SUFFIX_TOKEN_COUNT
            abs_start         = text_start + text_insert_local
            abs_end           = abs_start  + neg_count

            logged_key_mask = f"mask_logged_{seqlen}"
            first_mask_log  = not _negpip_state.get(logged_key_mask, False)

            if x_mask is not None and first_mask_log:
                _negpip_state[logged_key_mask] = True
                if x_mask.dim() == 2 and x_mask.shape[1] == seqlen and abs_end <= seqlen:
                    neg_mask_vals = x_mask[:, abs_start:abs_end]
                    all_zeros = (neg_mask_vals == 0).all() or (neg_mask_vals < -1e4).all()
                    if all_zeros:
                        x_mask = x_mask.clone()
                        fill_val = x_mask[:, 0].item()
                        x_mask[:, abs_start:abs_end] = fill_val
                elif x_mask.dim() == 2:
                    pass
            elif x_mask is None and first_mask_log:
                _negpip_state[logged_key_mask] = True

        qkv = block.qkv(x)
        xq, xk, xv = torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)

        xq = xq.view(bsz, seqlen, block.n_local_heads,    block.head_dim)
        xk = xk.view(bsz, seqlen, block.n_local_kv_heads, block.head_dim)
        xv = xv.view(bsz, seqlen, block.n_local_kv_heads, block.head_dim)

        if negpip_active:
            text_start        = 0 
            text_insert_local = orig_tokens - ZIMAGE_SUFFIX_TOKEN_COUNT
            abs_start         = text_start + text_insert_local
            abs_end           = abs_start  + neg_count

            logged_key = f"logged_seqlen_{seqlen}"
            first_for_seqlen = not _negpip_state.get(logged_key, False)

            if abs_end <= seqlen:
                if first_for_seqlen:
                    _negpip_state[logged_key] = True

                for b in range(bsz):
                    is_uncond = (
                        cond_or_uncond is not None and
                        b < len(cond_or_uncond) and
                        cond_or_uncond[b] == 1
                    )
                    if not is_uncond:
                        v_slice = xv[b:b+1, abs_start:abs_end]
                        strength_single = _negpip_state.get("strength", None)
                        if strength_single is not None:
                            s = strength_single.to(dtype=v_slice.dtype, device=xv.device)
                            s = s.view(1, neg_count, 1, 1)
                            xv[b:b+1, abs_start:abs_end] = v_slice * s
                        else:
                            xv[b:b+1, abs_start:abs_end] = -v_slice
                    elif first_for_seqlen:
                        pass
            else:
                if first_for_seqlen:
                    _negpip_state[logged_key] = True

        xq = block.q_norm(xq)
        xk = block.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis)

        n_rep = block.n_local_heads // block.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        output = optimized_attention_masked(
            xq.movedim(1, 2),
            xk.movedim(1, 2),
            xv.movedim(1, 2),
            block.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        return block.out(output)

    return patched_forward
