"""transformers 4.57 / 5.x compatibility shims: rope "default" init, StaticLayer lazy init, codebook tie."""
import inspect
from types import SimpleNamespace

import torch

from breeze_models.cache_compat import lazy_init_static_layer
from breeze_models.rope_compat import compute_default_rope_parameters, resolve_rope_init_fn


def test_default_rope_matches_reference_formula():
    cfg = SimpleNamespace(rope_theta=10000.0, head_dim=64, hidden_size=512, num_attention_heads=16)
    inv_freq, scaling = compute_default_rope_parameters(cfg, device="cpu")
    expected = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.int64).float() / 64))
    assert torch.allclose(inv_freq, expected) and scaling == 1.0
    assert inv_freq.shape == (32,)


def test_default_rope_derives_head_dim_and_partial_factor():
    cfg = SimpleNamespace(rope_theta=10000.0, hidden_size=512, num_attention_heads=8, partial_rotary_factor=0.5)
    inv_freq, _ = compute_default_rope_parameters(cfg)
    assert inv_freq.shape == (16,)  # head_dim 64 * 0.5 = 32 -> 16 frequencies


def test_resolve_rope_init_fn_default_and_known_types():
    assert callable(resolve_rope_init_fn("default"))
    assert callable(resolve_rope_init_fn("llama3"))
    try:
        resolve_rope_init_fn("no-such-rope")
    except KeyError as e:
        assert "no-such-rope" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_lazy_init_static_layer_on_installed_transformers():
    from transformers.cache_utils import StaticLayer

    layer = StaticLayer(max_cache_len=8)
    lazy_init_static_layer(layer, torch.zeros(1, 2, 1, 4))
    assert layer.is_initialized and tuple(layer.keys.shape) == (1, 2, 8, 4)


def test_lazy_init_static_layer_handles_both_signatures():
    calls = []
    one = SimpleNamespace(lazy_initialization=lambda key_states: calls.append(("one", key_states)))
    two = SimpleNamespace(lazy_initialization=lambda key_states, value_states: calls.append(("two", key_states, value_states)))
    lazy_init_static_layer(one, "k")
    lazy_init_static_layer(two, "k")
    assert calls == [("one", "k"), ("two", "k", "k")]


def test_codebook_tie_mapping_follows_tie_codebooks_embeddings():
    from breeze_models.breeze import BreezeForConditionalGeneration as M

    assert M._tied_weights_keys == {
        "backbone_model.embed_tokens.embed_audio_tokens.weight": "depth_decoder.model.embed_tokens.weight"
    }
    on = SimpleNamespace(config=SimpleNamespace(tie_codebooks_embeddings=True), _tied_weights_keys=M._tied_weights_keys)
    off = SimpleNamespace(config=SimpleNamespace(tie_codebooks_embeddings=False), _tied_weights_keys=M._tied_weights_keys)
    assert M.get_expanded_tied_weights_keys(on) == M._tied_weights_keys
    assert M.get_expanded_tied_weights_keys(off) == {}


def test_mask_compat_translates_kwargs_for_installed_signature():
    from breeze_models import mask_compat
    from transformers import masking_utils

    params = inspect.signature(masking_utils.create_causal_mask).parameters
    seen = {}
    if "inputs_embeds" in params:
        # transformers 5.x: input_embeds -> inputs_embeds, cache_position dropped
        def five(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
            seen.update(dict(inputs_embeds=inputs_embeds))
        mask_compat._wrap(five)(config=None, input_embeds="E", attention_mask=None, cache_position="P", past_key_values=None)
        assert seen == {"inputs_embeds": "E"}
    else:
        # transformers 4.57: passthrough
        def four(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None):
            seen.update(dict(input_embeds=input_embeds, cache_position=cache_position))
        mask_compat._wrap(four)(config=None, input_embeds="E", attention_mask=None, cache_position="P", past_key_values=None)
        assert seen == {"input_embeds": "E", "cache_position": "P"}


def test_cache_layer_kv_on_installed_dynamic_cache():
    from breeze_models.cache_compat import cache_layer_kv
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    k, v = torch.ones(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)
    cache.update(k, v, 0)
    got_k, got_v = cache_layer_kv(cache, 0)
    assert torch.equal(got_k, k) and torch.equal(got_v, v)


def test_cache_layer_kv_indexable_fallback():
    from breeze_models.cache_compat import cache_layer_kv

    assert cache_layer_kv([("k0", "v0"), ("k1", "v1")], 1) == ("k1", "v1")


def test_reinit_computed_buffers_restores_every_kind_of_buffer():
    from breeze_models.buffer_compat import reinit_computed_buffers

    class Offsets(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_codebooks, self.vocab_size = 3, 10
            self.register_buffer("audio_tokens_offsets", torch.full((3,), 999), persistent=False)

    class Rope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.rope_init_fn = lambda config, device: (torch.tensor([1.0, 0.5]), 2.0)
            self.register_buffer("inv_freq", torch.zeros(2), persistent=False)
            self.original_inv_freq = torch.zeros(2)

    class Gemma(torch.nn.Module):
        layer_types = ["full_attention"]

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.register_buffer("full_attention_inv_freq", torch.zeros(2), persistent=False)

        @staticmethod
        def _compute_inv_freq(config, layer_type, device=None):
            return torch.tensor([3.0, 4.0]), 1.0

    class Scaled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scalar_embed_scale = 32.0
            self.register_buffer("embed_scale", torch.tensor(0.0), persistent=False)

    root = torch.nn.Module()
    root.a, root.b, root.c, root.d = Offsets(), Rope(), Gemma(), Scaled()
    touched = reinit_computed_buffers(root)
    assert sorted(touched) == ["a.audio_tokens_offsets", "b.inv_freq", "c.full_attention_inv_freq", "d.embed_scale"]
    assert root.a.audio_tokens_offsets.tolist() == [0, 10, 20]
    assert root.b.inv_freq.tolist() == [1.0, 0.5] and root.b.original_inv_freq.tolist() == [1.0, 0.5] and root.b.attention_scaling == 2.0
    assert root.c.full_attention_inv_freq.tolist() == [3.0, 4.0]
    assert float(root.d.embed_scale) == 32.0


def test_breeze_embeddings_recompute_offsets_from_config():
    from breeze_models.breeze import BreezeBackboneModelEmbeddings
    from breeze_models.buffer_compat import reinit_computed_buffers

    cfg = SimpleNamespace(hidden_size=8, audio_embed_size=None, num_codebooks=4, vocab_size=5)
    emb = BreezeBackboneModelEmbeddings(cfg)
    emb.audio_tokens_offsets.fill_(-1)
    reinit_computed_buffers(emb)
    assert emb.audio_tokens_offsets.tolist() == [0, 5, 10, 15]


def _static_cache(impl, max_len=8):
    from transformers import PretrainedConfig
    from transformers.cache_utils import StaticCache
    from breeze_models.cache_compat import lazy_init_static_layer

    cfg = PretrainedConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=1,
                           head_dim=4, max_position_embeddings=16)
    cfg._attn_implementation = impl
    cache = StaticCache(config=cfg, max_cache_len=max_len, batch_size=1)
    for layer in cache.layers:
        lazy_init_static_layer(layer, torch.zeros(1, 2, 1, 4))
    return cfg, cache


def test_causal_mask_honours_explicit_cache_position_on_empty_static_cache():
    """The depth decoder prebuilds masks for future positions against an empty StaticCache."""
    from breeze_models.mask_compat import create_causal_mask

    cfg, cache = _static_cache("sdpa")
    mask = create_causal_mask(config=cfg, input_embeds=torch.zeros(1, 1, 8), attention_mask=None,
                              cache_position=torch.tensor([5]), past_key_values=cache)
    assert mask is not None and tuple(mask.shape) == (1, 1, 1, 8)
    assert mask.dtype == torch.bool and mask[0, 0, 0].int().tolist() == [1, 1, 1, 1, 1, 1, 0, 0]

    cfg, cache = _static_cache("eager")
    mask = create_causal_mask(config=cfg, input_embeds=torch.zeros(1, 2, 8, dtype=torch.bfloat16), attention_mask=None,
                              cache_position=torch.tensor([3, 4]), past_key_values=cache)
    assert mask.dtype == torch.bfloat16 and tuple(mask.shape) == (1, 1, 2, 8)
    assert (mask[0, 0] > -1e4).int().tolist() == [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]


def test_causal_mask_passes_prepared_4d_masks_through():
    from breeze_models.mask_compat import create_causal_mask

    cfg, cache = _static_cache("sdpa")
    ready = torch.zeros(1, 1, 1, 8, dtype=torch.bool)
    out = create_causal_mask(config=cfg, input_embeds=torch.zeros(1, 1, 8), attention_mask=ready,
                             cache_position=torch.tensor([5]), past_key_values=cache)
    assert out is ready
