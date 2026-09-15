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


# --- regression tests for the transformers-5 compat review (2026-09-15) -------------------------

def _tiny_depth_decoder_config():
    from breeze_models.breeze_base_config import BreezeDepthDecoderConfig

    return BreezeDepthDecoderConfig(
        num_codebooks=4, backbone_hidden_size=8, vocab_size=16, hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=8, head_dim=4,
    )


def _gate_stub_model():
    model = torch.nn.Module()
    model.config = SimpleNamespace(vocab_size=8, codec_config=SimpleNamespace(codebook_size=4))
    return model


def test_static_kv_len_only_answers_for_compileable_caches():
    """Critical 1: 5.x `Cache.max_cache_len` also resolves for DynamicCache (-1, or a raise)."""
    from breeze_models.mask_compat import _static_kv_len
    from transformers.cache_utils import DynamicCache

    assert _static_kv_len(None) is None
    assert _static_kv_len(DynamicCache()) is None  # bare cache: max_cache_len raises on both
    filled = DynamicCache()
    filled.update(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4), 0)
    assert _static_kv_len(filled) is None
    _, static = _static_cache("sdpa", max_len=8)
    assert _static_kv_len(static) == 8


def test_causal_mask_delegates_for_a_dynamic_cache():
    """Critical 1: the explicit branch must not fire for a growing cache, with or without a mask."""
    from breeze_models.mask_compat import create_causal_mask
    from transformers import PretrainedConfig
    from transformers.cache_utils import DynamicCache

    cfg = PretrainedConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=2,
                           num_hidden_layers=1, head_dim=4, max_position_embeddings=16)
    cfg._attn_implementation = "sdpa"
    embeds = torch.zeros(1, 2, 8)

    without = create_causal_mask(config=cfg, input_embeds=embeds, attention_mask=None,
                                 cache_position=torch.tensor([0, 1]), past_key_values=DynamicCache())
    assert without is None or without.shape[-1] == 2  # library's own answer, never arange(-1)

    with_mask = create_causal_mask(config=cfg, input_embeds=embeds,
                                   attention_mask=torch.ones(1, 2, dtype=torch.long),
                                   cache_position=torch.tensor([0, 1]), past_key_values=DynamicCache())
    assert with_mask is None or with_mask.shape[-1] == 2


def test_depth_decoder_forward_with_a_dynamic_cache_and_no_attention_mask():
    """Critical 1, as reproduced in the review: RuntimeError from torch.arange(-1) on 5.15."""
    from breeze_models.breeze import BreezeDepthDecoderModel
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model = BreezeDepthDecoderModel(_tiny_depth_decoder_config())
    out = model(input_ids=torch.tensor([[1, 2]]), backbone_last_hidden_state=torch.zeros(1, 8),
                past_key_values=DynamicCache(), use_cache=True, cache_position=torch.tensor([0, 1]))
    assert tuple(out[0].shape) == (1, 2, 8)


def test_causal_mask_returns_none_for_backends_without_a_dense_mask():
    """Important 4: 4.57 returned None for flash_attention_2; a bool tensor would be wrong."""
    from breeze_models import mask_compat
    from breeze_models.mask_compat import create_causal_mask

    impls = ["flash_attention_2"]
    if "cache_position" not in inspect.signature(mask_compat._mu.create_causal_mask).parameters:
        # 5.x only: on 4.57 this is a passthrough and flex_attention compiles a real BlockMask,
        # which needs a C++ toolchain. Breeze never advertises flex support anyway.
        impls.append("flex_attention")
    for impl in impls:
        cfg, cache = _static_cache(impl)
        assert create_causal_mask(config=cfg, input_embeds=torch.zeros(1, 1, 8), attention_mask=None,
                                  cache_position=torch.tensor([5]), past_key_values=cache) is None


def test_mask_compat_passes_positional_calls_straight_through():
    """Minor 9: the explicit branch reads keywords only, so a positional caller must delegate."""
    from breeze_models import mask_compat

    if "cache_position" in inspect.signature(mask_compat._mu.create_causal_mask).parameters:
        return  # 4.57: the wrapper is a plain passthrough anyway

    seen = []

    def five(config, inputs_embeds=None, attention_mask=None, past_key_values=None):
        seen.append(config)
        return "delegated"

    cfg, cache = _static_cache("sdpa")
    out = mask_compat._wrap(five)(cfg, inputs_embeds=torch.zeros(1, 1, 8), attention_mask=None,
                                  cache_position=torch.tensor([5]), past_key_values=cache)
    assert out == "delegated" and seen == [cfg]


def test_depth_decoder_prepare_inputs_for_generation_forwards_by_keyword():
    """Critical 2: 5.x reordered the base signature and dropped `cache_position`."""
    from breeze_models.breeze import BreezeDepthDecoderForCausalLM
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    lm = BreezeDepthDecoderForCausalLM(_tiny_depth_decoder_config())
    cache = DynamicCache()
    prefill = lm.prepare_inputs_for_generation(
        input_ids=torch.tensor([[1, 2]]), past_key_values=cache,
        attention_mask=torch.ones(1, 2, dtype=torch.long), cache_position=torch.tensor([0, 1]),
        backbone_last_hidden_state=torch.zeros(1, 8), use_cache=True)
    assert prefill["cache_position"].tolist() == [0, 1]
    assert "backbone_last_hidden_state" in prefill  # kept on the first step
    assert "position_ids" not in prefill  # popped with a default, never KeyError
    lm(**prefill)  # the forward accepts exactly what we built

    decode = lm.prepare_inputs_for_generation(
        input_ids=torch.tensor([[1, 2, 3]]), past_key_values=cache,
        attention_mask=torch.ones(1, 3, dtype=torch.long), cache_position=torch.tensor([2]),
        backbone_last_hidden_state=torch.zeros(1, 8), use_cache=True)
    assert "backbone_last_hidden_state" not in decode


def test_depth_decoder_prepare_inputs_synthesises_cache_position_on_transformers_5():
    """5's generate no longer creates `cache_position`; the depth decoder indexes with it."""
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        return  # 4.57 always supplies one (and its base dereferences it)
    from breeze_models.breeze import BreezeDepthDecoderForCausalLM
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    lm = BreezeDepthDecoderForCausalLM(_tiny_depth_decoder_config())
    cache = DynamicCache()
    prefill = lm.prepare_inputs_for_generation(
        input_ids=torch.tensor([[1, 2]]), past_key_values=cache,
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        backbone_last_hidden_state=torch.zeros(1, 8), use_cache=True)
    assert prefill["cache_position"].tolist() == [0, 1]
    lm(**prefill)
    decode = lm.prepare_inputs_for_generation(
        input_ids=torch.tensor([[1, 2, 3]]), past_key_values=cache,
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        backbone_last_hidden_state=torch.zeros(1, 8), use_cache=True)
    assert decode["cache_position"].tolist() == [2, 3, 4]
    assert "backbone_last_hidden_state" not in decode


def test_breeze_rotary_embedding_without_rope_scaling_survives_init_weights():
    """Important 3: 5's `_init_weights` calls `module.compute_default_rope_parameters(config)`."""
    from breeze_models.breeze import BreezeDepthDecoderForCausalLM, BreezeRotaryEmbedding
    from breeze_models.rope_compat import compute_default_rope_parameters

    cfg = _tiny_depth_decoder_config()  # no rope_scaling passed in -> rope_type "default"
    rotary = BreezeRotaryEmbedding(cfg)
    assert rotary.rope_type == "default" and hasattr(rotary, "original_inv_freq")

    expected, scaling = compute_default_rope_parameters(cfg)
    got, got_scaling = rotary.compute_default_rope_parameters(cfg)  # the hook 5 reaches for
    assert torch.allclose(got, expected) and (got_scaling, scaling) == (1.0, 1.0)

    torch.manual_seed(0)
    lm = BreezeDepthDecoderForCausalLM(cfg)  # post_init() runs _init_weights over every submodule
    lm._init_weights(rotary)
    assert torch.allclose(rotary.inv_freq, expected)


def test_vendored_qwen_decoder_rotary_keeps_its_rotatory_spelling():
    """Important 3 (note): the misspelling is what keeps it out of 5's rotary init branch.

    Read as text rather than imported: ``test_qwen_tokenizer_vendored`` reloads that package and
    importing it from here first would leave two copies of its classes around.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "breeze_models" / "qwen_tokenizer"
              / "modeling_qwen3_tts_tokenizer_v2.py").read_text(encoding="utf-8")
    assert "class Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(nn.Module):" in source
    assert "RotaryEmbedding" not in source  # nothing here matches 5's `_init_weights` branch


def test_reinit_rebinds_original_inv_freq_off_the_meta_device():
    """Important 5: on 5.x it is a plain attribute the loader leaves on meta; copy_ is a no-op."""
    from breeze_models.buffer_compat import reinit_computed_buffers

    class Rope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.rope_init_fn = lambda config, device: (torch.tensor([1.0, 0.5]), 2.0)
            self.register_buffer("inv_freq", torch.zeros(2), persistent=False)
            self.original_inv_freq = torch.zeros(2, device="meta")

    root = torch.nn.Module()
    root.r = Rope()
    assert root.r.original_inv_freq.device.type == "meta"
    reinit_computed_buffers(root)
    assert root.r.original_inv_freq.device.type != "meta"
    assert root.r.original_inv_freq.tolist() == [1.0, 0.5]


def test_reinit_preserves_float32_inv_freq_on_a_real_rotary():
    """buffer_compat preserves the buffer dtype; pin it (rope math must stay fp32)."""
    from breeze_models.breeze import BreezeRotaryEmbedding
    from breeze_models.buffer_compat import reinit_computed_buffers
    from breeze_models.rope_compat import compute_default_rope_parameters

    rotary = BreezeRotaryEmbedding(_tiny_depth_decoder_config())
    rotary.inv_freq.fill_(float("nan"))
    touched = reinit_computed_buffers(rotary)
    assert touched == [".inv_freq"]
    assert rotary.inv_freq.dtype == torch.float32
    assert torch.allclose(rotary.inv_freq, compute_default_rope_parameters(rotary.config)[0])
    assert rotary.original_inv_freq.device.type == rotary.inv_freq.device.type


def test_fast_path_gate_refuses_cuda_graphs_on_transformers_5(monkeypatch):
    """Blind spot: 5's StaticLayer.update ignores cache_position, so graph replay corrupts audio."""
    import transformers
    from breeze_models.fast_streaming import (
        FastBreezeStreamingRuntime,
        FastStreamingConfig,
        require_fast_path_support,
    )

    monkeypatch.setattr(transformers, "__version__", "5.15.0")
    for config in (FastStreamingConfig(fast_all=True), FastStreamingConfig(fast_depth_decoder=True)):
        assert config.any_stage_fast()
        try:
            require_fast_path_support(config)
        except RuntimeError as e:
            assert "transformers" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
        try:
            FastBreezeStreamingRuntime(_gate_stub_model(), None, config)
        except RuntimeError as e:
            assert "5.15.0" in str(e)
        else:
            raise AssertionError("expected RuntimeError from __init__")


def test_fast_path_gate_is_a_no_op_on_4_57_and_for_the_eager_path(monkeypatch):
    import transformers
    from breeze_models.fast_streaming import (
        FastBreezeStreamingRuntime,
        FastStreamingConfig,
        require_fast_path_support,
    )

    monkeypatch.setattr(transformers, "__version__", "4.57.1")
    require_fast_path_support(FastStreamingConfig(fast_all=True))  # 4.57: the fast path is fine

    monkeypatch.setattr(transformers, "__version__", "5.15.0")
    for config in (FastStreamingConfig(), FastStreamingConfig(fast_all=False),
                   FastStreamingConfig(fast_all=False, fast_depth_decoder=True)):
        assert not config.any_stage_fast()
        require_fast_path_support(config)  # the eager path is unaffected
    # ... and __init__ gets past the gate, failing only on the pre-existing CUDA check
    try:
        FastBreezeStreamingRuntime(_gate_stub_model(), None, FastStreamingConfig(fast_all=False))
    except RuntimeError as e:
        assert str(e) == "fast streaming requires a CUDA device"
    else:
        raise AssertionError("expected the CUDA device check")
