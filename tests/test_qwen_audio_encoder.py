"""The Qwen3-TTS audio encoder owns its Mimi code, so a transformers refactor cannot reach it.

Why these exist: transformers 5.21 moved MimiModel's sub-modules under ``.encoder``/``.decoder`` and
the old subclass crashed at load (``property 'upsample' ... has no setter``). No unit test noticed
because none of them built the audio tokenizer through ``from_pretrained``; the round trip below
does that on the CPU with a tiny random model, so it needs neither the checkpoint nor a GPU.

Imports stay inside the tests: ``test_qwen_tokenizer_vendored`` reloads this package.
"""
import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "breeze_models" / "qwen_tokenizer"

# The Breeze checkpoint's audio_tokenizer/config.json, encoder_config section, verbatim.
CHECKPOINT_ENCODER_CONFIG = {
    "_frame_rate": 12.5, "attention_bias": False, "attention_dropout": 0.0, "audio_channels": 1,
    "codebook_dim": 256, "codebook_size": 2048, "compress": 2, "dilation_growth_rate": 2, "dtype": "float32",
    "head_dim": 64, "hidden_act": "gelu", "hidden_size": 512, "initializer_range": 0.02,
    "intermediate_size": 2048, "kernel_size": 7, "last_kernel_size": 3, "layer_scale_initial_scale": 0.01,
    "max_position_embeddings": 8000, "norm_eps": 1e-05, "normalize": False, "num_attention_heads": 8,
    "num_filters": 64, "num_hidden_layers": 8, "num_key_value_heads": 8, "num_quantizers": 32,
    "num_residual_layers": 1, "num_semantic_quantizers": 1, "pad_mode": "constant", "residual_kernel_size": 3,
    "rope_theta": 10000.0, "sampling_rate": 24000, "sliding_window": 250, "transformers_version": "4.57.0.dev0",
    "trim_right_ratio": 1.0, "upsample_groups": 512, "upsampling_ratios": [8, 6, 5, 4], "use_cache": False,
    "use_causal_conv": True, "use_conv_shortcut": False, "use_streaming": False,
    "vector_quantization_hidden_dimension": 256,
}

# A small model for the tests that run it. Sizes are reduced, the settings match the checkpoint.
TINY = dict(
    hidden_size=32,
    intermediate_size=64,
    num_filters=4,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    codebook_size=16,
    codebook_dim=8,
    vector_quantization_hidden_dimension=8,
    num_quantizers=4,
    num_semantic_quantizers=1,
    sliding_window=5,
    upsampling_ratios=[8, 6, 5, 4],
    audio_channels=1,
    sampling_rate=24000,
    kernel_size=7,
    last_kernel_size=3,
    residual_kernel_size=3,
    dilation_growth_rate=2,
    compress=2,
    num_residual_layers=1,
    use_causal_conv=True,
    use_conv_shortcut=False,
    pad_mode="constant",
    hidden_act="gelu",
    norm_eps=1e-5,
    layer_scale_initial_scale=1.0,  # the checkpoint's 0.01 hides attention bugs from a random tiny model
    attention_bias=False,
    attention_dropout=0.0,
)


def _tiny_encoder_config():
    from breeze_models.qwen_tokenizer.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2EncoderConfig,
    )

    return Qwen3TTSTokenizerV2EncoderConfig(**TINY, _frame_rate=12.5, rope_theta=10000.0)


def _tiny_encoder():
    from breeze_models.qwen_tokenizer.audio_encoder import Qwen3TTSAudioEncoder

    torch.manual_seed(0)
    encoder = Qwen3TTSAudioEncoder(_tiny_encoder_config()).eval()
    for param in encoder.parameters():
        if param.dim() > 1:
            torch.nn.init.normal_(param, std=0.1)
    for name, buffer in encoder.named_buffers():
        if name.endswith("embed_sum"):
            torch.nn.init.normal_(buffer)
    return encoder


def test_no_transformers_mimi_import_is_left():
    for name in ("audio_encoder.py", "modeling_qwen3_tts_tokenizer_v2.py", "configuration_qwen3_tts_tokenizer_v2.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s.*Mimi", source, re.MULTILINE), name


def test_nothing_matches_the_rotary_init_branch_of_transformers_5():
    """transformers 5 special-cases modules whose class name ends in RotaryEmbedding at load time."""
    assert "RotaryEmbedding" not in (ROOT / "audio_encoder.py").read_text(encoding="utf-8")


def test_config_reads_the_breeze_checkpoint_encoder_section():
    from breeze_models.qwen_tokenizer.audio_encoder import Qwen3TTSAudioEncoder
    from breeze_models.qwen_tokenizer.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2EncoderConfig,
    )

    config = Qwen3TTSTokenizerV2EncoderConfig(**CHECKPOINT_ENCODER_CONFIG)
    assert config.frame_rate == 12.5
    assert config.encodec_frame_rate == 25  # ceil(24000 / (8 * 6 * 5 * 4))
    assert config.sliding_window == 250
    encoder = Qwen3TTSAudioEncoder(config)
    assert encoder.downsample.conv.kernel_size == (4,)
    assert encoder.downsample.conv.stride == (2,)


def test_each_frame_sees_only_the_sliding_window():
    """With one layer, a change in frame 0 reaches frames below the window and no frame beyond it."""
    from breeze_models.qwen_tokenizer.audio_encoder import _Transformer
    from breeze_models.qwen_tokenizer.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2EncoderConfig,
    )

    window = TINY["sliding_window"]
    config = Qwen3TTSTokenizerV2EncoderConfig(**{**TINY, "num_hidden_layers": 1}, _frame_rate=12.5, rope_theta=10000.0)
    torch.manual_seed(3)
    transformer = _Transformer(config).eval()
    frames = torch.randn(1, window + 4, TINY["hidden_size"])
    changed = frames.clone()
    changed[:, 0] += 1.0
    with torch.inference_mode():
        before, after = transformer(frames), transformer(changed)
    assert not torch.equal(before[:, window - 1], after[:, window - 1])
    assert torch.equal(before[:, window:], after[:, window:])


def test_state_dict_uses_the_checkpoint_names():
    """Every name is what the Breeze checkpoint stores after its ``encoder.`` prefix."""
    names = {re.sub(r"\.\d+", ".N", name) for name in _tiny_encoder().state_dict()}
    expected = {
        "downsample.conv.weight",
        "encoder.layers.N.block.N.conv.bias",
        "encoder.layers.N.block.N.conv.weight",
        "encoder.layers.N.conv.bias",
        "encoder.layers.N.conv.weight",
    }
    for leaf in ("input_layernorm.bias", "input_layernorm.weight", "mlp.fc1.weight", "mlp.fc2.weight",
                 "mlp_layer_scale.scale", "post_attention_layernorm.bias", "post_attention_layernorm.weight",
                 "self_attn.k_proj.weight", "self_attn.o_proj.weight", "self_attn.q_proj.weight",
                 "self_attn.v_proj.weight", "self_attn_layer_scale.scale"):
        expected.add(f"encoder_transformer.layers.N.{leaf}")
    for kind in ("semantic", "acoustic"):
        for leaf in ("input_proj.weight", "output_proj.weight", "layers.N.codebook.cluster_usage",
                     "layers.N.codebook.embed_sum", "layers.N.codebook.initialized"):
            expected.add(f"quantizer.{kind}_residual_vector_quantizer.{leaf}")
    assert names == expected


def _transformers_mimi_parts(model):
    """(seanet, transformer, downsample, quantizer) of transformers' Mimi model, or None if unknown."""
    if isinstance(vars(type(model)).get("upsample"), property):  # 5.21+ layout
        inner = model.encoder
        return inner.encoder_seanet, inner.encoder_transformer, inner.downsample, model.quantizer
    if hasattr(model, "encoder_transformer"):  # layout up to 5.20
        return model.encoder, model.encoder_transformer, model.downsample, model.quantizer
    return None


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("samples", [1920, 12345, 24000, 48001])
def test_codes_match_transformers_mimi_exactly(attn, samples):
    """Our copy must stay numerically identical to the code the checkpoint was made for."""
    import transformers
    from transformers import MimiConfig, MimiModel

    config = MimiConfig(**TINY, frame_rate=12.5, use_cache=False, upsample_groups=TINY["hidden_size"],
                        rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
    config._attn_implementation = attn
    reference = MimiModel(config).eval()
    parts = _transformers_mimi_parts(reference)
    if parts is None:
        pytest.skip("transformers changed its Mimi layout: this comparison no longer runs, "
                    "so compare the codes on the real checkpoint by hand before releasing")
    ours = _tiny_encoder()
    if transformers.__version__.startswith("4."):
        # transformers 4 ignores sliding_window and always uses the full context. Compare everything
        # else; the window has its own test above. See the module docstring of audio_encoder.py.
        ours.encoder_transformer.sliding_window = 10**9
    seanet, transformer, downsample, quantizer = parts
    seanet.load_state_dict(ours.encoder.state_dict())
    transformer.load_state_dict(ours.encoder_transformer.state_dict())
    downsample.load_state_dict(ours.downsample.state_dict())
    quantizer.load_state_dict(ours.quantizer.state_dict())

    # Codes are rounded to the nearest codebook entry and hide small numeric differences, so compare
    # the attention stack's raw output as well.
    frames = torch.randn(1, 12, TINY["hidden_size"], generator=torch.Generator().manual_seed(samples))
    with torch.inference_mode():
        assert torch.allclose(ours.encoder_transformer(frames), transformer(frames)[0], atol=1e-5)

    audio = torch.randn(1, 1, samples, generator=torch.Generator().manual_seed(samples))
    with torch.inference_mode():
        expected = reference.encode(input_values=audio, return_dict=True).audio_codes
        codes = ours.encode(audio)
    assert codes.shape == expected.shape
    assert torch.equal(codes, expected)


def test_audio_tokenizer_survives_from_pretrained(tmp_path):
    """The load path that crashed: config.json + weights -> from_pretrained -> encode -> decode."""
    from breeze_models.qwen_tokenizer.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from breeze_models.qwen_tokenizer.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    config = Qwen3TTSTokenizerV2Config(
        encoder_config={**TINY, "_frame_rate": 12.5, "rope_theta": 10000.0},
        decoder_config={
            # The Breeze checkpoint's decoder_config, with only the sizes reduced.
            "attention_bias": False, "attention_dropout": 0.0, "latent_dim": 64, "codebook_dim": 16,
            "codebook_size": 16, "decoder_dim": 32, "hidden_act": "silu", "hidden_size": 32,
            "intermediate_size": 64, "layer_scale_initial_scale": 0.01, "max_position_embeddings": 128,
            "head_dim": 16, "num_attention_heads": 2, "num_hidden_layers": 1, "num_key_value_heads": 2,
            "num_quantizers": 4, "num_semantic_quantizers": 1, "rms_norm_eps": 1e-05, "rope_theta": 10000,
            "semantic_codebook_size": 16, "sliding_window": 5, "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2], "vector_quantization_hidden_dimension": 16,
        },
        encoder_valid_num_quantizers=4,
    )
    torch.manual_seed(1)
    model = Qwen3TTSTokenizerV2Model(config).eval()
    for name, buffer in model.named_buffers():
        if name.endswith("embed_sum") and name.startswith("encoder."):
            torch.nn.init.normal_(buffer)
    model.save_pretrained(tmp_path)

    loaded = Qwen3TTSTokenizerV2Model.from_pretrained(tmp_path).eval()

    saved = model.state_dict()
    for name, tensor in loaded.encoder.state_dict().items():
        assert torch.equal(tensor, saved[f"encoder.{name}"]), name
    audio = torch.randn(1, 4800, generator=torch.Generator().manual_seed(2))
    mask = torch.ones_like(audio, dtype=torch.long)
    with torch.inference_mode():
        out = loaded.encode(audio, mask, return_dict=True)
        again = model.encode(audio, mask, return_dict=True)
    assert torch.equal(out.audio_codes[0], again.audio_codes[0])
    codes = out.audio_codes[0]
    assert codes.shape == (3, 4)  # ceil(4800 / 1920) frames, 4 quantizers

    with torch.inference_mode():
        audio_out = loaded.decode(codes[None], return_dict=True).audio_values[0]
    assert audio_out.shape[-1] == 3 * 1920
    assert torch.isfinite(audio_out).all()


def test_no_transformers_docstring_validator():
    """It prints alarming [ERROR] lines for these classes on recent transformers."""
    package = ROOT.parent
    for path in package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        imports = re.findall(r"from transformers[\w.]* import (?:\([^)]*\)|.*)", source)
        assert not any("auto_docstring" in line for line in imports), path.name


def test_auto_docstring_stand_in_supports_both_forms():
    from breeze_models.docstring_compat import auto_docstring

    def function():
        pass

    assert auto_docstring(function) is function
    assert auto_docstring(custom_intro="text")(function) is function
