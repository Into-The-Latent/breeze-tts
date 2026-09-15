"""The vendored Qwen3-TTS 12 Hz tokenizer must import without the qwen_tts package."""
import sys


def test_vendored_tokenizer_imports_without_qwen_tts(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_tts", None)  # make `import qwen_tts` raise ImportError
    for name in [m for m in list(sys.modules) if m.startswith("breeze_models.qwen_tokenizer")]:
        monkeypatch.delitem(sys.modules, name)
    from breeze_models.qwen_tokenizer import (
        Qwen3TTSTokenizer,
        Qwen3TTSTokenizerV2Config,
        Qwen3TTSTokenizerV2Model,
    )
    assert callable(Qwen3TTSTokenizer.from_pretrained)
    assert Qwen3TTSTokenizerV2Config.model_type == "qwen3_tts_tokenizer_12hz"
    assert Qwen3TTSTokenizerV2Model.config_class is Qwen3TTSTokenizerV2Config


def test_compat_reexports_from_vendored_package():
    from breeze_models.stream_runtime.core import compat
    from breeze_models import qwen_tokenizer

    assert compat.Qwen3TTSTokenizer is qwen_tokenizer.Qwen3TTSTokenizer
    assert compat.Qwen3TTSTokenizerV2Decoder is qwen_tokenizer.Qwen3TTSTokenizerV2Decoder


def test_runtime_does_not_reference_qwen_tts_package():
    import inspect
    from breeze_infer import runtime

    assert "from qwen_tts" not in inspect.getsource(runtime)
