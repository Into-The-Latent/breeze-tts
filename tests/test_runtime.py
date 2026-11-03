from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch

import pytest

from breeze_infer.runtime import load_runtime


def test_load_runtime_disables_inapplicable_mistral_regex_fix(tmp_path) -> None:
    (tmp_path / "audio_tokenizer").mkdir()
    model = MagicMock()
    audio_tokenizer = MagicMock()

    with (
        patch(
            "breeze_infer.runtime.AutoTokenizer.from_pretrained",
            return_value=object(),
        ) as load_tokenizer,
        patch(
            "breeze_infer.runtime.BreezeForConditionalGeneration.from_pretrained",
            return_value=model,
        ),
        patch(
            "breeze_models.qwen_tokenizer.Qwen3TTSTokenizer.from_pretrained",
            return_value=audio_tokenizer,
        ) as load_audio_tokenizer,
    ):
        load_runtime(tmp_path, device="cpu", attn_implementation="eager")

    load_tokenizer.assert_called_once_with(
        tmp_path,
        fix_mistral_regex=False,
    )
    # The audio tokenizer is loaded without device_map (that would require `accelerate`) and moved
    # afterwards, like the main model.
    load_audio_tokenizer.assert_called_once_with(str(tmp_path / "audio_tokenizer"))
    audio_tokenizer.model.to.assert_called_once_with("cpu")
    assert audio_tokenizer.device is audio_tokenizer.model.device


def _load_runtime_with_tokenizer_loader(tmp_path, load_tokenizer) -> None:
    (tmp_path / "audio_tokenizer").mkdir()
    with (
        patch("breeze_infer.runtime.AutoTokenizer.from_pretrained", load_tokenizer),
        patch(
            "breeze_infer.runtime.BreezeForConditionalGeneration.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "breeze_models.qwen_tokenizer.Qwen3TTSTokenizer.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        load_runtime(tmp_path, device="cpu", attn_implementation="eager")


def test_load_runtime_retries_without_the_flag_when_transformers_passes_it_twice(tmp_path) -> None:
    # transformers 5.3.0 forwards fix_mistral_regex to _patch_mistral_regex both by name and in
    # **kwargs, so any caller that sets it gets this TypeError for a >100k-vocab tokenizer.
    load_tokenizer = MagicMock(
        side_effect=[
            TypeError(
                "TokenizersBackend._patch_mistral_regex() got multiple values for keyword "
                "argument 'fix_mistral_regex'"
            ),
            object(),
        ]
    )

    _load_runtime_with_tokenizer_loader(tmp_path, load_tokenizer)

    assert load_tokenizer.call_args_list == [
        call(tmp_path, fix_mistral_regex=False),
        call(tmp_path),
    ]


def test_load_runtime_does_not_swallow_other_tokenizer_type_errors(tmp_path) -> None:
    load_tokenizer = MagicMock(side_effect=TypeError("unexpected keyword argument 'trust_me'"))

    with pytest.raises(TypeError, match="trust_me"):
        _load_runtime_with_tokenizer_loader(tmp_path, load_tokenizer)
    load_tokenizer.assert_called_once()


def _record_tokenizer_warnings():
    """Where transformers 5.3 logs the false warning, with a handler that keeps what gets through."""
    logger = logging.getLogger("transformers.tokenization_utils_tokenizers")
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Keep()
    logger.addHandler(handler)
    return logger, handler, records


def _fails_with_the_flag_then(after):
    def loader(*args, **kwargs):
        if "fix_mistral_regex" in kwargs:
            raise TypeError("got multiple values for keyword argument 'fix_mistral_regex'")
        return after()

    return loader


def test_tokenizer_fallback_hides_only_the_false_regex_warning(tmp_path):
    """Unset, transformers 5.3 warns about incorrect tokenization on this non-Mistral tokenizer."""
    logger, handler, records = _record_tokenizer_warnings()

    def load():
        logger.warning("The tokenizer you are loading from 'x' with an incorrect regex pattern: https://y")
        logger.warning("A genuine problem with the tokenizer files")
        return MagicMock()

    try:
        with patch("breeze_infer.runtime.AutoTokenizer.from_pretrained", _fails_with_the_flag_then(load)):
            from breeze_infer.runtime import _load_text_tokenizer

            _load_text_tokenizer(tmp_path)
        assert records == ["A genuine problem with the tokenizer files"]
        logger.warning("The tokenizer you are loading from 'z' with an incorrect regex pattern: https://y")
        assert len(records) == 2  # nothing stays filtered after the load
    finally:
        logger.removeHandler(handler)


def test_tokenizer_fallback_removes_its_filter_when_loading_fails(tmp_path):
    logger = logging.getLogger("transformers.tokenization_utils_tokenizers")
    before = list(logger.filters)

    def fail():
        raise OSError("broken tokenizer files")

    with patch("breeze_infer.runtime.AutoTokenizer.from_pretrained", _fails_with_the_flag_then(fail)):
        from breeze_infer.runtime import _load_text_tokenizer

        with pytest.raises(OSError):
            _load_text_tokenizer(tmp_path)
    assert logger.filters == before
