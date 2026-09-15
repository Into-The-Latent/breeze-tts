from __future__ import annotations

from unittest.mock import MagicMock, patch

from breeze_infer.runtime import load_runtime


def test_load_runtime_disables_inapplicable_mistral_regex_fix(tmp_path) -> None:
    (tmp_path / "audio_tokenizer").mkdir()
    model = MagicMock()
    audio_tokenizer = object()

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
        ),
    ):
        load_runtime(tmp_path, device="cpu", attn_implementation="eager")

    load_tokenizer.assert_called_once_with(
        tmp_path,
        fix_mistral_regex=False,
    )
