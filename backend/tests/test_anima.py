"""Regression checks for Anima's published tokenizer layout."""

from unittest.mock import MagicMock, patch

import torch

from sodalite_backend.inference.anima import ANIMA_DIFFUSERS_MODEL_ID, load_anima_pipeline


def test_loads_fast_tokenizers_before_remaining_components() -> None:
    pipeline = MagicMock()
    events = []
    pipeline.update_components.side_effect = lambda **components: events.append(components)
    pipeline.load_components.side_effect = lambda **kwargs: events.append("load")

    with (
        patch("sodalite_backend.inference.anima.ModularPipeline.from_pretrained", return_value=pipeline),
        patch("sodalite_backend.inference.anima._load_transformer", return_value=object()),
        patch("sodalite_backend.inference.anima.Qwen2TokenizerFast.from_pretrained", return_value="qwen") as qwen,
        patch("sodalite_backend.inference.anima.T5TokenizerFast.from_pretrained", return_value="t5") as t5,
    ):
        load_anima_pipeline("checkpoint.safetensors", "cpu", torch.float32)

    qwen.assert_called_once_with(ANIMA_DIFFUSERS_MODEL_ID, subfolder="tokenizer")
    t5.assert_called_once_with(
        ANIMA_DIFFUSERS_MODEL_ID, subfolder="t5_tokenizer", extra_special_tokens={}
    )
    assert "transformer" in events[0]
    assert events[1:] == [{"tokenizer": "qwen", "t5_tokenizer": "t5"}, "load"]
