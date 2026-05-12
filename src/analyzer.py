from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src.models import Highlight, parse_highlight

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except ImportError:  # pragma: no cover - optional path, keeps tests importable on CPU
    process_vision_info = None  # type: ignore


PROMPT = (
    "You are reviewing a gameplay clip for short-form viral content.\n"
    "Frames are sampled at 1 fps in chronological order; the clip lasts {duration:.1f} seconds.\n"
    "Decide whether this clip contains a highlight worth posting as a 9:16 short.\n"
    "\n"
    "Return ONLY a single JSON object with these exact keys:\n"
    '{{\n'
    '  "is_highlight": boolean,\n'
    '  "score": integer 1-10 (10 = must-post),\n'
    '  "start_sec": float (seconds from clip start where the highlight begins),\n'
    '  "end_sec": float (seconds from clip start where the highlight ends),\n'
    '  "action_center_x": float 0.0-1.0 (horizontal center of the action; 0.5 = middle),\n'
    '  "title": string (<= 60 chars, punchy, no quotes),\n'
    '  "description": string (1-2 sentences),\n'
    '  "hashtags": list of 3-5 strings, each starting with #\n'
    '}}\n'
    "Do not output anything except the JSON object."
)

RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. Reply with ONLY the JSON object, "
    "no prose, no markdown fencing."
)


class VLMAnalyzer:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        if process_vision_info is None:
            raise RuntimeError(
                "qwen-vl-utils is required at runtime; install via requirements.txt"
            )

        print(f"Loading {model_id} in 4-bit...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id

    def _build_messages(
        self, frames: list[Image.Image], duration_sec: float, retry: bool
    ) -> list[dict[str, Any]]:
        prompt = PROMPT.format(duration=duration_sec)
        if retry:
            prompt = prompt + RETRY_SUFFIX
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": 1.0},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _unpack_vision(self, messages) -> tuple[Any, Any, dict]:
        """Tolerant unpack: qwen-vl-utils returns 2- or 3-tuples by version."""
        result = process_vision_info(messages)
        if isinstance(result, tuple) and len(result) == 3:
            image_inputs, video_inputs, video_kwargs = result
            return image_inputs, video_inputs, dict(video_kwargs or {})
        image_inputs, video_inputs = result  # type: ignore[misc]
        return image_inputs, video_inputs, {}

    def _generate(self, messages) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self._unpack_vision(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def analyze(self, frames: list[Image.Image], duration_sec: float) -> Highlight | None:
        if not frames:
            return None

        for retry in (False, True):
            messages = self._build_messages(frames, duration_sec, retry=retry)
            raw = self._generate(messages)
            highlight = parse_highlight(raw, chunk_duration_sec=duration_sec)
            if highlight is not None:
                return highlight
            print(f"[analyzer] Unparseable output (retry={retry}): {raw[:200]!r}")

        return None
