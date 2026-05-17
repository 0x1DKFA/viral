from __future__ import annotations

import gc
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
    "You are reviewing a {duration:.1f}-second gameplay clip for short-form viral content.\n"
    "Frames are sampled in chronological order.\n"
    "\n"
    "Viral gameplay moments include: multi-kills, clutch 1vN saves, ace rounds, "
    "improbable shots, perfect timing, hilarious deaths, glitchy/funny fails, jumpscares, "
    "rage moments, comeback wins, sick movement (bhop/wall-bang/no-scope), big damage numbers, "
    "boss kills, surprising reactions. Mundane gameplay (walking, looting, menus, loading, "
    "downtime) is NOT viral, even if technically competent.\n"
    "\n"
    "Pick a clip window that includes:\n"
    "  - 2-4 seconds of LEAD-IN (the setup before the moment)\n"
    "  - the PAYOFF itself\n"
    "  - 1-2 seconds of AFTERMATH (reaction, kill feed, scoreboard)\n"
    "The window MUST be at least {min_clip:.0f} seconds long and at most {max_clip:.0f} seconds.\n"
    "If no moment in this {duration:.1f}-second segment meets the bar, set is_highlight=false "
    "and score <= 4 — do NOT manufacture a highlight from filler.\n"
    "\n"
    "Return ONLY a single JSON object with these exact keys:\n"
    '{{\n'
    '  "is_highlight": boolean,\n'
    '  "score": integer 1-10 (10 = must-post, 7+ = worth posting, <=4 = skip),\n'
    '  "start_sec": float (window start, seconds from clip start),\n'
    '  "end_sec": float (window end, seconds from clip start; end_sec - start_sec >= {min_clip:.0f}),\n'
    '  "action_center_x": float 0.0-1.0 (horizontal center of the action; 0.5 = middle),\n'
    '  "title": string (<= 60 chars, punchy, no quotes, no clickbait emojis),\n'
    '  "description": string (1-2 sentences describing what happens),\n'
    '  "hashtags": list of 3-5 strings, each starting with #\n'
    '}}\n'
    "Output ONLY the JSON object, nothing else."
)

RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. Reply with ONLY the JSON object, "
    "no prose, no markdown fencing."
)

EXPLAIN_SIZING_FRAGMENT = (
    "\nAlso include these two extra keys in the JSON object:\n"
    '  "recommended_duration_sec": float (your ideal length for this clip in seconds, between {min_clip:.0f} and {max_clip:.0f}),\n'
    '  "sizing_reason": string (one short sentence explaining why this length is right, e.g. "dense action, keep tight" or "needs setup time")\n'
    "Set end_sec - start_sec close to recommended_duration_sec."
)


class VLMAnalyzer:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_frame_pixels: int = 480 * 854,  # ~480p; shrinks RAM/VRAM use a lot
        min_clip_sec: float = 8.0,
        max_clip_sec: float = 25.0,
        explain_sizing: bool = False,
    ):
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
        # low_cpu_mem_usage: stream weight shards directly onto the GPU instead of
        #   materializing full fp16 weights in CPU RAM first. Critical on 16GB hosts.
        # device_map={"": 0}: pin the whole model to GPU 0 so device_map="auto" can't
        #   silently offload pieces to CPU/disk (which causes swap thrash + OOM kill).
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map={"": 0},
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self.max_frame_pixels = max_frame_pixels
        self.min_clip_sec = min_clip_sec
        self.max_clip_sec = max_clip_sec
        self.explain_sizing = explain_sizing

    def _shrink_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        """Downscale frames so total pixel count per frame stays under the budget.

        The VLM doesn't need 1080p to judge a clip; shrinking to ~480p slashes the
        encoder activation memory dramatically with minimal quality loss for scoring.
        """
        if self.max_frame_pixels <= 0:
            return frames
        out: list[Image.Image] = []
        for f in frames:
            w, h = f.size
            pixels = w * h
            if pixels <= self.max_frame_pixels:
                out.append(f)
                continue
            scale = (self.max_frame_pixels / pixels) ** 0.5
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            out.append(f.resize(new_size, Image.BILINEAR))
        return out

    def _build_messages(
        self, frames: list[Image.Image], duration_sec: float, retry: bool
    ) -> list[dict[str, Any]]:
        prompt = PROMPT.format(
            duration=duration_sec,
            min_clip=self.min_clip_sec,
            max_clip=self.max_clip_sec,
        )
        if self.explain_sizing:
            prompt = prompt + EXPLAIN_SIZING_FRAGMENT.format(
                min_clip=self.min_clip_sec,
                max_clip=self.max_clip_sec,
            )
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

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        finally:
            # Release activation / KV-cache memory between chunks so the next call
            # doesn't compound. Cheap to do, big difference on tight-RAM systems.
            del inputs
            if "generated_ids" in locals():
                del generated_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def analyze(self, frames: list[Image.Image], duration_sec: float) -> Highlight | None:
        if not frames:
            return None

        frames = self._shrink_frames(frames)

        for retry in (False, True):
            messages = self._build_messages(frames, duration_sec, retry=retry)
            raw = self._generate(messages)
            highlight = parse_highlight(
                raw,
                chunk_duration_sec=duration_sec,
                min_clip_sec=self.min_clip_sec,
            )
            if highlight is not None:
                return highlight
            print(f"[analyzer] Unparseable output (retry={retry}): {raw[:200]!r}")

        return None
