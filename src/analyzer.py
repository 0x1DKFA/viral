from __future__ import annotations

import gc
import logging
import time
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src.models import Highlight, ScoutRegion, parse_highlight
from src.scout import SCOUT_PROMPT, SCOUT_RETRY_SUFFIX, parse_scout_regions

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except ImportError:  # pragma: no cover - optional path, keeps tests importable on CPU
    process_vision_info = None  # type: ignore

logger = logging.getLogger(__name__)


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
    "If a real highlight is present, pick the natural window of the moment:\n"
    "  - 2-4 seconds of LEAD-IN (the setup before the moment)\n"
    "  - the PAYOFF itself\n"
    "  - 1-2 seconds of AFTERMATH (reaction, kill feed, scoreboard)\n"
    "Typical natural windows are 5-15 seconds. Do not exceed {max_clip:.0f} seconds.\n"
    "We will pad your window on our side so the final saved clip is at least {min_clip:.0f} seconds — "
    "you do NOT need to stretch the window to {min_clip:.0f} seconds yourself. "
    "Pick the moment that's actually there; we'll handle minimum length.\n"
    "If no real highlight is present, set is_highlight=false and score <= 4 — "
    "do NOT manufacture a highlight from filler.\n"
    "\n"
    "Return ONLY a single JSON object with these exact keys:\n"
    '{{\n'
    '  "is_highlight": boolean,\n'
    '  "score": integer 1-10 (10 = must-post, 7+ = worth posting, <=4 = skip),\n'
    '  "start_sec": float (natural window start, seconds from clip start),\n'
    '  "end_sec": float (natural window end, seconds from clip start),\n'
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

REGION_HINT = (
    "A coarse scout pass tagged this region as: {region_type}.\n"
    "Confirm whether it's actually viral, refine the start/end if needed.\n\n"
)


class VLMAnalyzer:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        min_clip_sec: float = 8.0,
        max_clip_sec: float = 25.0,
        explain_sizing: bool = False,
    ):
        if process_vision_info is None:
            raise RuntimeError(
                "qwen-vl-utils is required at runtime; install via requirements.txt"
            )

        logger.info("loading %s in 4-bit...", model_id)
        t0 = time.time()
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
        self.min_clip_sec = min_clip_sec
        self.max_clip_sec = max_clip_sec
        self.explain_sizing = explain_sizing
        logger.info("model loaded in %.1fs on device=%s", time.time() - t0, self.model.device)

    @staticmethod
    def shrink_frames(
        frames: list[Image.Image], max_frame_pixels: int
    ) -> list[Image.Image]:
        """Downscale frames so width*height <= max_frame_pixels per frame.

        Caller picks the budget — detail pass uses ~480p, scout uses ~240p.
        """
        if max_frame_pixels <= 0:
            return frames
        out: list[Image.Image] = []
        for f in frames:
            w, h = f.size
            pixels = w * h
            if pixels <= max_frame_pixels:
                out.append(f)
                continue
            scale = (max_frame_pixels / pixels) ** 0.5
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            out.append(f.resize(new_size, Image.BILINEAR))
        return out

    def _build_messages(
        self, frames: list[Image.Image], prompt: str, fps_used: float
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": fps_used},
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

    def _generate(
        self,
        messages,
        frame_count: int,
        duration_sec: float,
        fps_used: float,
        max_new_tokens: int = 256,
    ) -> str:
        t0 = time.time()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.debug(
            "prompt prepared: %d chars, frames=%d duration=%.1fs fps=%.2f",
            len(text), frame_count, duration_sec, fps_used,
        )
        image_inputs, video_inputs, video_kwargs = self._unpack_vision(messages)

        # Surface real video metadata so the Qwen3-VL processor doesn't warn and
        # silently assume source fps=24 (which would mis-place MRoPE temporal
        # positions for our pre-sampled frames).
        # NOTE: VideoMetadata's frame-count field is named `total_num_frame`
        # (singular) in the installed transformers version. Don't "fix" to
        # `total_frames` — that raises `unexpected keyword argument`.
        video_kwargs.setdefault(
            "video_metadata",
            [{
                "fps": fps_used,
                "total_num_frame": frame_count,
                "duration": duration_sec,
            }],
        )

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
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            logger.debug(
                "generation done in %.1fs (%d new token(s)); output: %r",
                time.time() - t0, trimmed[0].shape[0], decoded[:300],
            )
            return decoded
        finally:
            del inputs
            if "generated_ids" in locals():
                del generated_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def scout(
        self,
        frames: list[Image.Image],
        window_duration_sec: float,
        fps_used: float,
        frame_pixels_budget: int = 240 * 432,
    ) -> list[ScoutRegion]:
        """Identify viral candidate regions within a scout window.

        Returns regions with start_sec/end_sec relative to the window start.
        Pipeline adds the window offset to convert to absolute time.
        """
        if not frames:
            return []
        frames = self.shrink_frames(frames, frame_pixels_budget)

        for retry in (False, True):
            prompt = SCOUT_PROMPT.format(window_duration=window_duration_sec)
            if retry:
                prompt = prompt + SCOUT_RETRY_SUFFIX
            messages = self._build_messages(frames, prompt=prompt, fps_used=fps_used)
            raw = self._generate(
                messages,
                frame_count=len(frames),
                duration_sec=window_duration_sec,
                fps_used=fps_used,
                max_new_tokens=512,  # scout output is a list; needs more headroom
            )
            regions = parse_scout_regions(raw, window_duration_sec=window_duration_sec)
            if regions or not retry:
                # Empty list on first try might be legitimate (no highlights in window).
                # Only retry if the first attempt looked like a parse failure (no JSON at all).
                if regions:
                    return regions
                if "{" not in raw:
                    logger.warning(
                        "scout: no JSON found in output (retry=%s): %r", retry, raw[:200]
                    )
                    continue
                return []
        return []

    def analyze(
        self,
        frames: list[Image.Image],
        duration_sec: float,
        fps_used: float,
        frame_pixels_budget: int = 480 * 854,
        region_type: str | None = None,
    ) -> Highlight | None:
        if not frames:
            return None

        frames = self.shrink_frames(frames, frame_pixels_budget)

        for retry in (False, True):
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
            if region_type:
                prompt = REGION_HINT.format(region_type=region_type) + prompt
            if retry:
                prompt = prompt + RETRY_SUFFIX
            messages = self._build_messages(frames, prompt=prompt, fps_used=fps_used)
            raw = self._generate(
                messages,
                frame_count=len(frames),
                duration_sec=duration_sec,
                fps_used=fps_used,
            )
            highlight = parse_highlight(
                raw,
                chunk_duration_sec=duration_sec,
                min_clip_sec=self.min_clip_sec,
            )
            if highlight is not None:
                return highlight
            logger.warning("analyzer: unparseable output (retry=%s): %r", retry, raw[:200])

        return None
