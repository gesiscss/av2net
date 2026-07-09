"""Local VisionInterpreter backend: a vision-language model served by Ollama.

The frame is JPEG-encoded in memory and sent to the model. Temperature defaults
to 0 for as-deterministic-as-possible captions, though LLM output is never fully
deterministic; the prompt and model id are recorded for provenance.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import cv2
import numpy as np
import ollama

from ..config import VisionConfig
from ..language import language_name
from .interfaces import VisionInterpreter


class OllamaVisionInterpreter(VisionInterpreter):
    def __init__(self, cfg: VisionConfig, caption_language: str | None = None):
        self._cfg = cfg
        self._caption_language = caption_language
        # Compose the final prompt, aligning the caption language when requested.
        prompt = cfg.prompt
        if caption_language:
            prompt = f"{prompt} Respond in {language_name(caption_language)}."
        self._prompt = prompt
        self._client = ollama.Client(host=cfg.host) if cfg.host else ollama.Client()
        try:
            self._version = version("ollama")
        except PackageNotFoundError:
            self._version = "unknown"
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    @property
    def model_id(self) -> str:
        return f"ollama:{self._cfg.model}@client{self._version}"

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def caption_language(self) -> str | None:
        return self._caption_language

    def describe(self, image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", image, self._encode_params)
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame for the vision model")
        response = self._client.chat(
            model=self._cfg.model,
            messages=[
                {
                    "role": "user",
                    "content": self._prompt,
                    "images": [buf.tobytes()],
                }
            ],
            options={"temperature": self._cfg.temperature},
        )
        return response.message.content.strip()
