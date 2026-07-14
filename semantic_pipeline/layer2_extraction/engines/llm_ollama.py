"""Local LLMEngine backend: a small instruction-tuned model served by Ollama.

This is the local-mode semantic-extraction engine. Temperature 0 and a fixed
seed make it as deterministic as an LLM allows; the model id is recorded for
provenance and the prompt is recorded by each extractor. Production swaps the
model (or backend) in config, not here.

The engine is generic: it returns parsed JSON for any (system, user, schema)
request, so several extractors can share one model. It does not know about
entities or any specific task.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import ollama

from ...config import LLMConfig
from ..interfaces import LLMEngine


class OllamaLLM(LLMEngine):
    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._client = ollama.Client(host=cfg.host) if cfg.host else ollama.Client()
        try:
            self._version = version("ollama")
        except PackageNotFoundError:
            self._version = "unknown"

    @property
    def model_id(self) -> str:
        return f"ollama:{self._cfg.model}@client{self._version}"

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> Any:
        response = self._client.chat(
            model=self._cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # A JSON Schema constrains the output to valid, parseable JSON. Falls
            # back to plain "json" mode if no schema is given.
            format=schema if schema is not None else "json",
            options={
                "temperature": self._cfg.temperature,
                "seed": self._cfg.seed,
                "num_ctx": self._cfg.num_ctx,
            },
        )
        content = response.message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned non-JSON output ({exc}); first 200 chars: {content[:200]!r}"
            ) from exc
