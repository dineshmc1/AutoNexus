"""Provider-independent LLM adapters used only for framework explanations."""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


def _validated_text(value: Any) -> str:
    """Require providers to return substantive text, not ``None`` or blanks."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("LLM provider returned an empty response.")
    return value.strip()


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, context: dict[str, Any]) -> str:
        raise NotImplementedError


class CallableLLMProvider(LLMProvider):
    def __init__(self, function: Callable[..., str]):
        self.function = function

    def generate(self, prompt: str, *, context: dict[str, Any]) -> str:
        return _validated_text(
            self.function(prompt=prompt, context=context)
        )


class LiteLLMProvider(LLMProvider):
    def __init__(self, model: str, **completion_options: Any):
        self.model = model
        self.completion_options = completion_options

    def generate(self, prompt: str, *, context: dict[str, Any]) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLM support requires: pip install AutoNexus[llm]"
            ) from exc
        response = litellm.completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Use only the supplied run context. Do not invent "
                        "metrics or recommend unsafe automatic promotion."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{json.dumps(context, indent=2)}",
                },
            ],
            **self.completion_options,
        )
        return _validated_text(response.choices[0].message.content)


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, *, context: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{prompt}\n\n{json.dumps(context, indent=2)}",
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.timeout
        ) as response:
            return _validated_text(json.load(response)["response"])


class TransformersProvider(LLMProvider):
    """Run a local Hugging Face text-generation model lazily."""

    def __init__(
        self,
        model: str,
        *,
        max_new_tokens: int = 1200,
        device_map: str = "auto",
        **pipeline_options: Any,
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.pipeline_options = pipeline_options
        self._pipeline = None

    def generate(self, prompt: str, *, context: dict[str, Any]) -> str:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Local Transformers support requires AutoNexus[llm,vision]"
            ) from exc
        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-generation",
                model=self.model,
                device_map=self.device_map,
                **self.pipeline_options,
            )
        full_prompt = f"{prompt}\n\n{json.dumps(context, indent=2)}"
        result = self._pipeline(
            full_prompt,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        return _validated_text(result[0]["generated_text"])


class HTTPJSONProvider(LLMProvider):
    """Adapter for arbitrary JSON APIs through request/response callables."""

    def __init__(
        self,
        url: str,
        *,
        request_builder: Callable[[str, dict[str, Any]], dict[str, Any]],
        response_parser: Callable[[dict[str, Any]], str],
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ):
        self.url = url
        self.request_builder = request_builder
        self.response_parser = response_parser
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    def generate(self, prompt: str, *, context: dict[str, Any]) -> str:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(
                self.request_builder(prompt, context)
            ).encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.timeout
        ) as response:
            return _validated_text(self.response_parser(json.load(response)))
