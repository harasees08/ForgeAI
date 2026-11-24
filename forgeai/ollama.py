"""Minimal client for the local Ollama server."""

from dataclasses import dataclass
import json
import re
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


DEFAULT_MODEL = "qwen3:4b"


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResponse:
    message: dict
    wall_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None
    output_tokens_per_second: float | None
    load_ns: int | None


class Ollama:
    def __init__(self, model: str = DEFAULT_MODEL, *, timeout: float = 90):
        self.model = model.strip()
        if not self.model or re.search(r"(?:^|[/:._-])cloud(?:$|[/:._-])", self.model, re.I):
            raise ValueError("Choose an installed local Ollama model.")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))
        self._verified_local = False

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        request = Request(
            f"http://127.0.0.1:11434/api/{endpoint}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=self.timeout if payload is not None else 5) as response:
                result = json.load(response)
        except HTTPError as error:
            raise OllamaError(f"Ollama HTTP {error.code} for /api/{endpoint}. Check the local model and server.") from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise OllamaError(f"Ollama /api/{endpoint} failed: {error}") from error
        if not isinstance(result, dict):
            raise OllamaError(f"Unexpected response from /api/{endpoint}.")
        return result

    def chat(self, messages: list[dict], *, schema: dict | None = None,
             tools: list[dict] | None = None) -> ChatResponse:
        if not self._verified_local:
            details = self._request("show", {"model": self.model})
            if details.get("remote_host") or details.get("remote_model"):
                raise OllamaError("Choose a local-only Ollama model. This tag points to a remote model.")
            self._verified_local = True
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "think": False, "keep_alive": "1m",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 500},
        }
        if schema is not None:
            payload["format"] = schema
        if tools is not None:
            payload["tools"] = tools
        start = perf_counter()
        result = self._request("chat", payload)
        wall_seconds = perf_counter() - start
        message = result.get("message")
        if result.get("done") is not True or not isinstance(message, dict):
            raise OllamaError("Ollama returned an incomplete chat response.")
        tokens = result.get("eval_count")
        eval_ns = result.get("eval_duration")
        return ChatResponse(
            message, wall_seconds, result.get("prompt_eval_count"), tokens,
            tokens * 1e9 / eval_ns if tokens is not None and eval_ns else None, result.get("load_duration"),
        )

    def models(self) -> list[dict]:
        return self._request("tags").get("models", [])

    def placement(self) -> dict | None:
        name = self.model if ":" in self.model else f"{self.model}:latest"
        for running in self._request("ps").get("models", []):
            if running.get("name") != name:
                continue
            size = running.get("size")
            vram = running.get("size_vram")
            status = "unknown"
            if vram == 0:
                status = "CPU"
            elif size and vram:
                status = "GPU" if vram >= size else "partial GPU"
            return {
                "status": status, "model": name, "digest": running.get("digest"),
                "vram_bytes": vram, "loaded_bytes": size, "context_length": running.get("context_length"),
            }
        return None
