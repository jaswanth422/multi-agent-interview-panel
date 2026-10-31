"""Provider-agnostic LLM client with JSON-mode, repair-retry and usage metering.

Why this exists: the panel makes 30-60 model calls per interview. We need
(a) structured output we can trust, (b) a latency/token ledger for the
reliability report, and (c) graceful degradation so a live demo never dies.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import SETTINGS, Settings


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    failures: int = 0
    repairs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_s: float = 0.0
    per_label: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "json_repairs": self.repairs,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_latency_s": round(self.total_latency_s, 2),
            "mean_latency_s": round(self.total_latency_s / self.calls, 3) if self.calls else 0.0,
            "calls_by_label": dict(sorted(self.per_label.items())),
        }


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if not m:
        raise LLMError(f"no JSON found in model output: {text[:200]!r}")
    frag = m.group(0)
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        # common small-model slips: trailing commas, single quotes
        cleaned = re.sub(r",(\s*[}\]])", r"\1", frag)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMError(f"unparseable JSON: {frag[:200]!r}") from e


class LLM:
    """Thread-safe-ish synchronous LLM facade."""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or SETTINGS
        self.provider, self.model = self.s.resolved()
        self.usage = Usage()
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=self.s.llm_timeout_s)

    # -- public API -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "temperature": self.s.temperature}

    def text(self, system: str, user: str, *, label: str = "text",
             max_tokens: int | None = None, temperature: float | None = None) -> str:
        return self._call(system, user, label=label, json_mode=False,
                          max_tokens=max_tokens, temperature=temperature)

    def json(self, system: str, user: str, *, label: str = "json",
             max_tokens: int | None = None, temperature: float | None = None,
             fallback: Any = None) -> Any:
        """Call the model and parse JSON. One repair attempt, then `fallback`."""
        last_err: Exception | None = None
        for attempt in range(self.s.llm_retries + 1):
            try:
                raw = self._call(
                    system, user, label=label, json_mode=True,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return _extract_json(raw)
            except Exception as e:  # noqa: BLE001 - we deliberately degrade
                last_err = e
                with self._lock:
                    self.usage.repairs += 1
                user = (
                    user
                    + "\n\nYour previous reply was not valid JSON. "
                      "Reply with ONLY a single valid JSON value. No prose, no code fences."
                )
                time.sleep(0.2 * (attempt + 1))
        with self._lock:
            self.usage.failures += 1
        if fallback is not None:
            return fallback
        raise LLMError(f"json call '{label}' failed: {last_err}")

    # -- dispatch -------------------------------------------------------
    def _call(self, system: str, user: str, *, label: str, json_mode: bool,
              max_tokens: int | None, temperature: float | None) -> str:
        mt = max_tokens or self.s.max_tokens
        temp = self.s.temperature if temperature is None else temperature
        t0 = time.time()
        try:
            fn = getattr(self, f"_p_{self.provider}", None)
            if fn is None:
                raise LLMError(f"unknown provider {self.provider!r}")
            out, ptok, ctok = fn(system, user, mt, temp, json_mode)
        finally:
            dt = time.time() - t0
            with self._lock:
                self.usage.calls += 1
                self.usage.total_latency_s += dt
                self.usage.per_label[label] = self.usage.per_label.get(label, 0) + 1
        with self._lock:
            self.usage.prompt_tokens += ptok
            self.usage.completion_tokens += ctok
        return out

    # -- providers ------------------------------------------------------
    def _p_ollama(self, system, user, mt, temp, json_mode):
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"num_predict": mt, "temperature": temp, "seed": self.s.seed},
        }
        if json_mode:
            body["format"] = "json"
        r = self._client.post(f"{self.s.ollama_host}/api/chat", json=body)
        r.raise_for_status()
        d = r.json()
        return d["message"]["content"], d.get("prompt_eval_count", 0) or 0, d.get("eval_count", 0) or 0

    def _openai_compatible(self, base, key, system, user, mt, temp, json_mode):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": mt,
            "temperature": temp,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self._client.post(
            f"{base}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return (d["choices"][0]["message"]["content"],
                u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    def _p_openai(self, *a):
        return self._openai_compatible(
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ["OPENAI_API_KEY"], *a)

    def _p_groq(self, *a):
        return self._openai_compatible(
            "https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"], *a)

    def _p_openrouter(self, *a):
        return self._openai_compatible(
            "https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"], *a)

    def _p_anthropic(self, system, user, mt, temp, json_mode):
        if json_mode:
            user += "\n\nReply with ONLY a single valid JSON value."
        r = self._client.post(
            "https://api.anthropic.com/v1/messages",
            json={"model": self.model, "max_tokens": mt, "temperature": temp,
                  "system": system, "messages": [{"role": "user", "content": user}]},
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return (d["content"][0]["text"], u.get("input_tokens", 0), u.get("output_tokens", 0))

    def _p_gemini(self, system, user, mt, temp, json_mode):
        cfg: dict[str, Any] = {"maxOutputTokens": mt, "temperature": temp}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        r = self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": os.environ["GEMINI_API_KEY"]},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": cfg},
        )
        r.raise_for_status()
        d = r.json()
        um = d.get("usageMetadata") or {}
        txt = d["candidates"][0]["content"]["parts"][0]["text"]
        return txt, um.get("promptTokenCount", 0), um.get("candidatesTokenCount", 0)

    def _p_mock(self, system, user, mt, temp, json_mode):
        """Deterministic offline provider. Keeps CI and air-gapped demos alive."""
        if json_mode:
            return '{"_mock": true}', 0, 0
        return "[mock response]", 0, 0
