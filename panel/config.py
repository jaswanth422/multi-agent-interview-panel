"""Central configuration. Everything overridable by env so the demo never hard-fails."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # ---- LLM provider -------------------------------------------------
    # auto: pick the first provider we have credentials for, else local ollama.
    provider: str = field(default_factory=lambda: _env("PANEL_PROVIDER", "auto"))
    model: str = field(default_factory=lambda: _env("PANEL_MODEL", ""))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("PANEL_OLLAMA_MODEL", "qwen2.5:3b"))
    temperature: float = field(default_factory=lambda: _env_float("PANEL_TEMPERATURE", 0.35))
    max_tokens: int = field(default_factory=lambda: _env_int("PANEL_MAX_TOKENS", 700))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("PANEL_LLM_TIMEOUT", 90.0))
    llm_retries: int = field(default_factory=lambda: _env_int("PANEL_LLM_RETRIES", 2))
    seed: int = field(default_factory=lambda: _env_int("PANEL_SEED", 7))

    # ---- interview shape ---------------------------------------------
    rounds: int = field(default_factory=lambda: _env_int("PANEL_ROUNDS", 3))
    debate_rounds: int = field(default_factory=lambda: _env_int("PANEL_DEBATE_ROUNDS", 2))
    # 3.0 because the panel's own rubric defines 3 = "meets the bar". A bar of
    # 3.2 rejected candidates who met every competency, which is incoherent.
    hire_threshold: float = field(default_factory=lambda: _env_float("PANEL_HIRE_THRESHOLD", 3.0))

    # ---- storage ------------------------------------------------------
    db_path: str = field(default_factory=lambda: _env("PANEL_DB", str(ROOT / "artifacts" / "panel.sqlite3")))

    # ---- tools --------------------------------------------------------
    enable_web: bool = field(default_factory=lambda: _env("PANEL_ENABLE_WEB", "1") == "1")
    enable_github: bool = field(default_factory=lambda: _env("PANEL_ENABLE_GITHUB", "1") == "1")
    enable_speech: bool = field(default_factory=lambda: _env("PANEL_ENABLE_SPEECH", "0") == "1")
    code_exec_timeout_s: float = field(default_factory=lambda: _env_float("PANEL_CODE_TIMEOUT", 6.0))
    http_timeout_s: float = field(default_factory=lambda: _env_float("PANEL_HTTP_TIMEOUT", 8.0))

    # ---- guardrails ---------------------------------------------------
    require_human_approval: bool = field(
        default_factory=lambda: _env("PANEL_REQUIRE_APPROVAL", "1") == "1"
    )

    # ---- demo preset --------------------------------------------------
    # PANEL_FAST=1 trims how much transcript/ledger context each prompt
    # carries. Same loop, same tools, same rules, same gates - only the
    # context windows shrink, which cuts prompt-eval time on a local model.
    # Off by default so the evaluated configuration is the honest one.
    fast: bool = field(default_factory=lambda: _env("PANEL_FAST", "0") == "1")

    def ctx(self, normal: int) -> int:
        """Context budget in characters, halved in fast mode."""
        return max(400, normal // 2) if self.fast else normal

    def tok(self, normal: int) -> int:
        """Output-token budget. Generation, not prompt size, dominates
        latency on a small local model, so fast mode caps what each call is
        allowed to write. Shorter scorecards and arguments, identical logic."""
        return max(96, int(normal * 0.45)) if self.fast else normal

    def resolved(self) -> tuple[str, str]:
        """Return (provider, model) after auto-detection."""
        p = self.provider
        if p != "auto":
            return p, self.model or _default_model(p, self)
        for name, key in (
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
        ):
            if os.environ.get(key):
                return name, self.model or _default_model(name, self)
        return "ollama", self.model or self.ollama_model


def _default_model(provider: str, s: "Settings") -> str:
    return {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
        "groq": "llama-3.3-70b-versatile",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
        "gemini": "gemini-1.5-flash",
        "ollama": s.ollama_model,
        "mock": "mock-deterministic",
    }.get(provider, s.ollama_model)


SETTINGS = Settings()
