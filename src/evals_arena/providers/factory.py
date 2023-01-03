"""Resolve provider specs such as ``anthropic:claude-opus-5-5`` or ``fake:fake-strong``."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from evals_arena.providers.base import Provider, ProviderError
from evals_arena.providers.fake import FakeProvider, load_fake_scripts

VENDORS = ("fake", "anthropic", "openai")


def parse_spec(spec: str | None) -> tuple[str, str | None]:
    """Split ``vendor:model``; infer the vendor from well-known model prefixes."""
    spec = (spec or os.getenv("ARENA_MODEL") or "").strip()
    if not spec:
        vendor = os.getenv("ARENA_PROVIDER", "fake")
        return vendor, None
    if ":" in spec:
        vendor, model = spec.split(":", 1)
        return vendor.lower(), model or None
    lowered = spec.lower()
    if lowered.startswith("claude"):
        return "anthropic", spec
    if lowered.startswith(("gpt", "o1", "o3", "o4")):
        return "openai", spec
    if lowered.startswith("fake"):
        return "fake", spec
    return os.getenv("ARENA_PROVIDER", "fake"), spec


def create_provider(spec: str | None, *, fake_script_paths: Sequence[str | Path] = ()) -> Provider:
    vendor, model = parse_spec(spec)
    if vendor == "fake":
        name = model or "fake-strong"
        return FakeProvider(name, load_fake_scripts(fake_script_paths, name))
    if vendor == "anthropic":
        from evals_arena.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model)
    if vendor == "openai":
        from evals_arena.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model)
    raise ProviderError(f"unknown provider '{vendor}' (expected one of {', '.join(VENDORS)})")
