"""Token pricing and cost estimation.

Prices are USD per one million tokens. The built-in table is a convenience
snapshot - vendors change prices, so override it with ``--pricing prices.yaml``
or ``ARENA_PRICING`` for anything you rely on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from evals_arena.providers.base import Usage


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


DEFAULT_PRICES: dict[str, Price] = {
    # Anthropic
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(0.10, 0.50),
    # OpenAI
    "gpt-4.1": Price(2.00, 8.00),
    "gpt-4.1-mini": Price(0.40, 1.60),
    # Offline fakes get nominal prices so cost tracking is exercised end-to-end.
    "fake-strong": Price(3.00, 15.00),
    "fake-fast": Price(0.25, 1.25),
    "fake-sloppy": Price(0.50, 2.00),
}


class PriceTable:
    def __init__(self, prices: dict[str, Price] | None = None) -> None:
        self.prices = dict(DEFAULT_PRICES if prices is None else prices)

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceTable:
        table = cls()
        path = path or os.getenv("ARENA_PRICING")
        if path:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            for model, p in data.items():
                table.prices[str(model)] = Price(float(p["input"]), float(p["output"]))
        return table

    def lookup(self, model: str) -> Price | None:
        if model in self.prices:
            return self.prices[model]
        # Allow dated / suffixed variants: longest known prefix wins.
        candidates = [m for m in self.prices if model.startswith(m)]
        return self.prices[max(candidates, key=len)] if candidates else None

    def cost(self, model: str, usage: Usage) -> float:
        price = self.lookup(model)
        return price.cost(usage) if price else 0.0
