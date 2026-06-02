"""Feed abstraction.

Everything downstream (model, strategy, metrics, server) consumes a stream of
`Tick` events and never cares where they came from. Replay of historical Alpaca
trades, a live websocket, and the synthetic unit-test generator all implement the
same `Feed` interface, so they are perfectly interchangeable.

A "tick" here is one mid-price observation. We deliberately use mid-price (not the
last trade) so that bid-ask bounce does not masquerade as avalanches; see the build
plan / SOC_Trading_Strategy.md for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class Tick:
    """One mid-price observation for a single symbol."""

    ts: float          # epoch seconds (or synthetic step index as float)
    symbol: str
    mid: float         # mid-price; the "height of the sandpile"


class Feed(Protocol):
    """A source of ticks. Iterating yields `Tick`s in time order."""

    def __iter__(self) -> Iterator[Tick]:
        ...
