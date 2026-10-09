"""Search-provider interface. The research agent depends only on this."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str = ""
    url: str
    content: str = ""


class SearchError(RuntimeError):
    """A provider failed to return results (network, auth, quota...)."""


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Return results, or raise SearchError."""
        ...
