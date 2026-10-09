"""Tavily implementation of SearchProvider."""

from __future__ import annotations

from tavily import TavilyClient

from lead_intel.search.base import SearchError, SearchResult


class TavilySearchProvider:
    name = "tavily"

    def __init__(self, api_key: str, client: TavilyClient | None = None) -> None:
        if not api_key:
            raise ValueError("Tavily API key is required")
        self._client = client or TavilyClient(api_key=api_key)  # no network call

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            response = self._client.search(query=query, max_results=max_results, search_depth="basic")
        except Exception as exc:
            raise SearchError(f"{type(exc).__name__}: {exc}") from exc
        return [
            SearchResult(title=r.get("title") or "", url=r["url"], content=r.get("content") or "")
            for r in response.get("results", [])
            if r.get("url")
        ]
