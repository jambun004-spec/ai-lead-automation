"""Search providers and the factory that selects one from Settings."""

from __future__ import annotations

from lead_intel.config import Settings
from lead_intel.search.base import SearchError, SearchProvider, SearchResult
from lead_intel.search.tavily import TavilySearchProvider


def build_search_provider(settings: Settings) -> SearchProvider:
    settings.require_search_config()
    if settings.search_provider == "tavily":
        return TavilySearchProvider(api_key=settings.tavily_api_key.get_secret_value())
    raise AssertionError(f"unhandled search provider {settings.search_provider!r}")  # pragma: no cover


__all__ = [
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "TavilySearchProvider",
    "build_search_provider",
]
