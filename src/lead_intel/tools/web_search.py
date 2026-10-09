"""Provider-agnostic CrewAI web search tool."""

from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from lead_intel.search.base import SearchError, SearchProvider


class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, description="Web search query.")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for information about a business. Returns titles, URLs and "
        "content snippets. Always cite the returned URL for any fact you use."
    )
    args_schema: type[BaseModel] = WebSearchInput
    max_results: int = 5
    _provider: SearchProvider = PrivateAttr()

    def __init__(self, provider: SearchProvider, max_results: int = 5, **kwargs: Any) -> None:
        super().__init__(max_results=max_results, **kwargs)
        self._provider = provider

    def _run(self, query: str) -> str:
        try:
            results = self._provider.search(query, self.max_results)
        except SearchError as exc:  # report to the agent instead of crashing its loop
            return f"Search failed for '{query}': {exc}"
        if not results:
            return f"No results for '{query}'."
        return "\n\n".join(
            f"[{i}] {r.title}\nURL: {r.url}\n{r.content}" for i, r in enumerate(results, start=1)
        )
