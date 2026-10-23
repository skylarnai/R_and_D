"""Web search agent: Tavily search used when the uploaded files can't answer."""
from functools import lru_cache

from langchain_tavily import TavilySearch
from langsmith import traceable

from app.config import get_settings
from app.schemas import WebResult


@lru_cache
def _get_tool() -> TavilySearch:
    return TavilySearch(
        max_results=get_settings().web_results,
        tavily_api_key=get_settings().tavily_api_key,
    )


@traceable(name="tavily_web_search", run_type="tool")
def search_web(query: str) -> list[WebResult]:
    response = _get_tool().invoke({"query": query})
    raw = response.get("results", []) if isinstance(response, dict) else []
    return [
        WebResult(
            title=item.get("title") or item.get("url", "web result"),
            url=item.get("url", ""),
            content=item.get("content", ""),
        )
        for item in raw
        if item.get("content")
    ]