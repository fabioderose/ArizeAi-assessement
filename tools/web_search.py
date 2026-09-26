from typing import Any

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.common import error_payload


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query, e.g. 'visa requirements for French citizens visiting Japan'")


class WebResult(BaseModel):
    title: str
    snippet: str
    link: str


class WebSearchResult(BaseModel):
    query: str
    source: str = "duckduckgo"
    results: list[WebResult]

_search = DuckDuckGoSearchResults(output_format="list", num_results=5)


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> dict[str, Any]:
    """Search the web for general travel information that the other tools do not cover, such as visa rules, local customs, events, or transport tips."""
    try:
        raw = _search.invoke(query)
        results = [
            WebResult(title=item.get("title", ""), snippet=item.get("snippet", ""), link=item.get("link", ""))
            for item in raw
        ]
        return WebSearchResult(query=query, results=results).model_dump()
    except Exception as exc:
        return error_payload("web_search", f"Web search unavailable: {exc}", query=query)
