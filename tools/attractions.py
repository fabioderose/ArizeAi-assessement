from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.common import error_payload, http_get_json
from tools.geocoding import geocode

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


class Attraction(BaseModel):
    name: str
    distance_from_center_m: int
    summary: str
    wikipedia_url: str


class AttractionsResult(BaseModel):
    city: str
    country: str
    source: str = "en.wikipedia.org geosearch"
    attractions: list[Attraction]


class AttractionsInput(BaseModel):
    city: str = Field(description="City to explore, e.g. 'Rome' or 'Kyoto, Japan'")
    radius_km: int = Field(default=5, ge=1, le=10, description="Search radius around the city center in kilometers")
    limit: int = Field(default=8, ge=1, le=15, description="Maximum number of attractions to return")


@tool("find_attractions", args_schema=AttractionsInput)
def find_attractions(city: str, radius_km: int = 5, limit: int = 8) -> dict[str, Any]:
    """Find notable attractions, landmarks and points of interest near a city center. Use it when the traveler asks what to see, visit or do in a destination."""
    try:
        location = geocode(city)
        if location is None:
            return error_payload("find_attractions", f"Could not find a location named {city!r}", city=city)
        geosearch = http_get_json(
            WIKIPEDIA_API_URL,
            {
                "action": "query",
                "list": "geosearch",
                "gscoord": f"{location.latitude}|{location.longitude}",
                "gsradius": radius_km * 1000,
                "gslimit": limit,
                "format": "json",
            },
        )
        pages = geosearch.get("query", {}).get("geosearch", [])
        if not pages:
            return AttractionsResult(city=location.name, country=location.country, attractions=[]).model_dump()
        page_ids = "|".join(str(page["pageid"]) for page in pages)
        extracts = http_get_json(
            WIKIPEDIA_API_URL,
            {
                "action": "query",
                "pageids": page_ids,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exsentences": 2,
                "inprop": "url",
                "format": "json",
            },
        )
        details = extracts.get("query", {}).get("pages", {})
        attractions = []
        for page in pages:
            detail = details.get(str(page["pageid"]), {})
            attractions.append(
                Attraction(
                    name=page["title"],
                    distance_from_center_m=int(page.get("dist", 0)),
                    summary=(detail.get("extract") or "").strip(),
                    wikipedia_url=detail.get("fullurl") or f"https://en.wikipedia.org/?curid={page['pageid']}",
                )
            )
        return AttractionsResult(city=location.name, country=location.country, attractions=attractions).model_dump()
    except Exception as exc:
        return error_payload("find_attractions", f"Attractions provider unavailable: {exc}")
