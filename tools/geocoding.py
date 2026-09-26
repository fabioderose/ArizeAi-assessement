from pydantic import BaseModel

from tools.common import http_get_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class Location(BaseModel):
    name: str
    country: str
    admin1: str | None = None
    latitude: float
    longitude: float
    timezone: str


def geocode(place: str) -> Location | None:
    data = http_get_json(GEOCODING_URL, {"name": place, "count": 1, "language": "en", "format": "json"})
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    return Location(
        name=first.get("name", place),
        country=first.get("country", ""),
        admin1=first.get("admin1"),
        latitude=first["latitude"],
        longitude=first["longitude"],
        timezone=first.get("timezone", "UTC"),
    )
