import random
from datetime import date
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.common import error_payload, parse_iso_date
from tools.flights import stable_seed

HOTEL_NAME_PARTS = (
    ["Grand", "Hotel", "The", "Maison", "Palazzo", "Riad", "Ryokan", "Casa", "Villa", "Boutique"],
    ["Central", "Royal", "Garden", "Harbour", "Old Town", "Riverside", "Skyline", "Plaza", "Botanic", "Lumière"],
)
NEIGHBORHOODS = ["Old Town", "City Center", "Waterfront", "Arts District", "Station Quarter", "Hillside", "Market Square"]
AMENITIES = ["free wifi", "breakfast included", "pool", "spa", "gym", "rooftop bar", "airport shuttle", "family rooms", "pet friendly", "coworking lounge"]


class HotelOption(BaseModel):
    name: str
    stars: int
    neighborhood: str
    guest_rating: float
    price_per_night: float
    total_price: float
    currency: str = "EUR"
    amenities: list[str]
    free_cancellation: bool


class HotelSearchResult(BaseModel):
    city: str
    check_in: str
    check_out: str
    nights: int
    guests: int
    source: str = "simulated inventory (deterministic, for demo purposes)"
    options: list[HotelOption]


class HotelSearchInput(BaseModel):
    city: str = Field(description="Destination city, e.g. 'Lisbon'")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out: str = Field(description="Check-out date in YYYY-MM-DD format")
    guests: int = Field(default=2, ge=1, le=8, description="Number of guests")
    max_price_per_night: float | None = Field(default=None, gt=0, description="Optional budget cap per night in EUR")
    min_stars: int = Field(default=1, ge=1, le=5, description="Minimum hotel star rating")


@tool("search_hotels", args_schema=HotelSearchInput)
def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    guests: int = 2,
    max_price_per_night: float | None = None,
    min_stars: int = 1,
) -> dict[str, Any]:
    """Search hotels in a city for a date range. Returns structured hotel options with price, rating and amenities. Use it when the traveler asks where to stay or about accommodation prices."""
    try:
        start = parse_iso_date(check_in, "check_in")
        end = parse_iso_date(check_out, "check_out")
    except ValueError as exc:
        return error_payload("search_hotels", str(exc))
    if end <= start:
        return error_payload("search_hotels", "check_out must be after check_in", check_in=check_in, check_out=check_out)
    if start < date.today():
        return error_payload("search_hotels", "check_in is in the past", check_in=check_in, today=date.today().isoformat())
    nights = (end - start).days
    rng = random.Random(stable_seed("hotels", city.strip().lower(), start.isoformat()))
    options = []
    for _ in range(rng.randint(4, 6)):
        stars = rng.choices([2, 3, 4, 5], weights=[15, 40, 30, 15])[0]
        price = round(rng.uniform(45, 90) * (stars ** 1.35) / 2.5 * (1 + 0.15 * (guests - 1)), 2)
        options.append(
            HotelOption(
                name=f"{rng.choice(HOTEL_NAME_PARTS[0])} {rng.choice(HOTEL_NAME_PARTS[1])}",
                stars=stars,
                neighborhood=rng.choice(NEIGHBORHOODS),
                guest_rating=round(rng.uniform(6.5, 9.7), 1),
                price_per_night=price,
                total_price=round(price * nights, 2),
                amenities=sorted(rng.sample(AMENITIES, k=rng.randint(2, 5))),
                free_cancellation=rng.random() > 0.35,
            )
        )
    options = [option for option in options if option.stars >= min_stars]
    if max_price_per_night is not None:
        options = [option for option in options if option.price_per_night <= max_price_per_night]
    return HotelSearchResult(
        city=city.strip(),
        check_in=start.isoformat(),
        check_out=end.isoformat(),
        nights=nights,
        guests=guests,
        options=sorted(options, key=lambda option: option.price_per_night),
    ).model_dump()
