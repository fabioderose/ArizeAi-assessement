import hashlib
import random
from datetime import date, datetime, timedelta
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.common import error_payload, parse_iso_date

AIRLINES = [
    ("AF", "Air France"),
    ("LH", "Lufthansa"),
    ("BA", "British Airways"),
    ("KL", "KLM"),
    ("IB", "Iberia"),
    ("DL", "Delta"),
    ("UA", "United"),
    ("EK", "Emirates"),
    ("QR", "Qatar Airways"),
    ("JL", "Japan Airlines"),
    ("SQ", "Singapore Airlines"),
    ("TP", "TAP Air Portugal"),
]

CABIN_MULTIPLIER = {"economy": 1.0, "premium_economy": 1.8, "business": 3.5, "first": 6.0}

CITY_TO_IATA = {
    "paris": "CDG",
    "london": "LHR",
    "new york": "JFK",
    "tokyo": "HND",
    "rome": "FCO",
    "madrid": "MAD",
    "barcelona": "BCN",
    "lisbon": "LIS",
    "berlin": "BER",
    "amsterdam": "AMS",
    "dubai": "DXB",
    "singapore": "SIN",
    "los angeles": "LAX",
    "san francisco": "SFO",
    "montreal": "YUL",
    "toronto": "YYZ",
    "bangkok": "BKK",
    "sydney": "SYD",
    "istanbul": "IST",
    "athens": "ATH",
    "kyoto": "KIX",
    "osaka": "KIX",
    "milan": "MXP",
    "zurich": "ZRH",
    "geneva": "GVA",
    "brussels": "BRU",
    "vienna": "VIE",
    "prague": "PRG",
    "copenhagen": "CPH",
    "reykjavik": "KEF",
    "marrakech": "RAK",
    "cairo": "CAI",
    "mexico city": "MEX",
    "rio de janeiro": "GIG",
    "buenos aires": "EZE",
    "seoul": "ICN",
    "hong kong": "HKG",
    "bali": "DPS",
}


class FlightOption(BaseModel):
    flight_number: str
    airline: str
    origin: str
    destination: str
    departure: str
    arrival: str
    duration_minutes: int
    stops: int
    cabin: str
    price_per_passenger: float
    total_price: float
    currency: str = "EUR"
    seats_left: int


class FlightSearchResult(BaseModel):
    origin: str
    destination: str
    departure_date: str
    passengers: int
    cabin: str
    source: str = "simulated inventory (deterministic, for demo purposes)"
    options: list[FlightOption]


class FlightSearchInput(BaseModel):
    origin: str = Field(description="Departure city or IATA airport code, e.g. 'Paris' or 'CDG'")
    destination: str = Field(description="Arrival city or IATA airport code, e.g. 'Tokyo' or 'HND'")
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    passengers: int = Field(default=1, ge=1, le=9, description="Number of passengers")
    cabin: Literal["economy", "premium_economy", "business", "first"] = Field(default="economy", description="Cabin class")
    max_price: float | None = Field(default=None, gt=0, description="Optional maximum total price in EUR")


def resolve_airport(place: str) -> str:
    cleaned = place.strip()
    if len(cleaned) == 3 and cleaned.isalpha():
        return cleaned.upper()
    return CITY_TO_IATA.get(cleaned.lower(), cleaned[:3].upper())


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16)


def build_options(origin: str, destination: str, departure_date: date, passengers: int, cabin: str) -> list[FlightOption]:
    rng = random.Random(stable_seed(origin, destination, departure_date.isoformat()))
    base_duration = rng.randint(90, 780)
    base_price = 60 + base_duration * rng.uniform(0.25, 0.55)
    options = []
    for _ in range(rng.randint(3, 5)):
        code, name = rng.choice(AIRLINES)
        stops = rng.choices([0, 1, 2], weights=[55, 35, 10])[0]
        duration = base_duration + stops * rng.randint(60, 180)
        departure = datetime.combine(departure_date, datetime.min.time()) + timedelta(hours=rng.randint(6, 22), minutes=rng.choice([0, 15, 30, 45]))
        arrival = departure + timedelta(minutes=duration)
        price = round(base_price * rng.uniform(0.8, 1.4) * CABIN_MULTIPLIER[cabin] * (0.85 if stops else 1.0), 2)
        options.append(
            FlightOption(
                flight_number=f"{code}{rng.randint(100, 999)}",
                airline=name,
                origin=origin,
                destination=destination,
                departure=departure.isoformat(timespec="minutes"),
                arrival=arrival.isoformat(timespec="minutes"),
                duration_minutes=duration,
                stops=stops,
                cabin=cabin,
                price_per_passenger=price,
                total_price=round(price * passengers, 2),
                seats_left=rng.randint(1, 9),
            )
        )
    return sorted(options, key=lambda option: option.total_price)


@tool("search_flights", args_schema=FlightSearchInput)
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    passengers: int = 1,
    cabin: str = "economy",
    max_price: float | None = None,
) -> dict[str, Any]:
    """Search available flights between two cities on a given date. Returns structured flight options sorted by price. Use it whenever the traveler wants to fly somewhere or asks how much a flight costs."""
    try:
        departure = parse_iso_date(departure_date, "departure_date")
    except ValueError as exc:
        return error_payload("search_flights", str(exc))
    if departure < date.today():
        return error_payload("search_flights", "departure_date is in the past", departure_date=departure_date, today=date.today().isoformat())
    origin_code = resolve_airport(origin)
    destination_code = resolve_airport(destination)
    if origin_code == destination_code:
        return error_payload("search_flights", "origin and destination resolve to the same airport", airport=origin_code)
    options = build_options(origin_code, destination_code, departure, passengers, cabin)
    if max_price is not None:
        options = [option for option in options if option.total_price <= max_price]
    return FlightSearchResult(
        origin=origin_code,
        destination=destination_code,
        departure_date=departure.isoformat(),
        passengers=passengers,
        cabin=cabin,
        options=options,
    ).model_dump()
