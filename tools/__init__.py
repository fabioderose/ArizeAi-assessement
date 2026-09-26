from tools.attractions import find_attractions
from tools.currency import convert_currency
from tools.flights import search_flights
from tools.hotels import search_hotels
from tools.weather import get_weather_forecast
from tools.web_search import web_search

TRAVEL_TOOLS = [
    search_flights,
    search_hotels,
    get_weather_forecast,
    find_attractions,
    convert_currency,
    web_search,
]

__all__ = [
    "TRAVEL_TOOLS",
    "search_flights",
    "search_hotels",
    "get_weather_forecast",
    "find_attractions",
    "convert_currency",
    "web_search",
]
