# Check if the tools are working as expected ("Step 3: Testing the Tools")
import json
from datetime import date, timedelta

import pytest

from tools import TRAVEL_TOOLS
from tools.currency import convert_currency
from tools.flights import search_flights
from tools.hotels import search_hotels
from tools.weather import get_weather_forecast


def in_days(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def test_every_tool_has_a_name_description_and_schema():
    for tool in TRAVEL_TOOLS:
        assert tool.name
        assert tool.description
        assert tool.args


def test_search_flights_returns_structured_sorted_options():
    result = search_flights.invoke({"origin": "Paris", "destination": "Tokyo", "departure_date": in_days(30), "passengers": 2})
    assert result["origin"] == "CDG"
    assert result["destination"] == "HND"
    prices = [option["total_price"] for option in result["options"]]
    assert prices == sorted(prices)
    assert all(option["total_price"] == round(option["price_per_passenger"] * 2, 2) for option in result["options"])
    assert json.loads(json.dumps(result))


def test_search_flights_is_deterministic():
    args = {"origin": "LHR", "destination": "JFK", "departure_date": in_days(45)}
    assert search_flights.invoke(args) == search_flights.invoke(args)


def test_search_flights_rejects_past_dates_with_structured_error():
    result = search_flights.invoke({"origin": "Paris", "destination": "Rome", "departure_date": "2020-01-01"})
    assert result["error"]
    assert result["tool"] == "search_flights"


def test_search_flights_rejects_bad_date_format():
    result = search_flights.invoke({"origin": "Paris", "destination": "Rome", "departure_date": "next friday"})
    assert "YYYY-MM-DD" in result["error"]


def test_search_hotels_filters_by_budget_and_stars():
    result = search_hotels.invoke(
        {"city": "Barcelona", "check_in": in_days(10), "check_out": in_days(13), "guests": 2, "max_price_per_night": 400, "min_stars": 3}
    )
    assert result["nights"] == 3
    assert all(option["stars"] >= 3 for option in result["options"])
    assert all(option["price_per_night"] <= 400 for option in result["options"])
    assert all(option["total_price"] == round(option["price_per_night"] * 3, 2) for option in result["options"])


def test_search_hotels_rejects_inverted_dates():
    result = search_hotels.invoke({"city": "Rome", "check_in": in_days(5), "check_out": in_days(3)})
    assert "check_out" in result["error"]


def test_convert_currency_same_currency_short_circuits():
    result = convert_currency.invoke({"amount": 100, "from_currency": "eur", "to_currency": "EUR"})
    assert result["converted_amount"] == 100
    assert result["rate"] == 1.0


@pytest.mark.network
def test_convert_currency_live_rate():
    result = convert_currency.invoke({"amount": 100, "from_currency": "EUR", "to_currency": "USD"})
    assert "error" not in result
    assert result["converted_amount"] > 0


@pytest.mark.network
def test_weather_forecast_live():
    result = get_weather_forecast.invoke({"city": "Lisbon", "days": 2})
    assert "error" not in result
    assert result["country"] == "Portugal"
    assert len(result["days"]) == 2
    assert result["days"][0]["conditions"]
