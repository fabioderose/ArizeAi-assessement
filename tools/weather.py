from datetime import date, timedelta
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.common import error_payload, http_get_json, parse_iso_date
from tools.geocoding import geocode

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_FORECAST_DAYS = 16

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class DailyForecast(BaseModel):
    date: str
    conditions: str
    weather_code: int
    temperature_max_c: float
    temperature_min_c: float
    precipitation_probability_max_pct: int | None = None
    wind_speed_max_kmh: float | None = None


class WeatherForecast(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    source: str = "open-meteo.com"
    days: list[DailyForecast]


class WeatherInput(BaseModel):
    city: str = Field(description="City name, optionally followed by the country, e.g. 'Lisbon, Portugal'")
    start_date: str | None = Field(
        default=None,
        description="First day of the forecast in YYYY-MM-DD format. Defaults to today. Forecasts cover at most 16 days ahead.",
    )
    days: int = Field(default=5, ge=1, le=7, description="Number of forecast days to return, between 1 and 7")


@tool("get_weather_forecast", args_schema=WeatherInput)
def get_weather_forecast(city: str, start_date: str | None = None, days: int = 5) -> dict[str, Any]:
    """Get the daily weather forecast for a city for the next days. Use it when the traveler asks about the weather, what to pack, or whether outdoor activities are realistic."""
    try:
        location = geocode(city)
        if location is None:
            return error_payload("get_weather_forecast", f"Could not find a location named {city!r}", city=city)
        first_day = parse_iso_date(start_date, "start_date") if start_date else date.today()
        horizon = date.today() + timedelta(days=MAX_FORECAST_DAYS - 1)
        if first_day < date.today() or first_day > horizon:
            return error_payload(
                "get_weather_forecast",
                f"Forecasts are only available from {date.today()} to {horizon}",
                requested_start_date=str(first_day),
            )
        last_day = min(first_day + timedelta(days=days - 1), horizon)
        data = http_get_json(
            FORECAST_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "timezone": location.timezone,
                "start_date": first_day.isoformat(),
                "end_date": last_day.isoformat(),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
            },
        )
        daily = data["daily"]
        forecast = WeatherForecast(
            city=location.name,
            country=location.country,
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=location.timezone,
            days=[
                DailyForecast(
                    date=daily["time"][i],
                    weather_code=daily["weather_code"][i],
                    conditions=WEATHER_CODES.get(daily["weather_code"][i], "unknown"),
                    temperature_max_c=daily["temperature_2m_max"][i],
                    temperature_min_c=daily["temperature_2m_min"][i],
                    precipitation_probability_max_pct=daily.get("precipitation_probability_max", [None] * len(daily["time"]))[i],
                    wind_speed_max_kmh=daily.get("wind_speed_10m_max", [None] * len(daily["time"]))[i],
                )
                for i in range(len(daily["time"]))
            ],
        )
        return forecast.model_dump()
    except ValueError as exc:
        return error_payload("get_weather_forecast", str(exc))
    except Exception as exc:
        return error_payload("get_weather_forecast", f"Weather provider unavailable: {exc}")
