from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.common import error_payload, http_get_json

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"


class CurrencyConversion(BaseModel):
    amount: float
    from_currency: str
    to_currency: str
    rate: float
    converted_amount: float
    rate_date: str
    source: str = "frankfurter.dev (European Central Bank reference rates)"


class CurrencyInput(BaseModel):
    amount: float = Field(gt=0, description="Amount of money to convert")
    from_currency: str = Field(description="ISO 4217 code of the source currency, e.g. 'EUR'")
    to_currency: str = Field(description="ISO 4217 code of the target currency, e.g. 'JPY'")


@tool("convert_currency", args_schema=CurrencyInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """Convert an amount between two currencies using the latest European Central Bank reference rates. Use it for budget questions such as 'how much is 500 EUR in yen'."""
    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    if source == target:
        return CurrencyConversion(
            amount=amount, from_currency=source, to_currency=target, rate=1.0, converted_amount=amount, rate_date="today"
        ).model_dump()
    try:
        data = http_get_json(FRANKFURTER_URL, {"base": source, "symbols": target})
        rates = data.get("rates", {})
        if target not in rates:
            return error_payload("convert_currency", f"Unsupported currency pair {source}->{target}", supported="ECB currencies only")
        rate = float(rates[target])
        return CurrencyConversion(
            amount=amount,
            from_currency=source,
            to_currency=target,
            rate=rate,
            converted_amount=round(amount * rate, 2),
            rate_date=data.get("date", "unknown"),
        ).model_dump()
    except Exception as exc:
        return error_payload("convert_currency", f"Currency provider unavailable: {exc}", pair=f"{source}->{target}")
