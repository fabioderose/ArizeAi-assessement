import json
from datetime import date, datetime
from typing import Any

import httpx
from pydantic import BaseModel

HTTP_TIMEOUT_SECONDS = 10.0
USER_AGENT = "travel-assistant-poc/0.2 (https://github.com/trevorlaviale/se-interview)"


class ToolError(BaseModel):
    error: str
    tool: str
    details: dict[str, Any] = {}


def error_payload(tool: str, message: str, **details: Any) -> dict[str, Any]:
    return ToolError(error=message, tool=tool, details=details).model_dump()


def parse_iso_date(value: str, field_name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must use the YYYY-MM-DD format, got {value!r}") from exc


def http_get_json(url: str, params: dict[str, Any]) -> Any:
    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def to_json(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, default=str)
