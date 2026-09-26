# Produce traces (step 4 in instructions: "Step 4: Tracing Experiments")
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = ROOT / "data" / "queries.json"
OUTPUT_FILE = ROOT / "exports" / "query_results.json"
API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 180.0


def check_api(client: httpx.Client) -> None:
    try:
        health = client.get(f"{API_URL}/health")
        health.raise_for_status()
    except Exception as exc:
        sys.exit(f"API is not reachable at {API_URL}: {exc}. Start it with `make api` first.")
    print(f"API reachable at {API_URL} -> {health.json()}")


def run_session(client: httpx.Client, session: dict) -> list[dict]:
    session_id = f"{session['session']}-{uuid.uuid4().hex[:8]}"
    records = []
    print(f"\n=== session {session_id} ===")
    for turn_index, message in enumerate(session["turns"], start=1):
        print(f"[{turn_index}] user: {message}")
        started = time.perf_counter()
        try:
            response = client.post(f"{API_URL}/chat", json={"message": message, "session_id": session_id})
            response.raise_for_status()
            payload = response.json()
            elapsed = round(time.perf_counter() - started, 2)
            tools = ", ".join(call["name"] for call in payload["tool_calls"]) or "none"
            print(f"    tools: {tools} | {elapsed}s | trace {payload['trace_id']}")
            print(f"    assistant: {payload['response'][:300].replace(chr(10), ' ')}")
            records.append({"session_id": session_id, "turn": turn_index, "message": message, "elapsed_seconds": elapsed, **payload})
        except Exception as exc:
            print(f"    request failed: {exc}")
            records.append({"session_id": session_id, "turn": turn_index, "message": message, "error": str(exc)})
    return records


def main() -> None:
    sessions = json.loads(QUERIES_FILE.read_text())
    total_turns = sum(len(session["turns"]) for session in sessions)
    print(f"Running {total_turns} queries across {len(sessions)} sessions against {API_URL}")
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        check_api(client)
        results = [record for session in sessions for record in run_session(client, session)]
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    failures = [record for record in results if "error" in record]
    print(f"\nDone: {len(results) - len(failures)} succeeded, {len(failures)} failed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
