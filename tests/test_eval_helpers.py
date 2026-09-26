# Test if evals data are OK + helpers
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_evals import NO_HISTORY, build_turns, render_messages, tool_output_structured


def root_spans():
    frame = pd.DataFrame(
        {
            "attributes.session.id": ["s1", "s1", "s2"],
            "attributes.input.value": ["hi", "no, I said Tokyo", "weather in Rome"],
            "attributes.output.value": ["hello", "here are flights to Tokyo", "sunny"],
            "start_time": pd.to_datetime(["2026-09-23T10:00:00", "2026-09-23T10:01:00", "2026-09-23T10:02:00"]),
            "context.trace_id": ["t1", "t2", "t3"],
        },
        index=pd.Index(["a", "b", "c"], name="context.span_id"),
    )
    return frame


def test_build_turns_rebuilds_history_per_session():
    turns = build_turns(root_spans())
    by_span = turns.set_index("context.span_id")
    assert by_span.loc["a", "conversation"] == NO_HISTORY
    assert by_span.loc["b", "conversation"] == "User: hi\nAssistant: hello"
    assert by_span.loc["c", "conversation"] == NO_HISTORY
    assert list(by_span.loc[["a", "b"], "turn_index"]) == [1, 2]


def test_render_messages_reads_flattened_phoenix_format():
    messages = [
        {"message.role": "user", "message.content": "Flights to Rome"},
        {
            "message.role": "assistant",
            "message.tool_calls": [{"tool_call.function.name": "search_flights", "tool_call.function.arguments": '{"origin": "Paris"}'}],
        },
    ]
    text = render_messages(messages)
    assert "user: Flights to Rome" in text
    assert "[tool_call] search_flights" in text


def test_render_messages_reads_nested_format():
    messages = [{"message": {"role": "system", "content": "You are a travel assistant"}}]
    assert render_messages(messages) == "system: You are a travel assistant"


def test_tool_output_structured_labels():
    ok = tool_output_structured.evaluate({"output": '{"origin": "CDG"}'})[0]
    err = tool_output_structured.evaluate({"output": '{"error": "check_in is in the past", "tool": "search_hotels"}'})[0]
    bad = tool_output_structured.evaluate({"output": "not json"})[0]
    assert (ok.label, ok.score) == ("structured_ok", 1.0)
    assert (err.label, err.score) == ("tool_error", 0.0)
    assert (bad.label, bad.score) == ("invalid_json", 0.0)
