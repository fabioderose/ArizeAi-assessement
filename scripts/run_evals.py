import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.evals import LLM, ClassificationEvaluator, create_evaluator, evaluate_dataframe
from phoenix.evals.metrics import ToolSelectionEvaluator, UserFrictionEvaluator
from phoenix.evals.utils import to_annotation_dataframe

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import TRAVEL_TOOLS
from tracing import collector_endpoint, project_name

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "exports"
ROOT_SPAN_NAME = "travel_assistant"
NO_HISTORY = "(first message of the conversation, no prior assistant behavior)"
NO_TOOL_RESULTS = "(no tool was called during this turn)"

HALLUCINATION_TEMPLATE = """You judge whether an assistant answer contains information that is not supported by the tool results it received.

hallucination: the answer states a fact (price, flight number, airline, hotel name, temperature, exchange rate, place, date) that does not appear in the tool results, or contradicts them, or gives such facts when no tool was called.
no_hallucination: every factual claim in the answer is present in the tool results, or the answer only asks a question, gives general knowledge, or says that the information is not available.

Rules:
- Rounding, unit conversion and rephrasing are not hallucinations.
- A recommendation or an opinion is not a hallucination.
- When no tool was called and the answer gives specific prices, schedules or weather, label hallucination.

<user_message>
{{user_message}}
</user_message>
<tool_results>
{{tool_results}}
</tool_results>
<assistant_answer>
{{assistant_response}}
</assistant_answer>

Answer with exactly one word: hallucination or no_hallucination."""


def judge_llm() -> LLM:
    return LLM(provider="openai", model=os.getenv("EVAL_MODEL", "gpt-4o-mini"))


def fetch_spans(client: Client, condition: str) -> pd.DataFrame:
    return client.spans.get_spans_dataframe(
        project_identifier=project_name(),
        query=SpanQuery().where(condition),
        limit=10000,
    )


def hallucination_judge(llm: LLM) -> ClassificationEvaluator:
    return ClassificationEvaluator(
        name="hallucination",
        prompt_template=HALLUCINATION_TEMPLATE,
        llm=llm,
        choices={"hallucination": 1.0, "no_hallucination": 0.0},
        direction="minimize",
    )


def attach_tool_results(turns: pd.DataFrame, tool_spans: pd.DataFrame) -> pd.DataFrame:
    if tool_spans.empty:
        turns["tool_results"] = NO_TOOL_RESULTS
        return turns
    grouped = tool_spans.groupby("context.trace_id")["attributes.output.value"].apply(lambda values: "\n".join(str(v) for v in values))
    turns["tool_results"] = turns["context.trace_id"].map(grouped).fillna(NO_TOOL_RESULTS)
    return turns


def build_turns(root_spans: pd.DataFrame) -> pd.DataFrame:
    turns = root_spans.copy()
    turns["context.span_id"] = turns.index
    turns["session_id"] = turns["attributes.session.id"].fillna(turns["context.trace_id"])
    turns["user_message"] = turns["attributes.input.value"].fillna("")
    turns["assistant_response"] = turns["attributes.output.value"].fillna("")
    turns = turns.sort_values(["session_id", "start_time"])
    conversations = []
    for _, group in turns.groupby("session_id", sort=False):
        history: list[str] = []
        for _, row in group.iterrows():
            conversations.append("\n".join(history) if history else NO_HISTORY)
            history.append(f"User: {row['user_message']}\nAssistant: {row['assistant_response']}")
    turns["conversation"] = conversations
    turns["turn_index"] = turns.groupby("session_id").cumcount() + 1
    return turns.reset_index(drop=True)


def message_field(message: dict, *candidates: str) -> Any:
    for key in candidates:
        if key in message:
            return message[key]
    nested = message.get("message") if isinstance(message.get("message"), dict) else None
    if nested:
        for key in candidates:
            short = key.split(".")[-1]
            if short in nested:
                return nested[short]
    return None


def render_tool_call(call: dict) -> str:
    name = message_field(call, "tool_call.function.name")
    arguments = message_field(call, "tool_call.function.arguments")
    if name is None and isinstance(call.get("function"), dict):
        name = call["function"].get("name")
        arguments = call["function"].get("arguments")
    return f"[tool_call] {name}({arguments})"


def render_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages or "")
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        role = message_field(message, "message.role") or "unknown"
        content = message_field(message, "message.content") or ""
        tool_calls = message_field(message, "message.tool_calls") or []
        for call in tool_calls:
            if isinstance(call, dict):
                content += "\n" + render_tool_call(call)
        lines.append(f"{role}: {content}".strip())
    return "\n".join(lines)


def available_tools_description() -> str:
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in TRAVEL_TOOLS)


def build_llm_rows(llm_spans: pd.DataFrame) -> pd.DataFrame:
    rows = llm_spans.copy()
    rows["context.span_id"] = rows.index
    rows["input"] = rows.get("attributes.llm.input_messages", pd.Series(index=rows.index, dtype=object)).apply(render_messages)
    rows["available_tools"] = available_tools_description()
    rows["tool_selection"] = rows.get("attributes.llm.output_messages", pd.Series(index=rows.index, dtype=object)).apply(render_messages)
    return rows.reset_index(drop=True)


@create_evaluator(name="tool_output_structured", kind="code", direction="maximize")
def tool_output_structured(output: str) -> dict[str, Any]:
    try:
        payload = json.loads(output or "")
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "label": "invalid_json", "explanation": "Tool output is not valid JSON"}
    if isinstance(payload, dict) and "error" in payload:
        return {"score": 0.0, "label": "tool_error", "explanation": str(payload["error"])}
    return {"score": 1.0, "label": "structured_ok", "explanation": "Valid JSON payload without error key"}


def summarize(name: str, results: pd.DataFrame) -> None:
    labels = results[f"{name}_score"].apply(lambda score: score.get("label") if isinstance(score, dict) else "failed")
    print(f"\n{name}: {len(results)} rows")
    print(labels.value_counts().to_string())


def log_annotations(client: Client, results: pd.DataFrame, name: str, annotator_kind: str) -> None:
    annotations = to_annotation_dataframe(dataframe=results, score_names=[name])
    client.spans.log_span_annotations_dataframe(dataframe=annotations, annotator_kind=annotator_kind, sync=True)
    print(f"Logged {len(annotations)} '{name}' annotations to Phoenix")


def annotate_sessions(client: Client, turns: pd.DataFrame) -> None:
    turns["friction_score"] = turns["user_friction_score"].apply(lambda score: score.get("score", 0.0) if isinstance(score, dict) else 0.0)
    per_session = turns.groupby("session_id")["friction_score"].agg(["max", "sum", "count"])
    for session_id, row in per_session.iterrows():
        frustrated = row["max"] >= 1.0
        client.sessions.add_session_annotation(
            session_id=session_id,
            annotation_name="session_frustration",
            annotator_kind="LLM",
            label="frustrated" if frustrated else "not_frustrated",
            score=float(row["sum"] / row["count"]),
            explanation=f"{int(row['sum'])} of {int(row['count'])} user turns expressed friction",
            sync=True,
        )
    print(f"Logged session_frustration annotations on {len(per_session)} sessions")


def main() -> None:
    client = Client(base_url=collector_endpoint())
    llm = judge_llm()
    EXPORT_DIR.mkdir(exist_ok=True)

    root_spans = fetch_spans(client, f"parent_id is None and name == '{ROOT_SPAN_NAME}'")
    if root_spans.empty:
        sys.exit("No root spans found. Run `make queries` first.")
    turns = build_turns(root_spans)
    print(f"Evaluating user frustration on {len(turns)} conversation turns across {turns['session_id'].nunique()} sessions")
    friction_results = evaluate_dataframe(
        dataframe=turns[["context.span_id", "session_id", "turn_index", "conversation", "user_message"]],
        evaluators=[UserFrictionEvaluator(llm=llm, temperature=0.0)],
        exit_on_error=False,
    )
    summarize("user_friction", friction_results)
    log_annotations(client, friction_results, "user_friction", "LLM")
    annotate_sessions(client, friction_results)
    friction_results.to_json(EXPORT_DIR / "eval_user_friction.json", orient="records", indent=2, default_handler=str)

    llm_spans = fetch_spans(client, "span_kind == 'LLM'")
    if not llm_spans.empty:
        llm_rows = build_llm_rows(llm_spans)
        print(f"\nEvaluating tool selection on {len(llm_rows)} LLM spans")
        tool_results = evaluate_dataframe(
            dataframe=llm_rows[["context.span_id", "input", "available_tools", "tool_selection"]],
            evaluators=[ToolSelectionEvaluator(llm=llm, temperature=0.0)],
            exit_on_error=False,
        )
        summarize("tool_selection", tool_results)
        log_annotations(client, tool_results, "tool_selection", "LLM")
        tool_results.to_json(EXPORT_DIR / "eval_tool_selection.json", orient="records", indent=2, default_handler=str)

    tool_spans = fetch_spans(client, "span_kind == 'TOOL'")
    turns = attach_tool_results(turns, tool_spans)
    print(f"\nEvaluating hallucination on {len(turns)} answers against their tool results")
    hallucination_results = evaluate_dataframe(
        dataframe=turns[["context.span_id", "user_message", "tool_results", "assistant_response"]],
        evaluators=[hallucination_judge(llm)],
        exit_on_error=False,
    )
    summarize("hallucination", hallucination_results)
    log_annotations(client, hallucination_results, "hallucination", "LLM")
    hallucination_results.to_json(EXPORT_DIR / "eval_hallucination.json", orient="records", indent=2, default_handler=str)

    if not tool_spans.empty:
        tool_rows = tool_spans.copy()
        tool_rows["context.span_id"] = tool_rows.index
        tool_rows["output"] = tool_rows["attributes.output.value"].fillna("")
        print(f"\nChecking structured output on {len(tool_rows)} tool spans")
        structured_results = evaluate_dataframe(
            dataframe=tool_rows[["context.span_id", "name", "output"]].reset_index(drop=True),
            evaluators=[tool_output_structured],
            exit_on_error=False,
        )
        summarize("tool_output_structured", structured_results)
        log_annotations(client, structured_results, "tool_output_structured", "CODE")

    print("\nAll evaluations attached to spans. Open Phoenix and filter with: annotations['user_friction'].label == 'friction'")

if __name__ == "__main__":
    main()
