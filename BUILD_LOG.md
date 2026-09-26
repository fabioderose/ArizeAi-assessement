# Build log

What I did, in order, with the reason and the way it was done. Dates are September 2026.

## 1. Read the starter

Cloned `trevorlaviale/se-interview`. Two-node LangGraph agent, one DuckDuckGo tool returning a text blob, FastAPI `POST /chat` with no session, generic prompt, no tests, no tracing. Python 3.13 on my machine, no Poetry, no Phoenix CLI.

Three things had to change before any feature: the tool output had to become structured, the agent needed sessions (a frustration evaluation only makes sense across turns), and the dependency pins had to move.

## 2. Tooling

Installed Poetry (Homebrew) because the starter README uses it, and the Phoenix CLI (`npm install -g @arizeai/phoenix-cli`) plus the Phoenix skills documentation, as recommended by the assessment. The skills document the current API of `phoenix.otel`, `phoenix.client` and `phoenix.evals` (evaluators 2.0, `evaluate_dataframe`, `to_annotation_dataframe`, filter expressions), which matters for a library that moves this fast.

## 3. Dependencies

`poetry lock` failed with the original pins: `arize-phoenix` 20 requires `fastapi >= 0.137`, the starter pinned `^0.115`. Moved to LangGraph 1.x, LangChain 1.x, langchain-openai 1.x, added the Phoenix packages, `httpx`, `pandas`, `pytest`. The original graph code only uses stable APIs and did not need to change. Virtualenv in the project (`poetry.toml`).

## 4. Tools

Replaced the single tool with a `tools/` package of six: `search_flights` and `search_hotels` (deterministic simulated inventories seeded with SHA-256 of route and date, because I found no keyless API for flight or hotel inventory), `get_weather_forecast` (Open-Meteo), `find_attractions` (Open-Meteo geocoding plus Wikipedia geosearch), `convert_currency` (Frankfurter, ECB rates), `web_search` (DuckDuckGo, now a structured list).

Every tool has a Pydantic `args_schema` and returns a Pydantic model dump. None raises: validation errors and provider failures come back as `{"error", "tool", "details"}` so the graph keeps running and the model can explain or ask. Shared helpers in `tools/common.py`.

## 5. Agent and API

`agent.py`: same two-node graph, `add_messages` reducer, travel-assistant prompt with today's date injected (the tools only accept ISO dates), `MemorySaver` checkpointer keyed by `thread_id`, tolerant `tool_node` (unknown tool name returns an error payload), `build_agent(model=None)` so tests can inject a fake model.

`api.py`: `/chat` accepts an optional `session_id` and returns `response`, `session_id`, `trace_id` and the tool calls of the turn. Added `/tools`, `/health`, and a small chat UI on `/`.

## 6. Phoenix instrumentation

`tracing.py` calls `register(project_name="travel-assistant", batch=True)` then `LangChainInstrumentor().instrument()`. `api.py` opens one manual root span per request (`travel_assistant`, OpenInference kind `AGENT`, `input.value` = user message, `output.value` = answer) inside `using_session(session_id)`, and passes the same id as LangGraph `thread_id`.

Two bugs found here. Passing `endpoint=` to `register()` drops the `/v1/traces` suffix and Phoenix answers 405; the endpoint now goes through `PHOENIX_COLLECTOR_ENDPOINT`. And `auto_instrument=True` also activated the OpenAI instrumentor pulled in as a transitive dependency, so every model call was traced twice (80 LLM spans for 40 calls, tokens and cost doubled). Found in the Spans tab, fixed by instrumenting LangChain explicitly. The traces from before the fix are kept in the reference export.

Phoenix runs with `poetry run phoenix serve` (data in `.phoenix/`). Docker Compose is provided as an alternative.

## 7. Tests

`tests/test_tools.py` (tool contracts, determinism, sorting, filters, structured errors, two live calls marked `network`), `tests/test_agent.py` (graph with a scripted fake model: tool call then answer, memory per thread, unknown tool), `tests/test_eval_helpers.py` (history reconstruction, message rendering, the code evaluator). 17 tests, none needs an API key. `pytest.ini` sets `pythonpath = .` because the root `__init__.py` inherited from the starter made pytest resolve imports from the parent directory on a fresh clone.

## 8. Queries and export

`data/queries.json`: 12 sessions, 24 turns, every tool covered, one no-tool question, an impossible request, an ambiguous one, three deliberately frustrated conversations. `scripts/run_queries.py` replays them with one `session_id` per session. `scripts/export_spans.py` writes `exports/spans.csv` and `exports/spans.jsonl` (`pd.isna` crashes on list attributes, hence the tolerant check).

## 9. Evaluations

`scripts/run_evals.py`:

- `user_friction`: Phoenix `UserFrictionEvaluator`, gpt-4o-mini, one row per user turn with the previous turns of the session rebuilt from the root spans. Logged on the root span, plus a `session_frustration` annotation per session.
- `tool_selection`: Phoenix `ToolSelectionEvaluator` on every LLM span, with the tool list rendered from `TRAVEL_TOOLS`.
- `hallucination`: custom `ClassificationEvaluator`, the answer judged against the tool outputs of the same trace.
- `tool_output_structured`: code evaluator, valid JSON without an `error` key.

Phoenix stores LLM messages flattened (`message.role`, `message.content`), the rendering reads both shapes. Results are written with `log_span_annotations_dataframe(sync=True)` so the dataset script can filter right after.

`scripts/build_dataset.py` turns any annotation filter into a dataset linked to the source spans, with presets `frustrated-interactions`, `hallucinated-answers`, `tool-errors`.

## 10. Reference run (23 September, Phoenix 20.16.0)

24/24 queries, 245 spans. `user_friction` 5/24, exactly the three scripted sessions, no false positive. `tool_selection` 76/80, the four misses are two judge errors counted twice. `tool_output_structured` 19/21 (weather horizon, past check-in date). Dataset `frustrated-interactions` with 5 examples. `hallucination` added later: 8 flags out of 48 turns, 5 of them false positives on review because the context was the current trace only.

## 11. What the traces revealed and what I left as is

- Open-Meteo forecasts 16 days ahead. A trip in three weeks gets an error payload. On one run the model dropped the weather silently, on the next it said the forecast was not available. Same input, two behaviours.
- One transient SSL timeout on Open-Meteo became a two-turn frustrated session. No retry in the HTTP helper.
- The `tool_selection` judge penalises a final answer because an earlier tool call failed, and once contradicted its own explanation.
- Tool spans keep status OK when the tool returns an error payload, by contract; the annotation carries the failure.

These stay in the traces on purpose: they are the material for the debugging part of the presentation. The fixes (prompt rule, retries, judge validation against human labels, session-wide context for the hallucination judge) are the next steps.

## 12. Submission pass

Upgraded to Phoenix 20.16.0 and phoenix-evals 3.9.0. Verified from a clean copy: `poetry install`, 17 tests, `import api`, one request through the API and its trace in Phoenix. README has a results table, known limitations and a checklist against the assessment.
