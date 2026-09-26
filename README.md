# Travel Assistant Agent, observed and evaluated with Arize Phoenix

A LangGraph travel assistant exposed through FastAPI, instrumented with OpenInference / OpenTelemetry into a local Arize Phoenix instance, and evaluated with Phoenix evals (user frustration, tool selection, structured tool output).

This repository is the submission for the Arize "Agent Framework Evaluation using Phoenix" technical assessment. It extends [trevorlaviale/se-interview](https://github.com/trevorlaviale/se-interview).

```
user ──HTTP──▶ FastAPI /chat ──▶ LangGraph agent ──▶ tools (flights, hotels, weather, attractions, currency, web)
                    │                  │
                    └──── OpenInference spans (agent, chain, llm, tool) ────▶ Phoenix (localhost:6006)
                                                                                 │
                                                        evals ◀── export spans ◀─┘
                                                          └──▶ annotations on spans + sessions
                                                          └──▶ dataset "frustrated-interactions"
```

## Contents

- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Agent architecture](#agent-architecture)
- [Tools](#tools)
- [Observability with Phoenix](#observability-with-phoenix)
- [Tracing experiments](#tracing-experiments)
- [Evaluation](#evaluation)
- [Design decisions](#design-decisions)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Quick start

Prerequisites: Python 3.11 to 3.13, [Poetry](https://python-poetry.org/docs/#installation), an OpenAI API key. Docker is optional.

```bash
git clone <this repository>
cd se-interview
cp .env.example .env
```

Put your key in `.env`:

```
OPENAI_API_KEY=sk-...
```

Then, in three terminals:

```bash
make install        # poetry install
make phoenix        # poetry run phoenix serve      -> http://localhost:6006 (data in ./.phoenix)
make api            # uvicorn api:app --reload      -> http://localhost:8000
```

Run the full experiment pipeline from a fourth terminal:

```bash
make demo           # queries -> export -> evals -> dataset
```

Or step by step:

```bash
make queries        # sends the 24 scripted turns (12 sessions) through the API
make export         # exports every span of the project to exports/spans.csv and exports/spans.jsonl
make evals          # runs the Phoenix evaluators and attaches results to spans and sessions
make dataset        # filters spans labeled "friction" in Phoenix and creates the dataset
make datasets       # same, plus hallucinated-answers and tool-errors
make test           # pytest (tools + graph, no LLM key needed)
```

Phoenix can also run in Docker instead of the Poetry environment: `make phoenix-docker` (image `arizephoenix/phoenix`, persistent volume).

Everything in Docker (Phoenix + API):

```bash
docker compose --profile full up --build
```

## Using the app

- Chat UI: open http://localhost:8000. Each answer shows the tools that were called and the Phoenix trace id. "New conversation" starts a new session; otherwise every message continues the same session so you can test multi-turn behavior.
- API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find me a flight from Paris to Lisbon in 3 weeks"}'
```

```json
{
  "response": "Here are the cheapest options ...",
  "session_id": "4c3d...",
  "trace_id": "8f0a...",
  "tool_calls": [{"name": "search_flights", "args": {"origin": "Paris", "destination": "Lisbon", "departure_date": "2026-10-12"}}]
}
```

Pass `session_id` back in the next request to continue the conversation. `GET /tools` lists the tools and their JSON schemas, `GET /health` reports the Phoenix endpoint in use.

## Agent architecture

The agent is a LangGraph `StateGraph` with two nodes and one conditional edge, the classic ReAct loop:

```
START ─▶ llm_call ─▶ should_continue ─┬─ tool_calls present ─▶ tool_node ─▶ llm_call
                                      └─ no tool calls ─────▶ END
```

- `llm_call` sends the system prompt (travel assistant persona, today's date, tool-usage rules) plus the conversation to `gpt-4o` with the six tools bound.
- `tool_node` executes every tool call from the last AI message and appends one `ToolMessage` per call, serialized as JSON.
- `should_continue` routes back to the tools while the model keeps requesting them.
- State is a list of messages with the `add_messages` reducer. A `MemorySaver` checkpointer keyed by `thread_id` keeps the conversation across HTTP calls, which is what makes the `session_id` in the API work.

The model, the graph and the tools are decoupled: `build_agent(model=...)` accepts any LangChain chat model, which is how the tests drive the graph with a scripted fake model and no API key.

## Tools

| Tool | Data source | Output model |
| --- | --- | --- |
| `search_flights` | Deterministic simulated inventory (seeded by route and date) | `FlightSearchResult` with sorted `FlightOption[]` |
| `search_hotels` | Deterministic simulated inventory (seeded by city and check-in) | `HotelSearchResult` with `HotelOption[]` |
| `get_weather_forecast` | [Open-Meteo](https://open-meteo.com) geocoding + forecast API, no key | `WeatherForecast` with `DailyForecast[]` |
| `find_attractions` | Open-Meteo geocoding + Wikipedia geosearch, no key | `AttractionsResult` with `Attraction[]` |
| `convert_currency` | [Frankfurter](https://frankfurter.dev) ECB rates, no key | `CurrencyConversion` |
| `web_search` | DuckDuckGo (from the original repository, now returning structured results) | `WebSearchResult` |

Every tool:

- declares its arguments with a Pydantic `args_schema`, so the LLM receives a strict JSON schema with field descriptions and bounds;
- returns a Pydantic model dumped to a dict, serialized to JSON in the `ToolMessage`;
- never raises: validation problems and provider outages come back as a structured `{"error", "tool", "details"}` payload so the agent can explain the problem and ask for what is missing instead of crashing the request.

## Observability with Phoenix

### OpenTelemetry and OpenInference

OpenTelemetry (OTel) is the vendor-neutral standard for traces: an SDK creates spans, a processor batches them, an exporter ships them over OTLP to a collector. OpenInference is Arize's set of semantic conventions on top of OTel for LLM applications: it defines span kinds (`AGENT`, `CHAIN`, `LLM`, `TOOL`, `RETRIEVER`, ...) and attribute names (`input.value`, `llm.token_count.total`, `tool.parameters`, `session.id`, ...) so that Phoenix can render prompts, tool calls and token usage instead of opaque blobs.

### Traces and spans

A trace is one end-to-end request, a tree of spans sharing a `trace_id`. A span is one timed operation with a kind, attributes, status and a parent. In this app one `/chat` call produces one trace:

```
travel_assistant (AGENT, root)              input = user message, output = final answer, session.id
└── LangGraph (CHAIN)
    ├── llm_call (CHAIN)
    │   └── ChatOpenAI (LLM)                model, messages, tools, tool_calls, token counts
    ├── tool_node (CHAIN)
    │   └── search_flights (TOOL)           tool.name, tool.parameters, JSON output
    └── llm_call (CHAIN)
        └── ChatOpenAI (LLM)
```

### How the instrumentation works

`tracing.py` calls `phoenix.otel.register(project_name="travel-assistant", auto_instrument=False, batch=True)` before the agent is built, then `LangChainInstrumentor().instrument(tracer_provider=...)`. `register` configures the OTel tracer provider with an OTLP HTTP exporter pointed at `PHOENIX_COLLECTOR_ENDPOINT`. The LangChain instrumentor hooks LangChain callbacks and emits `CHAIN`, `LLM` and `TOOL` spans for every LangGraph node, model call and tool invocation. It is enabled explicitly rather than through `auto_instrument=True`, which would also activate the OpenAI instrumentor installed as a transitive dependency and trace every model call twice.

On top of the automatic spans, `api.py` opens one manual root span per request with `tracer.start_as_current_span("travel_assistant", openinference_span_kind="agent")`, sets `input.value` to the raw user message and `output.value` to the final answer, and wraps the call in `using_session(session_id)` so that every span in the tree carries `session.id`. Phoenix groups traces by that attribute in its Sessions view, which is what makes multi-turn evaluation possible.

The Phoenix CLI (`px`) and the Phoenix skills documentation were used during development to inspect traces from the terminal:

```bash
export PHOENIX_ENDPOINT=http://localhost:6006 PHOENIX_PROJECT=travel-assistant
px project list
px trace list --limit 10
px span list --span-kind TOOL --limit 20
px session list --include-annotations
```

## Tracing experiments

`data/queries.json` holds 12 scripted sessions (24 user turns). They cover every tool, one no-tool question, an impossible request (Martian credits) to exercise the error path, an ambiguous request, and three deliberately frustrated conversations where the user corrects the assistant or complains. `scripts/run_queries.py` replays them through the API, reusing one `session_id` per session, and saves the responses to `exports/query_results.json`.

`scripts/export_spans.py` downloads every span of the project with the Phoenix client (`client.spans.get_spans_dataframe`) and writes `exports/spans.csv` and `exports/spans.jsonl`, printing a breakdown by span kind and tool.

## Evaluation

`scripts/run_evals.py` runs three evaluators and attaches every result to the relevant span as an annotation, so they show up in the Phoenix UI, can be filtered, and can be turned into datasets.

### 1. User frustration (required)

Method: LLM-as-a-judge with Phoenix's built-in `UserFrictionEvaluator` (the current name of the user frustration evaluator in `phoenix.evals.metrics`). The judge is `gpt-4o-mini` at temperature 0.

Unit of evaluation: one user turn. Frustration is a reaction to what the assistant did before, so each row contains the `conversation` (all previous user and assistant turns of the same session, reconstructed from the root spans ordered by start time) and the `user_message` of the current turn. The first turn of a session has no history and is expected to be labeled `no_friction`.

Output: label `friction` / `no_friction`, score 1 / 0 (direction minimize) and an explanation from the judge. Results are logged with `client.spans.log_span_annotations_dataframe` on the root span of the turn under the name `user_friction`. In addition, one `session_frustration` annotation per session (`frustrated` if any turn was labeled friction, score = share of frustrated turns) is written with `client.sessions.add_session_annotation`.

Dataset: `scripts/build_frustration_dataset.py` queries Phoenix with the filter

```
parent_id is None and annotations['user_friction'].label == 'friction'
```

and creates the dataset `frustrated-interactions` (input = user message, output = assistant response, metadata = session and trace ids, linked to the span). The same filter can be pasted in the Phoenix traces view to select the examples and add them to a dataset manually.

Why this design: the Phoenix rubric judges only expressed friction (corrections, retries, complaints, challenges) and favors precision over recall. Evaluating per turn with history gives a signal you can act on (which answer triggered the complaint) and aggregating per session gives the product metric (share of frustrated conversations).

### 2. Tool selection

Method: `ToolSelectionEvaluator` on every `LLM` span. The row contains the rendered input messages, the list of available tools with descriptions (rendered from `TRAVEL_TOOLS`, the same list bound to the model) and the output message with its tool calls. The judge answers whether the model picked the right tool, avoided tools when none was needed, or hallucinated one. This checks the decision-making component of the agent independently of the tools themselves.

### 3. Structured tool output

Method: a code evaluator (no LLM) on every `TOOL` span: does the output parse as JSON and is it free of an `error` key. Labels `structured_ok`, `tool_error`, `invalid_json`. This is the cheap, deterministic check that the tool contract holds, and it makes provider outages and validation failures visible as annotations.

### 4. Hallucination

Method: a `ClassificationEvaluator` with a custom rubric (`HALLUCINATION_TEMPLATE` in `scripts/run_evals.py`), judge `gpt-4o-mini`, labels `hallucination` / `no_hallucination`, direction minimize. Each root span is judged against the joined outputs of the TOOL spans of the same trace: a price, flight, hotel, temperature or rate that is not in the tool results, or that is given when no tool ran, is a hallucination. Rounding, rephrasing and recommendations are not. The judge writes an explanation on every annotation.

Known limit: the context is the current trace only. An answer that reuses a tool result from a previous turn of the same session is flagged, which produced 5 false positives out of 8 flags on review. Widening the context to the whole session is the next step.

### Datasets from annotations

`scripts/build_dataset.py` turns any annotation filter into a Phoenix dataset linked to the source spans. Presets: `frustrated-interactions` (required by the brief), `hallucinated-answers`, `tool-errors`. `make datasets` builds the three. Any other pair works with `--annotation` and `--label`.

### Results of the reference run (23 September 2026, Phoenix 20.16.0)

| Evaluation | Scope | Result |
| --- | --- | --- |
| `user_friction` | 24 root spans (one per turn) | 5 friction / 19 no_friction, all 5 in the three scripted frustrated sessions, no false positive |
| `session_frustration` | 12 sessions | 3 frustrated |
| `tool_selection` | 80 LLM spans | 76 correct / 4 incorrect. The 4 are 2 real calls counted twice because of duplicated LLM spans, and both are judge errors |
| `tool_output_structured` | 21 TOOL spans | 19 structured_ok / 2 tool_error (weather horizon, past check-in date) |
| `hallucination` (added later, on 48 turns) | 48 root spans | 40 no_hallucination / 8 hallucination, 5 of the 8 are judge false positives on review |
| Dataset `frustrated-interactions` | | 5 examples linked to their spans |

Export: 24 traces, 245 spans (120 CHAIN, 80 LLM, 24 AGENT, 21 TOOL), P50 latency 3.2 s, P99 9.4 s. `exports/` holds the most recent run on this Phoenix instance; runs accumulate in the project, so the counts there can be higher than the reference run.

### What the traces revealed

- With `auto_instrument=True`, every LLM call was traced twice (`ChatOpenAI` from the LangChain instrumentor, `ChatCompletion` from the OpenAI instrumentor pulled in as a transitive dependency): 80 LLM spans for 40 calls, tokens and cost doubled in the metrics, and the tool selection judge ran twice per call. Fixed by instrumenting LangChain explicitly. Traces from before the fix are still in the reference export and show the duplicates.
- `get_weather_forecast` returns a structured error when the trip is more than 16 days away (Open-Meteo horizon). On one run the model dropped the weather from its answer without saying so, on the next run it said the forecast was not available yet. Same input, two behaviours.
- The `tool_selection` judge penalises a final answer because an earlier tool call in the same context failed, and once labelled a correct choice `incorrect` while its own explanation said it was correct. The judge needs validation against human labels before its score is quoted.
- One transient SSL timeout on Open-Meteo (10 s, the client timeout) turned into a two-turn frustrated session on the first run and did not reproduce on the second. There is no retry in the HTTP helper.

### Known limitations

- Flights and hotels are simulated. Weather, attractions, currency and web search are live public APIs.
- `MemorySaver` is in memory: conversation history is lost when the API restarts.
- The judges are not validated against human labels, and judge and agent are from the same model family.
- TOOL spans keep status OK when the tool returns an error payload, by contract; the annotation carries the failure.
- No Phoenix experiment has been run yet; the frustrated dataset is the regression set for the next prompt change.

## Assessment checklist

| Requirement | Where |
| --- | --- |
| Step 1, run the existing application | Quick start, `BUILD_LOG.md` step 2 |
| Step 2, at least one tool, integrated, invoked when relevant, structured output, design decisions documented | `tools/`, Tools and Design decisions sections |
| Step 3, Phoenix locally, PX CLI and skills, LLM calls and tool usage captured | `tracing.py`, Observability section |
| Step 4, at least 10 queries traced, spans exported | `scripts/run_queries.py` (24 turns), `scripts/export_spans.py`, `exports/` |
| Step 5, user frustration evaluation attached to spans, filter, dataset, methodology | `scripts/run_evals.py`, `scripts/build_dataset.py`, Evaluation section |
| Dockerfile (optional) | `Dockerfile`, `docker-compose.yml` |

## Design decisions

- Deterministic simulated inventories for flights and hotels. Real providers (Amadeus, Skyscanner, Booking) need contracts and keys. A seeded generator gives realistic, structured, reproducible data, which also makes evaluation and debugging reproducible. The three other tools use free public APIs to show real integrations and real failure modes.
- Structured errors instead of exceptions. A tool exception would abort the graph and return HTTP 500. Returning an error payload keeps the loop alive, lets the model recover, and gives evaluators something to grade.
- A manual `AGENT` root span. The LangGraph auto-instrumentation root span carries the whole state dict as input. A dedicated root span with clean `input.value` / `output.value` makes evaluation dataframes trivial to build and reads better in the UI.
- Sessions via `using_session` and LangGraph `thread_id`. Multi-turn conversation is required for a meaningful frustration evaluation; both mechanisms are set from the same `session_id`.
- Evals run as offline scripts with the Phoenix client rather than inside the request path, so the app stays fast and the judge model can be swapped by changing `EVAL_MODEL`.
- Dependencies were upgraded to LangGraph 1.x / LangChain 1.x / Phoenix 20 because the original pins (LangGraph 0.2, FastAPI 0.115) conflict with the current Phoenix server package. The original graph logic is unchanged.
- Phoenix runs from the same Poetry environment by default (`phoenix serve`, zero extra install, data persisted in `.phoenix/`). Docker Compose is provided as an alternative for Phoenix alone or for Phoenix plus the API (`full` profile).

## Project structure

```
se-interview/
├── agent.py                     LangGraph agent (graph, system prompt, model factory)
├── api.py                       FastAPI app, root AGENT span, sessions, chat UI route
├── tracing.py                   Phoenix register() configuration
├── tools/
│   ├── flights.py               search_flights (simulated inventory)
│   ├── hotels.py                search_hotels (simulated inventory)
│   ├── weather.py               get_weather_forecast (Open-Meteo)
│   ├── attractions.py           find_attractions (Wikipedia geosearch)
│   ├── currency.py              convert_currency (Frankfurter / ECB)
│   ├── web_search.py            web_search (DuckDuckGo)
│   ├── geocoding.py             shared Open-Meteo geocoder
│   └── common.py                error payloads, HTTP helper, JSON helpers
├── scripts/
│   ├── run_queries.py           replay data/queries.json through the API
│   ├── export_spans.py          export spans to exports/
│   ├── run_evals.py             user friction, tool selection, hallucination, structured output evals
│   ├── build_dataset.py         Phoenix filter -> dataset (presets: frustrated-interactions, hallucinated-answers, tool-errors)
│   └── build_frustration_dataset.py  thin wrapper kept for the make target
├── data/queries.json            12 sessions / 24 turns used for the experiments
├── exports/                     query results, exported spans, eval results
├── static/index.html            minimal chat UI
├── tests/                       pytest: tools contracts + graph with a fake model
├── BUILD_LOG.md                 step-by-step log of how this was built
├── docker-compose.yml           Phoenix (+ optional app)
├── Dockerfile
└── Makefile
```

## Troubleshooting

- `API is not reachable`: start `make api` and check `http://localhost:8000/health`.
- No traces in Phoenix: check that `PHOENIX_COLLECTOR_ENDPOINT` in `.env` matches the running instance and that the API was started after Phoenix. Spans are batched and appear within a few seconds.
- `No root spans found` when running evals: run `make queries` first.
- Live tools (weather, attractions, currency, web search) need outbound internet access; they degrade to a structured error otherwise.
