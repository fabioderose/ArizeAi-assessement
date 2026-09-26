import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from openinference.instrumentation import using_session
from opentelemetry.trace import format_trace_id
from pydantic import BaseModel, Field

load_dotenv()

from tracing import collector_endpoint, project_name, setup_tracing

tracer_provider = setup_tracing()
tracer = tracer_provider.get_tracer(__name__)

from agent import build_agent
from tools import TRAVEL_TOOLS

agent = build_agent()

app = FastAPI(
    title="Travel Assistant Agent API",
    description="LangGraph travel assistant instrumented with Arize Phoenix",
    version="0.2.0",
)

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = Field(default=None, description="Reuse the same id to continue a conversation")


class ToolCallSummary(BaseModel):
    name: str
    args: dict


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_id: str
    tool_calls: list[ToolCallSummary]


def new_messages_since_last_human(messages: list) -> list:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index + 1 :]
    return messages


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}, "metadata": {"session_id": session_id}}
    try:
        with using_session(session_id):
            with tracer.start_as_current_span("travel_assistant", openinference_span_kind="agent") as span:
                span.set_input(request.message)
                result = agent.invoke({"messages": [HumanMessage(content=request.message)]}, config=config)
                answer = result["messages"][-1].content
                span.set_output(answer)
                trace_id = format_trace_id(span.get_span_context().trace_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent failed: {type(exc).__name__}: {exc}") from exc
    tool_calls = [
        ToolCallSummary(name=call["name"], args=call["args"])
        for message in new_messages_since_last_human(result["messages"])
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    return ChatResponse(response=answer, session_id=session_id, trace_id=trace_id, tool_calls=tool_calls)


@app.get("/tools")
def list_tools():
    return [{"name": tool.name, "description": tool.description, "args": tool.args} for tool in TRAVEL_TOOLS]


@app.get("/health")
def health():
    return {"status": "ok", "phoenix_endpoint": collector_endpoint(), "phoenix_project": project_name()}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
