import os
from datetime import date
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from tools import TRAVEL_TOOLS
from tools.common import to_json

load_dotenv()

DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a travel assistant. You help travelers plan trips: flights, hotels, weather, attractions, budgets and practical questions.

Today's date is {today}.

Rules:
- Use the tools whenever the answer depends on data (prices, availability, weather, exchange rates, places). Never invent flight, hotel or weather data.
- Resolve relative dates (next Friday, in two weeks) into YYYY-MM-DD before calling tools.
- If a tool returns an error, explain the problem plainly and ask for what is missing instead of retrying blindly.
- Ask a clarifying question when a required detail (dates, origin city, budget) is missing, unless a sensible default is obvious.
- Keep answers concise and structured: short paragraphs or bullet lists, prices with currency, dates written out.
- When you present options, name your recommendation and say why in one sentence."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_model() -> BaseChatModel:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), temperature=0)


def build_agent(model: BaseChatModel | None = None, checkpointer=None):
    chat_model = model or build_model()
    model_with_tools = chat_model.bind_tools(TRAVEL_TOOLS)
    tools_by_name = {tool.name: tool for tool in TRAVEL_TOOLS}

    def llm_call(state: AgentState) -> dict:
        system_message = SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat()))
        response = model_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        results = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name.get(tool_call["name"])
            if tool is None:
                observation = {"error": f"Unknown tool {tool_call['name']}", "tool": tool_call["name"]}
            else:
                observation = tool.invoke(tool_call["args"])
            results.append(ToolMessage(content=to_json(observation), tool_call_id=tool_call["id"], name=tool_call["name"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")
    return graph_builder.compile(checkpointer=checkpointer or MemorySaver())
