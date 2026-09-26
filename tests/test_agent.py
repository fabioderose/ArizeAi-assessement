# Test if the graph works as expected
import json
from datetime import date, timedelta

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import build_agent


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted_agent(responses):
    return build_agent(model=ScriptedModel(messages=iter(responses)))


def test_agent_calls_tool_then_answers():
    departure = (date.today() + timedelta(days=20)).isoformat()
    agent = scripted_agent(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_flights", "args": {"origin": "Paris", "destination": "Rome", "departure_date": departure}, "id": "call_1"}],
            ),
            AIMessage(content="Here are your flights."),
        ]
    )
    result = agent.invoke({"messages": [HumanMessage(content="Flights Paris to Rome")]}, config={"configurable": {"thread_id": "t1"}})
    messages = result["messages"]
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0].content)
    assert payload["origin"] == "CDG"
    assert messages[-1].content == "Here are your flights."


def test_agent_keeps_conversation_state_per_thread():
    agent = scripted_agent([AIMessage(content="Hello"), AIMessage(content="Second turn")])
    config = {"configurable": {"thread_id": "t2"}}
    agent.invoke({"messages": [HumanMessage(content="hi")]}, config=config)
    result = agent.invoke({"messages": [HumanMessage(content="again")]}, config=config)
    assert len(result["messages"]) == 4


def test_agent_handles_unknown_tool_gracefully():
    agent = scripted_agent(
        [
            AIMessage(content="", tool_calls=[{"name": "teleport", "args": {}, "id": "call_x"}]),
            AIMessage(content="Sorry, I cannot do that."),
        ]
    )
    result = agent.invoke({"messages": [HumanMessage(content="teleport me")]}, config={"configurable": {"thread_id": "t3"}})
    tool_message = [message for message in result["messages"] if isinstance(message, ToolMessage)][0]
    assert "Unknown tool" in tool_message.content
