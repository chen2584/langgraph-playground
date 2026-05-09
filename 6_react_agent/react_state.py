from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # This single list now replaces agent_outcome and intermediate_steps
    # AIMessage with tool_calls = AgentAction
    # ToolMessage = Observation/Result
    # AIMessage without tool_calls = AgentFinish
    messages: Annotated[Sequence[BaseMessage], add_messages]