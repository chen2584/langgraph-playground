from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph
from nodes import reason_node, act_node
from react_state import AgentState

# Node Names
REASON = "reason"
ACT = "act"

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM didn't call a tool, we finish
    if not last_message.tool_calls:
        return END
    # Otherwise, we go to the tool execution node
    return ACT

workflow = StateGraph(AgentState)

workflow.add_node(REASON, reason_node)
workflow.add_node(ACT, act_node)

workflow.set_entry_point(REASON)

workflow.add_conditional_edges(
    REASON,
    should_continue,
    {ACT: ACT, END: END} # Explicit mapping is now preferred
)

workflow.add_edge(ACT, REASON)

app = workflow.compile()

# Execution
inputs = {"messages": [("user", "How many days ago was the latest SpaceX launch?")]}
result = app.invoke(inputs)

# Accessing the final answer
print(result["messages"][-1].content, "final result")