from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState, MessageGraph
from chains import generation_chain, reflection_chain

REFLECT = "reflect"
GENERATE = "generate"
graph = StateGraph(MessagesState)


def generate_node(state: MessagesState):
    return {"messages": generation_chain.invoke({
        "messages": state["messages"]
    })}


def reflect_node(state: MessagesState):
    response = reflection_chain.invoke({
        "messages": state["messages"]
    })
    return {"messages": [HumanMessage(content=response.content)]}


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)


def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# For debugging, you can visualize the graph structure
# print(app.get_graph().draw_mermaid())
# app.get_graph().print_ascii()

response = app.invoke({"messages": [HumanMessage(content="AI Agents taking over content creation")]})

print(response)
