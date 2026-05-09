from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from react_state import AgentState
# Note: Ensure your tools are imported as a list of BaseTool/StructuredTool
from agent_reason_runnable import tools 

load_dotenv()

# Initialize the LLM and bind tools once
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

def reason_node(state: AgentState):
    """
    Decides whether to continue or finish based on the message history.
    """
    # Simply pass the message list to the model
    response = llm_with_tools.invoke(state["messages"])
    
    # We return the AI Message; 'add_messages' in AgentState handles the append
    return {"messages": [response]}


def act_node(state: AgentState):
    """
    Executes the tools requested by the LLM.
    """
    last_message = state["messages"][-1]
    tool_results = []
    
    # Map tool names to actual tool objects for quick lookup
    tool_map = {tool.name: tool for tool in tools}
    
    # Handle all tool calls requested in the last AI message
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tool_map.get(tool_name)
        
        if selected_tool:
            # Execute tool and wrap result in a ToolMessage
            observation = selected_tool.invoke(tool_call["args"])
            tool_results.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                )
            )
        else:
            tool_results.append(
                ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found.",
                    tool_call_id=tool_call["id"]
                )
            )
    
    return {"messages": tool_results}