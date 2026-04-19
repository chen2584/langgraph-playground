from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    thinking_budget=1024,   # Optional: Set a token limit for the agent's internal reasoning (thoughts)
    include_thoughts=True
)

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

agent = create_agent(tools=tools, model=llm)

inputs = {"messages": [{"role": "user", "content": "When was SpaceX's last launch and how many days ago?"}]}

# This is for debug, to see the internal reasoning and tool calls of the agent in real-time as it processes the input.
for chunk in agent.stream(inputs, stream_mode="values"):
    message = chunk["messages"][-1]
    
    # 1. Extract Internal Reasoning (The 'Thought' part)
    # Gemini 2.5+ returns thoughts as a list of dictionaries in 'content' 
    # if it's a multi-part message, or in additional_kwargs['thought']
    
    if isinstance(message.content, list):
        for part in message.content:
            # Look specifically for the 'thinking' type
            if part.get("type") == "thinking":
                print(f"\n[INTERNAL REASONING]:\n{part.get('thinking')}")
            
            # Look for the 'text' type (the actual answer)
            elif part.get("type") == "text":
                print(f"\n[AGENT RESPONSE]:\n{part.get('text')}")

    # 2. Identify Tool Actions
    if hasattr(message, "tool_calls") and message.tool_calls:
        print(f"\n[ACTION]: Calling {message.tool_calls[0]['name']}...")
        print(f"[PARAMETERS]: {message.tool_calls[0]['args']}")
        
    # 3. Identify Final Answer
    elif message.type == "ai" and not message.tool_calls:
        # Final response text might also be in a list of parts
        final_text = message.content
        if isinstance(final_text, list):
            final_text = next((p["text"] for p in final_text if p.get("type") == "text"), "")
            
        print(f"\n[FINAL RESPONSE]:\n{final_text}")

# use this to get result, which is the final response after the agent finishes all its reasoning and tool calls.
# response = agent.invoke({
#     "messages": [{"role": "user", "content": "When was SpaceX's last launch and how many days ago?"}]
# })