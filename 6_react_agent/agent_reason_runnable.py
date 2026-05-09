from langchain_google_genai import ChatGoogleGenerativeAI
import datetime
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_agent
from langchain.tools import tool

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

search_tool = TavilySearchResults(search_depth="basic")

system_prompt = """You are a helpful assistant. 
Use the provided tools to answer questions. 
If you don't know the answer, say you don't know."""

tools = [get_system_time, search_tool]

react_agent_runnable = create_agent(tools=tools, model=llm, system_prompt=system_prompt)