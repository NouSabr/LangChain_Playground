import os
from dotenv import load_dotenv

from typing_extensions import List

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_tavily import TavilySearch
from tavily import TavilyClient

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,  # type: ignore
)

tavily = TavilyClient()

class Source(BaseModel):
    """
    Schema for a source used by the Agent

    """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for agent response with answer and sources

    """

    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer.")


@tool
def search_v1(query: str = "Hello world!") -> str:
    """
    Tool that searches over the internet

    Args:
        query: the query to search for

    Returns:
        The search result
    
    """
    tool = TavilySearch(
        max_results=5,
        topic="general",
        # include_answer=False,
        # include_raw_content=False,
        # include_images=False,
        # include_image_descriptions=False,
        # search_depth="basic",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )

    res = tool.invoke({"query": query})
    return res

@tool
def search_v2(query: str = "Hello World!") -> str:
    """
    Tool that searches over the internet

    Args:
        query: the query to search for

    Returns:
        The search result
    
    """
    return tavily.search(query = query) #type: ignore

tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools, response_format=AgentResponse)

def main():
    res = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an AI engineer using LangChain on linkedin and list their details.")}) # type: ignore
    print(res)



if __name__ == "__main__":
    main()
