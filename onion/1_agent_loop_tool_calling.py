from dotenv import load_dotenv
import os 

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI

from langchain.tools import tool
from langsmith import traceable

from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

MAX_ITERATIONS=10
MODEL="gpt-4o-2"

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ----- Tools (LangChain @tool decorator) ------

@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog

    """
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    return prices.get(product, 0)

@tool
def appy_discount(price: float, dicsount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold

    """
    print(f" >> Executing apply_discount(price={price}, discount_tier={dicsount_tier})")
    discount_percentages = {"bronze":5, "silver": 12, "gold":23}
    discount = discount_percentages.get(dicsount_tier,0)
    
    return round(price * (1-discount/100), 2)

# ----- Agent Loop -----

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools =  [get_product_price, appy_discount]

    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"azure_openai:{MODEL}",
        temperature=0, azure_deployment=DEPLOYMENT,
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY
        )


    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("*" * 60)

    messages = [
        SystemMessage(content="""You are a helpful shopping assistant.
        "You have access to a product catalog tool and a discount tool.\n\n
        STRICT RULES: You must follow these exactly:\n
        1. NEVER guess or assume any product price
        2. You MUST call get_product_price first to get the real price
        3. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price. Do NOT pass a made-up number.
        4. Always use the apply_discount tool\n
            If the user does not specify a discount tier, ask them which tier to use. DO NOT ASSUME"""
        ),
        HumanMessage(content=question)
    ]

    for iteration in range (1, MAX_ITERATIONS + 1):
        print(f"\n----- Iteration {iteration} -----")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


    













if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = (run_agent("What is the price of a laptop after applying a gold discount?"))
