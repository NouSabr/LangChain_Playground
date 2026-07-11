from dotenv import load_dotenv
import os 
import json

from openai import AzureOpenAI

load_dotenv()

from langsmith import traceable

MAX_ITERATIONS=10
MODEL="gpt-4o-2"

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ----- Tools ------

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog

    """
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold

    """
    print(f" >> Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze":5, "silver": 12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    
    return round(price * (1-discount/100), 2)

tools_for_llm =  [
        {
            "type":"function",
            "function":{
                "name":"get_product_price",
                "description":"Look up the price of a product in the catalog",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "product":{"type":"string", "description":"The name of the product"}
                    },
                "required": ["product"]
                }
            }
        },
                {
            "type":"function",
            "function":{
                "name":"apply_discount",
                "description":" Apply a discount tier to a price and return the final price.Available tiers: bronze, silver, gold",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "price":{"type":"number", "description": "The price of the product"},
                        "discount_tier":{"type":"string", "description": "the discount tier. Either bronze, silver or gold"}
                    },
                "required": ["price","discount_tier"]
                }
            }
        }
    ]

# ----- Helper: Traced AzureOpenAI call -----


llm = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT # type: ignore
)

@traceable(name="Azure OpenAI Chat", run_type="llm")
def azure_openai_chat_traced(messages:list):
    return llm.chat.completions.create(
            model=DEPLOYMENT, #type: ignore
            messages= messages,
            tools = tools_for_llm if tools_for_llm else None #type: ignore
        )

# ----- Agent Loop -----

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    tools =  [get_product_price, apply_discount]
    tools_dict = {t.__name__: t for t in tools}

    print(f"Question: {question}")
    print("*" * 60)

    messages = [
        {
            "role":"system",
            "content":"""You are a helpful shopping assistant.
            "You have access to a product catalog tool and a discount tool.\n\n
            STRICT RULES: You must follow these exactly:\n
            1. NEVER guess or assume any product price
            2. You MUST call get_product_price first to get the real price
            3. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price. Do NOT pass a made-up number.
            4. Always use the apply_discount tool\n
                If the user does not specify a discount tier, ask them which tier to use. DO NOT ASSUME
            5. The product names are generic. Example: 'laptop', 'headphones', etc. There is no need to ask for more details regarding exact models/names"""
        }
        ,
        {
            "role":"user",
            "content":question
        }
    ]

    for iteration in range (1, MAX_ITERATIONS + 1):
        print(f"\n----- Iteration {iteration} -----")

        ai_message = azure_openai_chat_traced(messages).choices[0]

        tool_calls = ai_message.message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.message.content}")
            return ai_message.message.content
        
        # Process only the FIRST tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name # type: ignore
        tool_args = json.loads(tool_call.function.arguments) # type: ignore
        tool_call_id = tool_call.id

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use(**tool_args) # type: ignore

        print(f"[Tool Result] {observation}")

        messages.append(ai_message.message) # type:ignore
        messages.append(
            {
                "role":"tool",
                "content": str(observation),
                "tool_call_id": tool_call_id
            }
        )

    print("ERROR: Max iterations reached without a final answer")
    return None















if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = (run_agent("What is the price of a laptop after applying a gold discount?"))
