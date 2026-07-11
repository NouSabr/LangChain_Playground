from dotenv import load_dotenv
import os 
import json
import inspect
import re

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
    
    return round(float(price) * (1-discount/100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}


def get_tool_descriptions(tools_dict:dict) -> str:

    descriptions = []
    
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
  STRICT RULES: You must follow these exactly:\n
            1. NEVER guess or assume any product price
            2. You MUST call get_product_price first to get the real price
            3. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price. Do NOT pass a made-up number.
            4. Always use the apply_discount tool\n
                If the user does not specify a discount tier, ask them which tier to use. DO NOT ASSUME
            5. The product names are generic. Example: 'laptop', 'headphones', etc. There is no need to ask for more details regarding exact models/names
        
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (in the format of key=value)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:

"""

# ----- Helper: Traced AzureOpenAI call -----

llm = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT # type: ignore
)

@traceable(name="Azure OpenAI Chat", run_type="llm")
def azure_openai_chat_traced(messages:list, options: dict):
    return llm.chat.completions.create(
            model=DEPLOYMENT, #type: ignore
            messages= messages,
            **options
        )

# ----- Agent Loop -----

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    prompt = react_prompt.format(question = question)
    scratchpad = ""

    print(f"Question: {question}")
    print("*" * 60)


    for iteration in range (1, MAX_ITERATIONS + 1):
        print(f"\n----- Iteration {iteration} -----")
        full_prompt = prompt + scratchpad

        response = azure_openai_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop":["\nObservation"], "temperature":0}).choices[0]

        output = response.message.content

        print(" [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input\s*:*(.+)", output)

        if not action_match or not action_input_match:
            print(
                " [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        print(f" [Tool Selected] {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]

        print(f" [Tool Executing] {tool_name}({args})...")

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name} not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"[Tool Result] {observation}")

        scratchpad += f"{output}\nObservation:{observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = (run_agent("What is the price of a laptop after applying a gold discount?"))
