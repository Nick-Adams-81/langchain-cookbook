from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Functions for the tools
def greet_user(name: str) -> str:
    """Greets the user by name"""
    return f"Hello, {name}!"

def reverse_string(text: str) -> str:
    """Reverses a string"""
    return text[::-1]

def concatenate_string(a: str, b: str) -> str:
    """Concatenates two strings"""
    return a + b

# Pydantic model for tool arguments
class ConcatenatesStringArgs(BaseModel):
    a: str = Field(description="First String")
    b: str = Field(description="second string")

# Create tools using the Tool and StructuredTool constructor approach
tools = [
    # Use Tool for simpler functions with a single input parameter
    # This is straightforward and doesn't require an input schema
    Tool(
        name="Greeter",
        func=greet_user,
        description="Greets the user by name"
    ),
    Tool(
        name="ReverseString",
        func=reverse_string,
        description="Reverses a given string"
    ),
    # Use StructuredTool for more complex functions that require multiple input parameters
    # StructuredTool allows us to define an inout schema using Pydantic, ensuring proper validation
    StructuredTool.from_function(
        func=concatenate_string,
        name="ConcatenateStrings",
        description="Concatenates two strings",
        args_schema=ConcatenatesStringArgs
    )
]

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the ReAct agent using the create_tool_calling_agent function
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice': ", response)

response2 = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello': ", response2)

response3 = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world': ", response3)