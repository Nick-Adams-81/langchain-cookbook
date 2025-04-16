from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

# List of tools available to the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need the current time."
    )
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
prompt = hub.pull("hwchase17/react")

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run the agent with a test query (agents expect to read 'input')
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response from the agent
print("response: ", response["output"])