from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Define tools
def get_current_time(*args, **kwargs):
    """Returns current time in H:MM AM/PM format."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """Searches wikipedia and returns the summary of the first result"""
    from wikipedia import summary
    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."

# Define the tools that the agent can use
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time"
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic"
    )
]

# Load the correct JSON chat prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create a structured Chat Agent with conversation buffer memory
# ConversationalBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# It combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = """
You are an AI assistant that can provide helpful answers using available tools.
If you are unable to answer, just say you don't know.
"""
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    # Add the users message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current cha history
    response = agent_executor.invoke({"input": user_input})
    print("Bot: ", response["output"])

    # Add the agent's response to the conversation history
    memory.chat_memory.add_message(AIMessage(content=response["output"]))