from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load env variables
load_dotenv()

# Create instance of chat model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Use an empty list to store chat history
chat_history = []

# Set an initial system message (optional but reccomended)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

# Create the chat loop
while True:

    # Get user query
    query = input("User: ")
    
    # Check if user query is quit, if so break out of the chat loop
    if query.lower() == "quit":
        print(f"AI: Goodbye")
        break
    chat_history.append(HumanMessage(content=query))

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

print("----- Message History -----")
print(chat_history)
    
