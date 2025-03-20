from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    SystemMessage(content="Solve the folowing math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

result = llm.invoke(messages)
print(f"AI: {result.content}")

messages = [
    SystemMessage(content="Solve the folowing math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

result = llm.invoke(messages)
print(f"AI: {result.content}")