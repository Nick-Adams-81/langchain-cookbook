from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?")
]

model1 = ChatOpenAI(model="gpt-3.5-turbo")
result1 = model1.invoke(messages)
print(f"OpenAI: {result1.content}")

model2 = ChatAnthropic(model="claude-3-opus-20240229")
result2 = model2.invoke(messages)
print(f"Anthropic: {result2.content}")

model3 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
result3 = model3.invoke(messages)
print(f"Gemini: {result3.content}")