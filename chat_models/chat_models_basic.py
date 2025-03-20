from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()

# Create an instance of an llm(can be any model you choose, i'm using gpt-3.5-turbo)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Invoke the model with a question
result = llm.invoke("What is the capitol of California?")
print("Full Result: ")
print(result)
print("Response Only: ")
print(result.content)
