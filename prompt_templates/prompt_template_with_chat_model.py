from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()

# Create instance of LLM
model = ChatOpenAI(model="gpt-3.5-turbo")

# Part 1: Create a chat prompt template using a template string
print("----- Prompt from template -----")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)

print(result.content)

# Part 2: Prompt with multiple placeholders
print("----- Prompt with multiple placeholders -----")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:
"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt2 = prompt_multiple.invoke({"adjective": "funny", "animal": "dog"})
result2 = model.invoke(prompt2)
print(result2.content)

# Part 3: Prompt with system and human messages (using tuples)
print("\n----- Prompt with system and human messages -----")
messages = [
    ("system", "You are a comedian who tells jokes about {topic},"),
    ("human", "Tell me {joke_count} jokes")
]
prompt_template3 = ChatPromptTemplate.from_messages(messages)
prompt3 = prompt_template3.invoke({"topic": "lawyers", "joke_count": 3})
result3 = model.invoke(prompt3)
print(result3.content)