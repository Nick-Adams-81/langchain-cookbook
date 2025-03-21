from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# PART 1: Create a chat prompt template using a template string
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("----- Prompt from Template -----")
prompt = prompt_template.invoke({"topic": "dogs"})
print(prompt)

# Part 2: Prompt with multiple placeholders
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:
"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt2 = prompt_multiple.invoke({"adjective": "funny", "animal": "cat"})
print("\n----- Prompt with multiple placeholders -----")
print(prompt2)

# Part 3: Prompt with System and human messages (Using a tuple)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "tell me {joke_count} jokes.")
]

prompt_template3 = ChatPromptTemplate.from_messages(messages)
prompt3 = prompt_template3.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with ststem and human messages -----")
print(prompt3)