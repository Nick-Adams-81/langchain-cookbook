from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()

# Create instance of model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tell jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes.")
    ]
)

# Define additional processing steps using RunneableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n{x}")

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Invoke the chain 
result = chain.invoke({"topic": "cats", "joke_count": 3})

# print the result
print(result)