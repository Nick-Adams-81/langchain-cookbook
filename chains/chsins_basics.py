from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()

# Create an instance of an LLM
model = ChatOpenAI(model="gpt-3.5-turbo")

# Create a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes.")
    ]
)

# Create a chain with the prompt template, model, and the StrOutputParser
# This chain is an example of LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()

# Create a result using the invoke method of the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Print result
print(result)