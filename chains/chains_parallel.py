from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()

# Create instance of model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}")
    ]
)

# Function to analyze pros step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features: {features}, list the pros of these features.")
        ]
    )
    return pros_template.format_prompt(features=features)

# Function to analyze cons step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given theswe features: {features}, list the cons of these features.")
        ]
    )
    return cons_template.format_prompt(features=features)

# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n\n{pros}\n\nCons:\n{cons}"

# Simplify branches with LCEL (pros branch)
pros_branch = (
    RunnableLambda(lambda x : analyze_pros(x)) | model | StrOutputParser()
)

# Simplify branches with LCEL (cons branch)
cons_branch = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template | 
    model | 
    StrOutputParser() | 
    RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch}) |
    RunnableLambda(lambda x : combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Print the result
print(result)