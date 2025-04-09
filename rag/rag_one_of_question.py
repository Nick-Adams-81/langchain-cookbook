import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load env variables
load_dotenv()

# Define persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define embeddding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load existing vector store with embedding function
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Define user's query
query = "How did Juliet die?"

# retrieve relevant docs based on the query
retriever = db.as_retriever(
    search_type="similarity",
    ssearch_kwargs={"k": 1}
)
relevant_docs = retriever.invoke(query)
# Display the relevant results with metadata
print("\n--- Relevant Docs ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant doc contents
combined_input = (
    "Here are some documents that might help answer the question:"
    + query 
    + "\n\nRelevant Docs:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provideddocuments"
)

# Create a chat OpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Invoke the model with combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
print(result)
print("Content only: ")
print(result.content)