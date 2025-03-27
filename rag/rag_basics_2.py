import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Who is Odysseus' wife?"

# Retrieve relavent docs based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4}
)
relevant_docs = retriever.invoke(query)

# Display relavent docs with metadata
print("\n--- Relevant Documents ---")
for i , doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
