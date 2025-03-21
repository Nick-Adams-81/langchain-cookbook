from dotenv import load_dotenv
from google.auth import compute_engine
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

"""
Steps to replace this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authentiucate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your local project to the new Firebase project you created
"""

load_dotenv()

# Set up Firebase firestore
PROJECT_ID = "chat-history-18f43"
SESSION_ID = "user1_session" # This can be a username or a unique id
COLLECTION_NAME = "chat_history"

# Initialize a firestore client
client = firestore.Client(project=PROJECT_ID)

# Create a chat history that gets saved to firestore
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)

# Using an AI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Starting the chat loop
while True:
    user_query = input("User: ")
    if user_query.lower() == "quit":
        break

    chat_history.add_user_message(user_query)

    response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(response.content)

    print(f"AI: {response.content}")
    


