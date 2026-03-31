from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the endpoint with the CORRECT task
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", # Changing this from 'conversational' fixes the error
)

# 2. Wrap it for Chat
# ChatHuggingFace will use its internal template to format your messages
model = ChatHuggingFace(llm=llm)

# 3. Use the message format
messages = [
    ("system", "You are a funny assistant."),
    ("human", "Write 5 joke to make me laugh like haa haa"),
]


result = model.invoke(messages)
print(result.content)
