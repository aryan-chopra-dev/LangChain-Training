from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile',temperature=1,max_tokens=100)
result=model.invoke("Write a joke to make me laugh like haa haa")
print(result.content)
