from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")


messages=[
    SystemMessage(content="You are a helpful assistant "),
    HumanMessage(content="Tell me about LangChain ")

]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)