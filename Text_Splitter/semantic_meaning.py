#Used for semantic meaning distribution of chunks

from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings


text="""
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

message=[
    SystemMessage(content="You are a helpful AI assistant ")
]
chat_history=[]
while(True):
    user_input=input("You: ")
    if(user_input=="exit"):
        break
    chat_history.append(HumanMessage(content=user_input))
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

"""
splitter=SemanticChunker(
    GoogleGenerativeAIEmbeddings(),breakpint_threshhold_type="standard_deviation",
    breakpint_threshhold_amount=1
)
chunks=splitter.create_documents([text])
print(len(chunks))
print(chunks)