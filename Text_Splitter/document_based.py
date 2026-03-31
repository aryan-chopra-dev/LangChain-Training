#Used for special documents like codes

from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

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
splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)
chunks=splitter.split_text(text)
print(len(chunks))