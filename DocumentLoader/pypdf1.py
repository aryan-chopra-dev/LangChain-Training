from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

# model=ChatGroq(model="llama-3.3-70b-versatile")

# prompt=PromptTemplate(
#     template='Write a joke about {topic}',
#     input_variables=['topic']
# )

# parser=StrOutputParser()



# prompt1=PromptTemplate(
#     template='Explain the following poem-{text}',
#     input_variables=['text']
# )
loader=PyPDFLoader('Assignment.pdf')

docs=loader.load()

# chain=prompt1|model|parser
# print(chain.invoke({'text':docs[0].page_content}))


print(type(docs))
print(len(docs))
print(docs)
# print(len(docs[0].page_content.split()))