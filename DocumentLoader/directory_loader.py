# Load  vs LazyLoad

from langchain_community.document_loaders import DirectoryLoader,TextLoader, PyPDFLoader, WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()
loader=DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
docs=loader.load()


# chain=prompt1|model|parser
# print(chain.invoke({'text':docs[0].page_content}))


print(type(docs))
print(len(docs))
print(docs[0])
# print(len(docs[0].page_content.split()))