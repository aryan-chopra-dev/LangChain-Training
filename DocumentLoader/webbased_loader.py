# Load  vs LazyLoad

from langchain_community.document_loaders import DirectoryLoader,TextLoader, PyPDFLoader, WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

loader=WebBaseLoader(
'https://www.youtube.com/watch?v=bL92ALSZ2Cg&t=2092s')

docs=loader.load()

print(len(docs))
print((docs))