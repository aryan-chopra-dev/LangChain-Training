from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGroq(model='llama-3.3-70b-versatile')

template1=PromptTemplate(
    template='Write a detailed report on the {topic}',
    input_variables=['topic']
)
template2=PromptTemplate(
    template='Write a five line summary on the following text.\n {text}',
    input_variables=['text']
)


parser=StrOutputParser()
chain= template1|model|parser|template2|model|parser
result=chain.invoke({'topic':'Internship Oppurtunities in 2026,Timeline included when to register what to prepare'})

print(result)