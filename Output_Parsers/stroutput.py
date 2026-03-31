from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGroq(model='llama-3.3-70b-versatile',max_tokens=200)

template1=PromptTemplate(
    template='Write a detailed report on the {topic}',
    input_variables=['topic']
)
template2=PromptTemplate(
    template='Write a five line summary on the following text.\n {text}',
    input_variables=['text']
)



prompt1=template1.invoke({'topic':" Suggest some good projects (that have a good impact)for Data Science Roles in 2026"})

result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result.content})

result2=model.invoke(prompt2)
print(result2)