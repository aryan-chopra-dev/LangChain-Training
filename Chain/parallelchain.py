from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

prompt1=PromptTemplate(
    template="Provide a detailed notes on  {topic}",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="Provide a quiz on {topic}",
    input_variables=["topic"]
)
prompt3=PromptTemplate(
    template="Merge the notes and quiz provided {notes} and {quiz}",
    input_variables=["notes","quiz"]
)


model=ChatGroq(model="llama-3.3-70b-versatile")

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model|parser,
    'quiz':prompt2|model|parser,
})
merge_chain=prompt3|model|parser
chain=parallel_chain|merge_chain

result=chain.invoke({'topic':'Linear Regression'})

print(result)
chain.get_graph().print_ascii()