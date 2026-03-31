from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

prompt=PromptTemplate(
    template='Write a tweet about {topic}',
    input_variables=['topic']
)
prompt1=PromptTemplate(
    template='Write a LinkedIn Post about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()




chain = RunnableParallel({
    'tweet':RunnableSequence(prompt,model,parser),
    'LinkedIn Post':RunnableSequence(prompt1,model,parser),
})
print(chain.invoke({'topic':'AI'}))
