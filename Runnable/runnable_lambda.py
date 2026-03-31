from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableLambda,RunnableParallel,RunnablePassthrough

load_dotenv()
def word_counter(text):
    return len(text.split())
runnable_word_counter=RunnableLambda(word_counter)


model=ChatGroq(model="llama-3.3-70b-versatile")

prompt=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()



prompt1=PromptTemplate(
    template='Explain the following joke-{text}',
    input_variables=['text']
)
chain1 = RunnableSequence(prompt, model, parser)

chain2=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_counter)
})
chain=RunnableSequence(chain1,chain2)
result=(chain.invoke({'topic':'AI'}))


final_result="""{}\n word count - {}""".format(result['joke'],result['word_count'])

print(final_result)