from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableBranch

load_dotenv()
def word_counter(text):
    return len(text.split())
runnable_word_counter=RunnableLambda(word_counter)


model=ChatGroq(model="llama-3.3-70b-versatile")


parser=StrOutputParser()



prompt1=PromptTemplate(
    template='Provide a report on-{text}',
    input_variables=['text']
)
chain1 = RunnableSequence(prompt1, model, parser)
word_counts=RunnableLambda(word_counter)


prompt2=PromptTemplate(
    template='Provide a summary on-{text}',
    input_variables=['text']
)
chain2=RunnableBranch(
    (lambda x:word_counter(x)>500,prompt2|model|parser),
    (RunnablePassthrough())
)

chain=chain1|chain2


result=(chain.invoke({'text':'AI'}))



print(result)