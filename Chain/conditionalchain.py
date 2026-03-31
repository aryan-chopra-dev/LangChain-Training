from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core .output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()



class Classify(BaseModel):
    sentiment:Literal['Positive','Negative']=Field(description="Classify  the sentiment of the feedback and return Positive or Negative")

parser2=PydanticOutputParser(pydantic_object=Classify)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Classify  the sentiment of the feedback {feedback} and return Positive or Negative\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)



model=ChatGroq(model="llama-3.3-70b-versatile")

chain1=prompt1|model|parser2


prompt2=PromptTemplate(
    template="Write an appropriate positive (grateful) response to this positive feedback {feedback}",
    input_variables=["feedback"],
)
prompt3=PromptTemplate(
    template="Write an appropriate (sorry) response to this negative feedback {feedback}",
    input_variables=["feedback"],
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='Positive',prompt2|model|parser),
    (lambda x:x.sentiment=='Negative',prompt3|model|parser),
    RunnableLambda(lambda x:"Could not find sentiment")
    )

chain=chain1|branch_chain

result=chain.invoke({"feedback":"The experience was inevitable"})

print(result)