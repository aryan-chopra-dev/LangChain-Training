from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

# 1. Define the desired schema

load_dotenv()
model=ChatGroq(model='llama-3.3-70b-versatile')

class Student(BaseModel):
    name:str=Field(description="Provide the name of the student"),
    age:int=Field(gt=18,description="Provide the age of the student"),
    city:str=Field(description="Provide the city of the student"),
# 2. Initialize the parser using the schemas
parser = PydanticOutputParser(pydantic_object=Student)



template=PromptTemplate(
    template='Give the name,age,cityof a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt1=template.invoke({'place':'Indian'})
prompt2=template.invoke({'place':'Pakistan'})
prompt3=template.invoke({'place':'Afghanistan'})
prompt4=template.invoke({'place':'Babath'})

result1=model.invoke(prompt1)
result2=model.invoke(prompt2)
result3=model.invoke(prompt3)
result4=model.invoke(prompt4)
print(result1.content)
print(result2.content)
print(result3.content)
print(result4.content)

print("\n\n",prompt1)
