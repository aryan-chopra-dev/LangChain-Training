from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Review(BaseModel):
    name:str
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=5)

student_review1={'name':'Aryan','age':32,'email':'abc@kawai.km','cgpa':4}
student=Review(**student_review1)

student_dict=(dict(student))

print(student_dict['age'])

student_json=student.model_dump_json()
print(student_json)