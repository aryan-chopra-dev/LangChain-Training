from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')
model=ChatGroq(model="llama-3.3-70b-versatile")
user_input=st.text_input('Enter your Prompt')

paper_input = st.selectbox("Select Research Paper Name",["Attention is All you Need","NLP and the Internet","ChatGPT and LLM"])
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]) 

template=load_prompt('template.json')
if(st.button('Summarize')):
    
    chain=template|model
    result=chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    
    st.write(result.content)