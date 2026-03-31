import fitz
from langchain_core.documents import Document
from transformers import CLIPProcessor,CLIPModel
from PIL import Image
import torch
import numpy as numpy
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#Clip Model
from dotenv import load_dotenv
load_dotenv()

clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(clip_model.eval())

#Embedding functions

def embed_image(image_data):
    if isinstance(image_data,str):
        image=Image.open(image_data.convert("RGB"))
    else:
        image=image_data
    clip_processor(images=image,return_tensors="pt")
    with torch.no_grad():
        features=clip_model.get_image_features(**inputs)
        #Normalize embeddings to a unit vector
        features=features/features.norm(dim=1,keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    inputs=clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        features=clip_model.get_text_features(**inputs)
        #Normalize embeddings to a unit vector
        features=features/features.norm(dim=1,keepdim=True)
        return features.squeeze().numpy()

pdf_path="sample_1.pdf"
doc=fitz.open(pdf_path)
all_docs=[]
all_embedding=[]
image_data_store={}

#Text Splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,

)

for i,page in enumerate(doc):
    text=page.get_text()
    if text.strip():
        temp_doc=Document(page_content=text,metadata={"page":i,"type":"text"})
        print("Temp-Doc",temp_doc.page_content[:10])
        text_chunks=splitter.split_documents([temp_doc])
        print(text_chunks[0])
        for chunk in text_chunks:
            embeddings = embed_text(chunk.page_content)
            all_embedding.append(embeddings)
            all_docs.append(chunk)
            print(f"Embedding added, total: {len(all_embedding)}")
            print(f"Doc content: {chunk.page_content[:50]}...")

