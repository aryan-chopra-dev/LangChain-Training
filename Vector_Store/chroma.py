from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv() 
doc1=Document(
    page_content="Sachin Tendulkar, known as the 'Master Blaster,' holds the record for the highest number of international centuries [11].",
    metadata={"team":"Mumbai Indians"}
)
doc2=Document(
    page_content="Virat Kohli redefined modern batting with his incredible consistency and aggressive chase mastery ",
    metadata={"team":"Royal Challengers Bangalore"}
)
doc3=Document(
    page_content="MS Dhoni, known as 'Captain Cool,' is the only captain to win all three major ICC white-ball trophies.",
    metadata={"team":"Chennai Super Kings"}
)
doc4=Document(
    page_content="Rohit Sharma: A highly successful captain who has led Mumbai Indians to six IPL title victories.",
    metadata={"team":"Mumbai Indians"}
)
doc5=Document(
    page_content="Suresh Raina: Known as 'Mr. IPL,' he was a consistent run-scorer and crucial part of Chennai Super Kings' success.",
    metadata={"team":"Chennai Super Kings"}
)

docs=[doc1,doc2,doc3,doc4,doc5]

vector_store=Chroma.from_documents(
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",output_dimensionality=300),
    persist_directory='chroma_db',
    collection_name='sample'
)

vector_store.add_documents(docs)

vector_store.get(include=['embeddings','documents','metadatas'])
vector_store.similarity_search(
    query="Who among these are a bowler",k=2
)
vector_store.similarity_search_with_score(
    query="Who among these are a bowler",k=2

)
vector_store.similarity_search_with_score(
    query="",filter={"team":"Chennai Super Kings"}

)
print(vector_store)