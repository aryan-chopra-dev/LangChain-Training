import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load Environment Variables
load_dotenv()

# Check if API Key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    exit()

# 2. Define Documents
doc1 = Document(
    page_content="Sachin Tendulkar, known as the 'Master Blaster,' holds the record for the highest number of international centuries [11].",
    metadata={"team": "Mumbai Indians"}
)
doc2 = Document(
    page_content="Virat Kohli redefined modern batting with his incredible consistency and aggressive chase mastery.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc3 = Document(
    page_content="MS Dhoni, known as 'Captain Cool,' is the only captain to win all three major ICC white-ball trophies.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Rohit Sharma: A highly successful captain who has led Mumbai Indians to six IPL title victories.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Suresh Raina: Known as 'Mr. IPL,' he was a consistent run-scorer and crucial part of Chennai Super Kings' success.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]


# 3. Initialize Embeddings
# We use 'text-embedding-004' which supports output_dimensionality
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=300,
    task_type="retrieval_document"
)

print("Creating vector store...")

# 4. Create Vector Store
# CRITICAL FIX: Use 'embedding=' instead of 'embedding_function=' for .from_documents()
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="sample"
)

# 5. Create Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# 6. Query
query = "Who is Captain Cool?"
print(f"\nQuerying: '{query}'")

results = retriever.invoke(query)

# 7. Display Results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

print("\nDone.")