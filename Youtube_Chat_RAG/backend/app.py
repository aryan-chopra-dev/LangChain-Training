import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI()

class AskRequest(BaseModel):
    video_id: str
    video_language: str = "en"
    question: str

def normalize_text(t: str) -> str:
    return " ".join(t.replace("\n", " ").split())

@lru_cache(maxsize=16)
def setup_rag_pipeline(video_id: str, video_language: str):
    transcript = ""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=[video_language])
        for snippet in fetched_transcript:
            transcript += snippet.text + " "
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="Transcripts are disabled for this video.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript fetch failed: {e}")

    transcript = normalize_text(transcript)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=250,
        temperature=0,
        groq_api_key=groq_api_key
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY from the following transcript context.
If the context is insufficient, just say: "I don't know from the transcript."
Answer only in english
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""",
        input_variables=["context", "question"]
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    rag_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    chain = setup_rag_pipeline(req.video_id, req.video_language)
    try:
        answer = chain.invoke(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)