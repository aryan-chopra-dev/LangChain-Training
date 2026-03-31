

import streamlit as st
import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
# Changed import from HuggingFaceEndpointEmbeddings to HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# --- RAG Pipeline Setup ---
@st.cache_resource
def setup_rag_pipeline(video_id, video_language):
    # 1. Fetch YouTube Transcript
    transcript = ""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=[video_language])
        for snippet in fetched_transcript:
            transcript += (snippet.text + " ") # Corrected: Access 'text' attribute directly
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video or the specified language.")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching transcript: {e}")
        return None

    # 2. Split Text into Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 3. Initialize Embeddings
    # No longer require explicit HUGGINGFACE_API_TOKEN for HuggingFaceEmbeddings if model is public
    # However, for some models or if you hit rate limits, setting HF_TOKEN can still help
    # huggingface_api_token = os.getenv('HF_TOKEN') # Still useful if model requires auth for download
    # if not huggingface_api_token:
    #     st.error("HUGGINGFACE_API_TOKEN not found in .env file.")
    #     return None

    # Debugging step: print a part of the token to verify it's loaded (if still used elsewhere)
    # if huggingface_api_token: st.write(f"Hugging Face Token loaded (partial): {huggingface_api_token[:5]}...{huggingface_api_token[-5:]}")

    # Changed to HuggingFaceEmbeddings and pass model_name directly
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. Define Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    # 6. Define LLM
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in .env file.")
        return None
    os.environ["GROQ_API_KEY"] = groq_api_key # Set for ChatGroq
    llm = ChatGroq(model='llama-3.3-70b-versatile', max_tokens=200)

    # 7. Define Prompt Template
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
        Answer ONLY from the following transcript context.
        If the context is insufficient, just say you don't know.
        Answer the questions only in english
        {context}
        QUESTION:{question}""",
        input_variables=["context", "question"]
    )

    # 8. Define format_docs function
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 9. Build the RAG Chain
    rag_chain = (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit UI ---
st.title("YouTube Transcript Q&A")
st.write("Enter YouTube video details and ask questions about its transcript!")

video_id = st.text_input("Enter YouTube Video ID (e.g., 7xTGNNLPyMI):")
video_language = st.text_input("Enter Video Language Code (e.g., en for English):")

rag_chain = None
if video_id and video_language:
    with st.spinner("Initializing RAG pipeline (fetching transcript and creating vector store). This may take a moment..."):
        rag_chain = setup_rag_pipeline(video_id, video_language)

if rag_chain:
    query = st.text_input("Ask a question about the video transcript:")

    if query:
        with st.spinner("Getting answer..."):
            try:
                response = rag_chain.invoke(query)
                st.write("**Answer:**")
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")
else:
    if video_id and video_language:
        st.warning("RAG pipeline could not be initialized. Please check video ID, language, and API keys.")
    else:
        st.info("Please enter a YouTube Video ID and language to begin.") 