from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",output_dimensionality=300)

documents = [
    "Sachin Tendulkar, known as the 'Master Blaster,' holds the record for the highest number of international centuries [11].",
    "Virat Kohli redefined modern batting with his incredible consistency and aggressive chase mastery [2].",
    "Sir Donald Bradman’s unparalleled Test batting average of 99.94 remains an unbreakable record in cricket history.",
    "Vivian Richards dominated bowlers with unmatched aggression, revolutionizing attacking batting in the 1970s and 80s.",
    "Shane Warne revived the art of leg-spin, famously bowling the 'Ball of the Century' to Mike Gatting.",
    "Wasim Akram, the 'Sultan of Swing,' revolutionized fast bowling with his mastery over reverse swing.",
    "Brian Lara holds the record for the highest individual scores in both Test (400*) and first-class cricket.",
    "Muttiah Muralitharan remains the highest wicket-taker in both Test and ODI cricket, mesmerizing batsmen with off-spin.",
    "MS Dhoni, known as 'Captain Cool,' is the only captain to win all three major ICC white-ball trophies.",
    "Ricky Ponting led Australia to unprecedented success, known for his prolific run-scoring and astute captaincy."
]

query="Tell me about Unicorn"

doc_embeddings=embedding.embed_documents(documents)

query_embedding=embedding.embed_query(query)

score=cosine_similarity([query_embedding],doc_embeddings)[0]

index,score=sorted(list(enumerate(score)),key=lambda x:x[1])[-1]
print(documents[index],score)