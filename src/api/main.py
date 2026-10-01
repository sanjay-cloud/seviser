from fastapi import FastAPI
from src.retrieval.retriever import Retriever
from src.retrieval.embedder import Embedder
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2" )
embedder = Embedder(embed_model=model) 
retriever = Retriever(embedder=embedder)

@app.get("/search")
def search(query: str, top_k: int = 3):
    matched_data = retriever.search(query=query, top_k = top_k)
    return matched_data