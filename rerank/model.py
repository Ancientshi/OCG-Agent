from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from FlagEmbedding import FlagReranker
import requests

class LLMRerank:
    def __init__(self, url='http://localhost:8501'):
        self.url = url
    
    def predict(self, query, documents,windows_size=20,step=10, model_name = "gpt-4o-mini"):
        response = requests.post(f"{self.url}/compute_scores", json={"query": query, "documents": documents, "window_size": windows_size, "step": step, "model_name": model_name})
        scores = response.json()['scores']
        return scores