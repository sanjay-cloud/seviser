
import faiss
import copy
# import utils.json_utils as json_utils
from src.utils import json_utils

class Retriever:
    def __init__(self, embedder, metadata_file= "metadata.json", index_path = "index.faiss"):
        self.embedder = embedder
        self.index = faiss.read_index(index_path)
        self.metadata = json_utils.load_json(metadata_file)

    def search(self, query, top_k = 5, evaluation = False):        
        query_embeddings = self.embedder.encode([query], batch_size = 1)
        faiss.normalize_L2(query_embeddings)
        similarities, similarity_ids = self.index.search(query_embeddings, top_k)
        if evaluation:
            return similarities[0], similarity_ids[0]
        else:    
            return self.get_matched_data(query, similarities[0], similarity_ids[0])

    def get_matched_data(self, query, similarities, similarity_ids):
        matched_data = []
        for similarity, similarity_id in zip(similarities, similarity_ids):
            data = copy.copy(self.metadata[int(similarity_id)])
            data["query"] = query
            data["similarity"] = float(similarity)
            matched_data.append(data)            

        return matched_data