
import faiss
import copy
import utils.json_utils as json_utils

class Retriever:
    def __init__(self, embedder, index, metadata_file= "metadata.json"):
        self.embedder = embedder
        self.index = index
        self.metadata = json_utils.load_json(metadata_file)

    def search(self, query, top_k):        
        query_embeddings = self.embedder.encode([query], batch_size = 1)
        faiss.normalize_L2(query_embeddings)
        similarities, similarity_ids = self.index.search(query_embeddings, top_k = 5)
    
        return self.get_matched_data(query, similarities[0], similarity_ids[0])

    def get_matched_data(self, query, similarities, similarity_ids):
        matched_data = []
        for similarity, similarity_id in zip(similarities, similarity_ids):
            data = copy.copy(self.metadata[similarity_id])
            data["query"] = query
            data["similarity"] = similarity
            matched_data.append(data)            

        return matched_data