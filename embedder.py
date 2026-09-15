import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
class embedder:
    def __init__(self, embed_model, input_folder, global_json_file = "global.json", query = None):
        self.embed_model = embed_model
        self.input_folder = input_folder
        self.global_json_file = global_json_file
        self.query = query

    def start(self):
        data, global_data, vector_count = self.get_all_chunks()
        with open(Path(self.global_json_file),"w", encoding="utf-8") as json_file:
            json.dump(global_data, json_file)
        
        index = self.create_index(data, vector_count)
        self.search(index)

    def get_all_chunks(self):
        data = []
        global_data = []
        folder = Path(self.input_folder)
        vector_count = 0
        for id, file in enumerate(folder.iterdir()):
            if file.is_file():
                with open(file,'r') as f:
                    metadata = json.load(f)
                    metadata.pop("chunking_strategy")
                    global_data.append(metadata)
                    chunks = metadata["chunks"]  
                    for chunk in chunks:
                        data.append(chunk["text"])
                        chunk["vector_id"] = vector_count                           
                        vector_count+=1             
        return data, global_data, vector_count        


    def create_index(self, data, vector_count):
        embeddings = self.embed_model.encode(data,batch_size = 64)
        print(embeddings.shape[0])
        dim = embeddings.shape[1] 

        db_vectors = embeddings.copy().astype(np.float32)
        db_ids = np.arange(0, vector_count)

        faiss.normalize_L2(db_vectors)

        index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(db_vectors, db_ids)

        return index

    def search(self, index):
        query_embeddings = self.embed_model.encode([self.query]).astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        similarities, similarity_ids = index.search(query_embeddings, 5)
        print(similarities)
        print(similarity_ids)
        


if __name__ == "__main__":
    query = "and toward the end, I looked like this."
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", )
    embedder = embedder(input_folder="chunks", embed_model=model, query= query) 
    embedder.start()        