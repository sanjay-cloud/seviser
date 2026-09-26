from pathlib import Path
import utils.json_utils as json_utils
import json
import faiss
import numpy as np

class Indexer:
    def __init__(self, embedder, input_folder, metadata_file = "metadata.json"):
        
        self.embedder = embedder
        self.input_folder = Path(input_folder)
        self.index = None
        self.metadata_file = metadata_file
        self.metadata = []
        self.texts = []

    def build_index(self):
        self.prepare_data()
        json_utils.save_json(self.metadata_file, self.metadata)
        
        self.create_index()
        print(f"seaching across {len(self.metadata)} trnscript chunks")
    
    
    def prepare_data(self):
        for file in self.input_folder.iterdir():
            if file.is_file(): 
                self.create_metadata(json_utils.load_json(file))         
        
    
    def create_metadata(self,chunk_data):  
        for chunk in chunk_data["chunks"]:    
            chunk["filename"] = chunk_data["filename"]                     
            self.metadata.append(chunk) 
            self.texts.append(chunk["segment_text"])


    def create_index(self):
        embeddings = self.embedder.encode(self.texts,batch_size = 64)
        dim = embeddings.shape[1] 

        db_ids = np.arange(0, len(self.metadata))

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, db_ids)
        
