import numpy as np

class Embedder:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        
    def encode(self, text, batch_size=None):
        return self.embed_model.encode(text, batch_size = batch_size).astype(np.float32)
    
    
    

# if __name__ == "__main__":
#     hf_logging.set_verbosity_error()
#     transformers_logging.set_verbosity_error()
#     model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2" )
#     print("Enter your query to search")
#     query = input()

#     embedder = Embedder(input_folder="chunks", embed_model=model, query= query) 
#     embedder.start()  