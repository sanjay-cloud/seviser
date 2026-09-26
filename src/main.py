from transcription.transcriber import Transcriber

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from chunking.chunkers import Chunkers
from retrieval.embedder import Embedder
from retrieval.indexer import Indexer
from retrieval.retriever import Retriever
from sentence_transformers import SentenceTransformer
from huggingface_hub.utils import logging as hf_logging
from transformers.utils import logging as transformers_logging

# def get_matched_data(self, sims):
#     matched_data = []
#     with open(self.index_data_file,'r') as f:
#         data = json.load(f) 
#     for sim in sims:
#         matched_data.append(data[sim])            

#     return matched_data

# def print_results(self, chunks, similarities):
#     for id,chunk in enumerate(chunks):
#         filename = chunk["filename"]
#         text = chunk["segment_text"]
#         start = chunk["start"]
#         end = chunk["end"]
                
#         print()
#         print(f"Result {id+1}")
#         print(f"file: {filename}")  
#         print(f"Timestamp: {float(start)} -> {float(end)}")        
#         print()
#         print(f"{text}")    
        
#         print()
#         print(f"similarity: {similarities[0][id]}")     
#         print("================================")   

if __name__ == "__main__":
    input_folder = "datasets/archive/train/wav"
    # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo").to("cuda")
    # processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")  
    # transcriber = Transcriber(model=model, processor=processor, input_folder=input_folder, output_folder="transcripts", device= "cuda")
    # transcriber.transcribe_directory()
    chunker = Chunkers(input_folder="transcripts", output_folder="chunks")
    chunker.start()

    hf_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2" )

    # embedder = Embedder(input_folder="chunks", embed_model=model) 
    # embedder.build_index()
    # matched_data = embedder.search(query="Where were images of the coach's trainees displayed?", top_k=3)
    # print(matched_data)

    embedder = Embedder(embed_model=model) 
    indexer = Indexer(embedder=embedder, input_folder="chunks")
    indexer.build_index()
    retriever = Retriever(embedder=embedder, index=indexer.index)
    matched_data = retriever.search(query="Where were images of the coach's trainees displayed?", top_k=3)
    print(matched_data)