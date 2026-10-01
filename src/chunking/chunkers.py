import spacy
from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path
import os
import json
import utils.file_utils as file_utils
import utils.json_utils as json_utils
class Chunkers:
    def __init__(self, input_folder, output_folder, output_format = ".json", chunking_type = None, embedding_model = None, num_clusters = 3):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_format = output_format
        self.embedding_model = embedding_model
        self.num_clusters = num_clusters
        self.supported_formats = [".json"]
        self.chunking_type = self.segment_chunking  
        file_utils.create_dir(self.output_folder)

    def start(self):
        folder = Path(self.input_folder)
        for file_path in folder.iterdir():
            if file_path.is_file() and Path(file_path).suffix in self.supported_formats:
                chunk_path = file_utils.change_file_extension(file_path.name, self.output_format, self.output_folder)
                if chunk_path.is_file():
                    continue
                file_name = file_path.name
                with open(file_path,'r') as f:
                    data = self.chunking_type(json.load(f))
                    file_name = file_utils.change_file_extension(file_name=file_name, extension=self.output_format, output_folder=self.output_folder) 
                    json_utils.save_json(json_file=file_name, json_data=data)  
    
    def segment_chunking(self, data):
        data["chunking_strategy"] = "no_chunking"
        data["chunks"] = []
        data["chunks"] = data.pop("segments")

        return data

    def fixed_chunking(self, data, chunk_size):
        data["chunking_strategy"] = "fixed_chunking"
        segments = data["segments"]
        # chunks =  [self.text_words[i:i+chunk_size] for i in range(0, len(self.text_words), self.chunk_size)]
        chunks = []
        chunk_text = ""
        for id, segment in enumerate(segments):
            chunk = {}
            if len(segment["segment_text"].split()) >= chunk_size:
                chunk["segment_id"] = segment["segment_id"]
                chunk["start"] = segment["start"]
                chunk["segment_text"] = segment["segment_text"]
                chunk["end"] = segment["end"]
                chunks.append(chunk)
            else:
                chunk_text += segment["segment_text"]

                if len(chunk_text.split()) >= chunk_size:
                    chunk["segment_id"] = segment["segment_id"]
                    chunk["start"] = segment["start"]
                    chunk["segment_text"] = chunk_text
                    chunk["end"] = segment["end"] 
                    chunk_text = ""
                    chunks.append(chunk)
        data["chunks"] = chunks
        data.pop("segments")
        return data



    def sentence_chunking(self):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(self.text)
        return [sent.text for sent in doc.sents]


    def semantic_chunking(self):
        sentences = self.sentence_chunking()
        embeddings = self.embedding_model(sentences)
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(np.array(embeddings))
        clusters = [[] for _ in range(self.num_clusters)]
        for i , label in enumerate(kmeans.labels_):
             clusters[label].append(sentences[i])

        return clusters  
  

if __name__ == "__main__":
    chunker = Chunkers(input_folder="transcriptions", output_folder="chunks")
    chunker.start()