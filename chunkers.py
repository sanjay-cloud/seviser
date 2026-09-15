import spacy
from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path
import os
import json

class Chunkers:
    def __init__(self, input_folder, output_folder, output_format = ".json", chunking_type = None, chunk_size = 10, embedding_model = None, num_clusters = 3):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output_format = output_format
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.num_clusters = num_clusters
        self.chunking_type = self.segment_chunking        
        self.create_dir()

    def start(self):
        folder = Path(self.input_folder)
        for file in folder.iterdir():
            if file.is_file():
                file_name = file.name
                with open(file,'r') as f:
                    data = self.chunking_type(json.load(f))
                    extension = Path(file_name).suffix
                    file_name = file_name.replace(extension, self.output_format)        
                    with open(Path(self.output_folder) / file_name,"w", encoding="utf-8") as json_file:
                        json.dump(data, json_file)

    def segment_chunking(self, data):
        data["chunking_strategy"] = "no_chunking"
        data["chunks"] = []
        data["chunks"] = data.pop("segments")

        return data

    def fixed_chunking(self):
        return [self.text_words[i:i+self.chunk_size] for i in range(0, len(self.text_words), self.chunk_size)]

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

    def create_dir(self):
        currentPath = os.getcwd()
        
        if not os.path.exists(Path(currentPath) / self.output_folder):
            os.makedirs(self.output_folder)     

if __name__ == "__main__":
    chunker = Chunkers(input_folder="transcriptions", output_folder="chunks")
    chunker.start()