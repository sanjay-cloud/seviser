from embedder import Embedder
import json
from sentence_transformers import SentenceTransformer

class Evaluator:
    def __init__(self, global_json_file, queries, similarities, evaluation_metric=None):
        self.global_json_file = global_json_file
        self.queries = queries
        self.evaluation_metric = evaluation_metric
        self.similarities = similarities
    def start(self):
        query_ids = self.get_query_ids()
        self.hit_rate_k(query_ids=query_ids)

    def get_query_ids(self):
        query_ids = []
        with open(self.global_json_file, 'r') as f: #this for reading the file
            global_json = json.load(f)
            for query in self.queries:
                for data in global_json:
                    if data["query"] == query:
                        query_ids.append(data["target"])     
        return query_ids
    def hit_rate_k(self, query_ids):
        hit_count = 0
        for id, similarity in enumerate(self.similarities):
            for sim in similarity:
                if(sim == query_ids[id]):
                    hit_count+=1
                    print(query_ids[id])
                    print(similarity)
                    break

        hit_rate = (hit_count/len(query_ids)) * 100        
        print(hit_rate)


if __name__ == "__main__":
    queries = ["What kind of scan was the speaker trying to obtain?",
"How does the speaker characterize what has driven their life?",
"What record involving military decorations is being described?",
"What made the athlete unique among the American sprinters?",
"How did people respond when the speaker asked for participation?",
"What was the research hoping to discover about childhood disabilities?",
"What contribution by Tina Brown does the speaker appreciate?",
"Which Native American activists are mentioned by name?",
"What preserved human organ did the speaker get to physically hold?",
"Where were images of the coach's trainees displayed?",
"Which publication does the speaker associate with the photograph?",
"What function did the substance serve?",
"Which publication did the speaker previously work for?",
"Why did the speakers choose an initial starting point?",
"What unusual religious experiment did the speaker undertake for an entire year?",
"Who did the speaker meet as a result of the events being described?"]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", )
    embedder = Embedder(input_folder="chunks", embed_model=model, query= queries) 
    sims = embedder.start()  
    evaluator = Evaluator("evaluator.json", queries=queries, similarities=sims)
    evaluator.start()
