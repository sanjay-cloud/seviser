import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.retrieval.retriever import Retriever
from src.retrieval.embedder import Embedder
from sentence_transformers import SentenceTransformer
from src.utils import json_utils

class Evaluator:
    def __init__(self, evaluator_file, retriever, evaluation_metric=None, top_k = 5):
        self.evaluator_file = Path(evaluator_file)
        self.retriever = retriever
        # self.evaluation_metric = evaluation_metric
        self.top_k = top_k
    def evaluate(self):
        
        evaluation_json = json_utils.load_json(self.evaluator_file)
        # self.hit_rate_k(query_ids=query_ids)
        hit_count = 0
        rank_count = 0
        for eval in evaluation_json:
            hit, rank= self.get_hit_or_miss(eval["query"], eval["target"])
            hit_count += hit
            rank_count += rank
        hit_rate = (hit_count/len(evaluation_json)) * 100  
        mrr =     rank_count/len(evaluation_json)   
        eval_results = {}
        eval_results["no_of_query"] = len(evaluation_json)
        eval_results["top_k"] = self.top_k
        eval_results["hit_rate"] = hit_rate
        eval_results["mrr"] = mrr
        print(eval_results)
        return eval_results



    def get_hit_or_miss(self, query, target):
        similarities, similarity_ids  = self.retriever.search(query= query, top_k=self.top_k, evaluation = True)
        for id, sim_id in enumerate(similarity_ids):
            if(sim_id == target):
                return 1, 1/(1+id)
                                
        return 0, 0





if __name__ == "__main__":
    # evaluator = Evaluator("evaluator.json", queries=queries, similarities=sims)
    # evaluator.start()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedder = Embedder(embed_model=model) 
    retriever = Retriever(embedder=embedder,metadata_file = "artifacts/metadata.json", index_path = "artifacts/index.faiss")
    # matched_data = retriever.search(query="Where were images of the coach's trainees displayed?", top_k=3)
    evaluator = Evaluator("data/evaluation/evaluator_segment.json", retriever=retriever)
    evaluator.evaluate()