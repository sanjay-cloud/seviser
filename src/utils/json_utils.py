import json
from pathlib import Path

def save_json(json_file, json_data):    
    with open(Path(json_file),"w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file)

def load_json(json_file):
    with open(Path(json_file),"r", encoding="utf-8") as json_data:
        return json.load(json_data)    