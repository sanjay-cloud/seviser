import torch
import torchaudio
from pathlib import Path
import re
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import json

class Transcriber:

    def __init__(self, model, processor, input_folder, output_folder, output_format = ".json"):
        self.model = model
        self.processor = processor
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output_format = output_format
        self.generation_config = self.model.generation_config
        self.create_dir()
        self.pattern = r"<\|(\d+(?:\.\d+)?)\|>\s*(.*?)\s*<\|(\d+(?:\.\d+)?)\|>"


    def transcribe(self):
        print(f"starting transcribing ...")
        folder = Path(self.input_folder)
        for file_path in folder.iterdir():
            if file_path.is_file():
                audio_sample, sampling_rate = torchaudio.load(file_path)
                # print(f"{file_path.name} loaded")
                text = self.generate_text(audio_sample=audio_sample, sampling_rate=sampling_rate)
                self.generate_transcription(text=text, file_name=file_path.name)

    @torch.inference_mode()
    def generate_text(self, audio_sample, sampling_rate):
        print(f"generating text")
        input_features = self.processor(audio_sample[0], sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to("cuda", dtype=torch.float16)      

        
        
        self.generation_config.language="english" 
        self.generation_config.task="transcribe"
        self.generation_config.return_timestamps=True    

        generated_ids = self.model.generate(
            input_features=input_features, # Ask for timestamps to be generated
            generation_config=self.generation_config
        )    

        text_output_with_timestamps = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=False,
            decode_with_timestamps=True # Ask for timestamps to be decode into text
        )      
        return text_output_with_timestamps

    def generate_transcription(self, text, file_name):
        print(f"generate trascripts")
        extension = Path(file_name).suffix
        data = self.create_json(file_name=file_name, text=text)
        file_name = file_name.replace(extension, self.output_format)        
        with open(Path(self.output_folder) / file_name,"w", encoding="utf-8") as json_file:
            json.dump(data, json_file)

    def create_json(self, file_name, text):
        data = {}
        data["filename"] = file_name
        matches = re.findall(self.pattern,text)
        segments = [{
                "id":int(id),
                "start":float(start),
                "text":text.strip(),
                "end":float(end)

            }for id, (start, text, end) in enumerate(matches)
        ]
        
        data["segments"] = segments
        return data
            
    def create_dir(self):
        currentPath = os.getcwd()
        
        if not os.path.exists(Path(currentPath) / self.output_folder):
            os.makedirs(self.output_folder)   
       


if __name__ == "__main__":
    input_folder = "datasets/archive/train/wav"
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo").to("cuda")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")          
    model.eval()
    transcriber = Transcriber(model=model, processor=processor, input_folder=input_folder, output_folder="transcriptions")
    transcriber.transcribe()