import torch
import torchaudio
from pathlib import Path
import re
import json
from utils import file_utils

class Transcriber:

    def __init__(self, model, processor, input_folder, output_folder, output_format = ".json", device = "cpu"):
        self.model = model
                
        self.model.eval()
        self.processor = processor
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_format = output_format
        self.generation_config = self.model.generation_config               
        self.generation_config.language="english" 
        self.generation_config.task="transcribe"
        self.generation_config.return_timestamps=True  
        self.device = device
        if self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32    

        file_utils.create_dir(self.output_folder)
        self.pattern = r"<\|(\d+(?:\.\d+)?)\|>\s*(.*?)\s*<\|(\d+(?:\.\d+)?)\|>"
        self.supported_audio_formats = {".wav",".flac",".mp3"}


    def transcribe_directory(self):
        print(f"starting transcribing ...")
        for file_path in self.input_folder.iterdir():

            if file_path.is_file() and Path(file_path).suffix in self.supported_audio_formats:
                transcription_path = file_utils.change_file_extension(file_path.name, self.output_format, self.output_folder)
                audio_filename = file_path.name
                if transcription_path.is_file():
                    continue
                self.transcribe_file(file_path=file_path, audio_filename=audio_filename, transcription_path=transcription_path)
                
    def transcribe_file(self, file_path, audio_filename, transcription_path):
        audio_sample, sampling_rate = torchaudio.load(file_path)
        decoded_transcription = self.transcribe_audio(audio_sample=audio_sample, sampling_rate=sampling_rate)
        self.save_transcription(decoded_transcription=decoded_transcription, audio_file_name=audio_filename, output_path=transcription_path)

    @torch.inference_mode()
    def transcribe_audio(self, audio_sample, sampling_rate):
        print(f"generating text")
        input_features = self.processor(audio_sample[0], sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.device, dtype=torch.float16)      

        generated_ids = self.model.generate(
            input_features=input_features, # Ask for timestamps to be generated
            generation_config=self.generation_config
        )    

        timestamped_text = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=False,
            decode_with_timestamps=True # Ask for timestamps to be decode into text
        )      
        return timestamped_text

    def save_transcription(self, decoded_transcription, audio_file_name, output_path):
        print(f"generate trascripts")
        transcription_data = self.parse_transcription(file_name=audio_file_name, decoded_transcription=decoded_transcription)       
        with open(output_path,"w", encoding="utf-8") as json_file:
            json.dump(transcription_data, json_file)

    def parse_transcription(self, file_name, decoded_transcription):
        transcription_data = {}
        transcription_data["filename"] = file_name
        timestamp_matches = re.findall(self.pattern, decoded_transcription)
        segments = [{
                "segment_id":int(segment_id),
                "start":float(start),
                "segment_text":segment_text.strip(),
                "end":float(end)

            }for segment_id, (start, segment_text, end) in enumerate(timestamp_matches)
        ]
        whole_text = "".join([segment["segment_text"] for segment in segments])
        transcription_data["segments"] = segments
        transcription_data["whole_text"] = whole_text
        return transcription_data
            
       


# if __name__ == "__main__":
#     input_folder = "datasets/archive/train/wav"
#     model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo").to("cuda")
#     processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")  
#     transcriber = Transcriber(model=model, processor=processor, input_folder=input_folder, output_folder="transcripts", device= "cuda")
#     transcriber.transcribe()