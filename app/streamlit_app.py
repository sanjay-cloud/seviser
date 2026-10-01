import streamlit as st
import requests
from pathlib import Path
from io import BytesIO

from pydub import AudioSegment, effects


BASE_URL = "http://127.0.0.1:8000"
input_folder = Path("datasets/archive/train/wav")     


query_text = st.chat_input("Hello ,enter your search query")
    

@st.cache_data
def prepare_audio_segment(
    audio_path: str,
    start: float,
    end: float,
    gain_db: float = 6.0,
) -> bytes:

    segment = AudioSegment.from_file(audio_path)

    # Pydub uses milliseconds
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)

    # segment = audio[start_ms:end_ms]

    print("Original segment dBFS:", segment.dBFS)

    # Bring the segment close to maximum safe volume
    segment = effects.normalize(segment, headroom=1.0)

    # Optional additional amplification
    segment = segment.apply_gain(gain_db)

    print("Amplified segment dBFS:", segment.dBFS)

    buffer = BytesIO()
    segment.export(buffer, format="wav")

    return buffer.getvalue()
      

if query_text:
    response = requests.get(f"{BASE_URL}/search",
                            params={"query":query_text})

    if response.status_code == 200:
        data = response.json()
        for match in data:
            st.write(f"Matched Transcript: {match['segment_text']}")
            st.write(f"Similariry: {match['similarity']:.2f}")
            audio_file_path = input_folder/match["filename"]
            audio_bytes = prepare_audio_segment(
                audio_path=str(audio_file_path),
                start=float(match["start"]),
                end=float(match["end"]),
                gain_db=6.0,
            )
            st.audio(audio_bytes, format="audio/wav", start_time=match["start"], end_time=match["end"])
            
    else:
        st.write("Search item not found.")