import sys
import json
import librosa
import soundfile as sf
import os
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

def main():
    try:
        audio_filepath = sys.argv[1]
        start_time = float(sys.argv[2])
        duration = float(sys.argv[3])

        pipe = pipeline("audio-classification", model="dima806/music_genres_classification")
        
        y, sr = librosa.load(audio_filepath, sr=16000, offset=start_time, duration=duration)
        temp_path = f"temp_{os.getpid()}.wav"
        sf.write(temp_path, y, sr)
        
        results = pipe(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        output_dict = {res['label']: res['score'] for res in results}
        
        print(json.dumps(output_dict))
        
    except Exception as e:
        print(f'{{"error": "{str(e)}"}}', file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()