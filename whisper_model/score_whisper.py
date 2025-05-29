
import os
import whisper
import base64
import json
from tempfile import NamedTemporaryFile

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'base.pt')
    model = whisper.load_model(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        audio_data = base64.b64decode(data["audio"])
        
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            audio_path = tmp.name
        
        result = model.transcribe(audio_path)
        os.unlink(audio_path)
        return {"transcription": result["text"]}
    except Exception as e:
        return {"error": str(e)}