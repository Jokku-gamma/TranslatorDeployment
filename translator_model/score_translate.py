import json
from transformers import pipeline

translator = None

def init():
    global translator
    translator = pipeline(
        "translation", 
        model="Helsinki-NLP/opus-mt-mul-en"
    )

def run(raw_data):
    try:
        data = json.loads(raw_data)
        text = data["text"]
        result = translator(text)
        return {"translation": result[0]["translation_text"]}
    except Exception as e:
        return {"error": str(e)}