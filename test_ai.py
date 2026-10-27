
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    print("Checking available models...")
    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name)
except Exception as e:
    print("Error:", e)
