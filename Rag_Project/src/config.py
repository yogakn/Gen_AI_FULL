from dotenv import load_dotenv
import os
from openai import OpenAI

# Load from parent folder (.env)
load_dotenv(dotenv_path="../.env", override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found")

client = OpenAI(api_key=api_key)