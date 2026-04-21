import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ✅ THIS MUST EXIST
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(texts, batch_size=50):
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        all_embeddings.extend([r.embedding for r in response.data])

        print(f"Processed batch {i//batch_size + 1}")

    return all_embeddings