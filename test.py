import os
import json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------
# CHARGER LA CLÉ API DEPUIS .env
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Clé API OpenRouter manquante. Ajoutez-la dans votre fichier .env")

# -------------------------
# CLIENT OPENROUTER
# -------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# -------------------------
# FONCTION D'ENVOI D'UNE IMAGE À OPENROUTER
# -------------------------
def send_url_image(prompt, image_url):
    """
    Envoie un prompt et une image à OpenRouter,
    puis retourne la réponse JSON.
    """
    try:
        completion = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )

        response_text = completion.choices[0].message.content

        return {
            "image_url": image_url,
            "status": "success",
            "data": response_text
        }

    except Exception as e:
        return {
            "image_url": image_url,
            "status": "error",
            "error": str(e)
        }

# -------------------------
# MULTI-TRAITEMENT
# -------------------------
def main():
    image_urls = [
        "https://i.ibb.co/BHLsKH83/photo-2025-09-19-19-53-53.jpg",
        "https://i.ibb.co/Z1vMYsvp/image-Test1.jpg"
    ]
    prompt = "Analyse cette image et retourne les informations textuelles visibles sous forme de texte clair."

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda url: send_url_image(prompt, url), image_urls))

    print(json.dumps(results, indent=2))

# -------------------------
# POINT D'ENTRÉE
# -------------------------
if __name__ == "__main__":
    main()
