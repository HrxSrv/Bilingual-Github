import os
from dotenv import load_dotenv
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please ensure it is defined in the environment.")

def translate_text(text, target_language):
    try:
        url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": "gpt-4o-mini",  
            "messages": [
                {"role": "system", "content": f"Translate this text to {target_language}."},
                {"role": "user", "content": text}
            ]
        }

        headers = {
           "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '').strip()}"
        }
        if "Bearer" not in headers["Authorization"]:
            print("Error: Authorization header is malformed.")

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            translation = result["choices"][0]["message"]["content"]
            return translation
        else:
            print(f"Failed to connect to OpenAI API. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"Error in translation: {e}")
        return None
