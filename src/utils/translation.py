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


def translate_incremental(base_content, current_content, existing_translation, target_lang):
    """
    Translate only the changed portions using GPT with three-file context.
    
    Args:
        base_content: The previous version of the source document
        current_content: The updated version of the source document (with changes)
        existing_translation: The current translation of the base version
        target_lang: Target language for translation
    
    Returns:
        Updated translation with only the necessary changes applied, or None on failure
    """
    prompt = f"""You are translating a markdown document to {target_lang}.

I will provide you with three versions of the document:
1. BASE VERSION: The previous version of the source document
2. CURRENT VERSION: The updated version of the source document (with changes)
3. EXISTING TRANSLATION: The current translation of the base version

Your task:
- Identify what changed between BASE and CURRENT versions
- Update ONLY the changed portions in the EXISTING TRANSLATION
- Preserve all unchanged parts of the EXISTING TRANSLATION exactly as they are
- Maintain the same markdown structure and formatting

BASE VERSION:
---
{base_content}
---

CURRENT VERSION:
---
{current_content}
---

EXISTING TRANSLATION:
---
{existing_translation}
---

Please provide the updated translation with only the necessary changes applied."""

    try:
        url = "https://api.openai.com/v1/chat/completions"
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": f"You are a precise translator. Translate only the changed portions to {target_lang}."},
                {"role": "user", "content": prompt}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '').strip()}"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            translation = result["choices"][0]["message"]["content"]
            return translation
        else:
            print(f"Incremental translation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error in incremental translation: {e}")
        return None
