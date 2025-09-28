"""
Translation utilities using OpenAI API (v1.x compatible)
"""

import os
from openai import OpenAI
import time
from typing import Optional


def translate_text(text: str, target_language: str, max_retries: int = 3) -> Optional[str]:
    """
    Translate text to target language using OpenAI API
    
    Args:
        text (str): The text to translate
        target_language (str): Target language code ('en' or 'ja')
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Translated text, or None if translation failed
    """
    if not text or not text.strip():
        return text
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return None
    
    client = OpenAI(api_key=api_key)
    
    language_names = {
        'en': 'English',
        'ja': 'Japanese'
    }
    
    if target_language not in language_names:
        print(f"Error: Unsupported target language '{target_language}'")
        return None
    
    target_lang_name = language_names[target_language]
    prompt = text.strip()
    
    for attempt in range(max_retries):
        try:
            print(f"Translating to {target_lang_name} (attempt {attempt + 1}/{max_retries})")
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate the following text into {target_lang_name}."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            if translated_text and len(translated_text) > 5:
                return translated_text
            else:
                if attempt < max_retries - 1:
                    continue
                    
        except Exception as e:
            error_type = type(e).__name__
            print(f"Error ({error_type}) on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 5)
                continue
    
    print(f"Translation failed after {max_retries} attempts")
    return None


def validate_translation(original: str, translated: str, target_lang: str) -> bool:
    """
    Basic validation of translation quality
    """
    if not translated or not translated.strip():
        return False
    
    original_len = len(original.strip())
    translated_len = len(translated.strip())
    
    if translated_len < original_len * 0.3 or translated_len > original_len * 3:
        return False
    
    return True


def get_translation_stats(original: str, translated: str) -> dict:
    """
    Get basic statistics about the translation
    """
    return {
        'original_length': len(original),
        'translated_length': len(translated),
        'original_words': len(original.split()) if original else 0,
        'translated_words': len(translated.split()) if translated else 0,
        'original_lines': len(original.splitlines()) if original else 0,
        'translated_lines': len(translated.splitlines()) if translated else 0,
        'compression_ratio': len(translated) / len(original) if original else 0
    }
