"""
Translation utilities using OpenAI API
"""

import os
import openai
import time
from typing import Optional


# Set up OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')


def translate_text(text: str, target_language: str, max_retries: int = 3) -> Optional[str]:
    """
    Translate markdown text to target language using OpenAI API
    
    Args:
        text (str): The markdown text to translate
        target_language (str): Target language code ('en' or 'ja')
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Translated text, or None if translation failed
    """
    if not text or not text.strip():
        return text
    
    if not openai.api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return None
    
    # Language mapping
    language_names = {
        'en': 'English',
        'ja': 'Japanese'
    }
    
    if target_language not in language_names:
        print(f"Error: Unsupported target language '{target_language}'")
        return None
    
    target_lang_name = language_names[target_language]
    
    # Create translation prompt
    prompt = create_translation_prompt(text, target_lang_name)
    
    for attempt in range(max_retries):
        try:
            print(f"Translating to {target_lang_name} (attempt {attempt + 1}/{max_retries})")
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator specializing in technical documentation. Translate the given markdown text to {target_lang_name} while preserving all markdown formatting, code blocks, links, and technical terms."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent translation
                max_tokens=4000,
                timeout=30
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            if translated_text and len(translated_text) > 10:  # Basic validation
                print(f"Translation successful ({len(translated_text)} characters)")
                return translated_text
            else:
                print(f"Translation returned unusually short result: '{translated_text}'")
                if attempt < max_retries - 1:
                    continue
                    
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            continue
            
        except openai.error.APIConnectionError as e:
            print(f"API connection error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
            
        except openai.error.APIError as e:
            print(f"OpenAI API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
            
        except Exception as e:
            print(f"Unexpected error during translation (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            continue
    
    print(f"Translation failed after {max_retries} attempts")
    return None


def create_translation_prompt(text: str, target_language: str) -> str:
    """
    Create a detailed prompt for translation
    
    Args:
        text (str): Text to translate
        target_language (str): Target language name
        
    Returns:
        str: Formatted prompt for the API
    """
    prompt = f"""Please translate the following markdown text to {target_language}.

IMPORTANT INSTRUCTIONS:
1. Preserve ALL markdown formatting including headers, links, code blocks, tables, lists, etc.
2. Do NOT translate content inside code blocks (```code```) or inline code (`code`)
3. Do NOT translate URLs, file paths, or technical identifiers
4. Keep the same structure and organization
5. Maintain the same tone and style appropriate for technical documentation
6. If there are proper names, brand names, or technical terms that are commonly used in their original language, keep them as-is
7. For Japanese translation: Use appropriate levels of formality (polite form recommended for documentation)
8. Return ONLY the translated markdown text, no additional commentary

TEXT TO TRANSLATE:
---
{text}
---

TRANSLATED TEXT:"""
    
    return prompt


def validate_translation(original: str, translated: str, target_lang: str) -> bool:
    """
    Basic validation of translation quality
    
    Args:
        original (str): Original text
        translated (str): Translated text  
        target_lang (str): Target language code
        
    Returns:
        bool: True if translation appears valid
    """
    if not translated or not translated.strip():
        return False
    
    # Check if translation is too short or too long compared to original
    original_len = len(original.strip())
    translated_len = len(translated.strip())
    
    if translated_len < original_len * 0.3 or translated_len > original_len * 3:
        print(f"Warning: Translation length seems unusual (original: {original_len}, translated: {translated_len})")
        return False
    
    # Check if markdown structure is preserved
    original_headers = original.count('#')
    translated_headers = translated.count('#')
    
    if abs(original_headers - translated_headers) > 1:
        print(f"Warning: Header count mismatch (original: {original_headers}, translated: {translated_headers})")
    
    # Check if code blocks are preserved
    original_code_blocks = original.count('```')
    translated_code_blocks = translated.count('```')
    
    if original_code_blocks != translated_code_blocks:
        print(f"Warning: Code block count mismatch (original: {original_code_blocks}, translated: {translated_code_blocks})")
        return False
    
    return True


def get_translation_stats(original: str, translated: str) -> dict:
    """
    Get basic statistics about the translation
    
    Args:
        original (str): Original text
        translated (str): Translated text
        
    Returns:
        dict: Translation statistics
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


# Test function for development
def test_translation():
    """Test the translation function with sample text"""
    sample_text = """# Sample Document

This is a test document with **bold** and *italic* text.

## Code Example

```python
def hello_world():
    print("Hello, World!")
```

- List item 1
- List item 2

[Link to GitHub](https://github.com)
"""
    
    print("Testing English to Japanese translation...")
    result = translate_text(sample_text, 'ja')
    if result:
        print("Translation successful!")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        stats = get_translation_stats(sample_text, result)
        print(f"Stats: {stats}")
    else:
        print("Translation failed!")


if __name__ == "__main__":
    test_translation()