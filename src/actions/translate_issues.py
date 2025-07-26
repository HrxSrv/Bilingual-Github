import sys
import os
import re
from github import Github

script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'src'))
sys.path.append(src_dir)

from utils.translation import translate_text

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
REPO_NAME = os.getenv("GITHUB_REPOSITORY", "").strip()
TRANSLATED_LABEL = "translated"
NEEDS_TRANSLATION_LABEL = "need translation"
ISSUE_NUMBER = os.getenv("ISSUE_NUMBER", "").strip()
ORIGINAL_CONTENT_MARKER = "Original Content:"

LANGUAGE_NAMES = {
    "ja": "日本語",
    "en": "English"
}

def get_original_content(content):
    if ORIGINAL_CONTENT_MARKER in content:
        parts = content.split(ORIGINAL_CONTENT_MARKER)
        return parts[1].strip()
    return content.strip()

def detect_language(text):
    # Unicode ranges for Japanese characters
    HIRAGANA = '\u3040-\u309F'
    KATAKANA = '\u30A0-\u30FF'
    KANJI = '\u4E00-\u9FFF'
    HALF_WIDTH_KATAKANA = '\uFF60-\uFF9F'
    
    # Check if text contains any Japanese character
    jp_pattern = f'[{HIRAGANA}{KATAKANA}{KANJI}{HALF_WIDTH_KATAKANA}]'
    has_japanese = bool(re.search(jp_pattern, text))
    
    # If any Japanese character is found, original is Japanese, translate to English
    # Otherwise, original is English, translate to Japanese
    return "ja" if has_japanese else "en"

def get_target_languages(original_language):
    if original_language == "en":
        return ["ja"]
    elif original_language == "ja":
        return ["en"]
    return ["en"]

def format_translations(title_translations, body_translations, original_content, original_language):
    formatted_parts = []
    
    for language, translation in title_translations.items():
        if translation and language != original_language:
            language_name = LANGUAGE_NAMES.get(language, language.capitalize())
            formatted_parts.append(f"<h2>{translation}</h2>")
    
    for language, translation in body_translations.items():
        if translation and language != original_language:
            language_name = LANGUAGE_NAMES.get(language, language.capitalize())
            formatted_parts.append(f"\n\n<details>\n<summary><b>{language_name}</b></summary>\n\n{translation}</details><br>")
    
    original_lang_name = LANGUAGE_NAMES.get(original_language, original_language.capitalize())
    formatted_parts.append(f"<b>{ORIGINAL_CONTENT_MARKER}</b>\n{original_content}")
    
    return "\n\n".join(formatted_parts)

def translate_content(content, original_language):
    translations = {original_language: content}
    target_languages = get_target_languages(original_language)
    
    for language in target_languages:
        translation = translate_text(content, language)
        if translation:
            translations[language] = translation
    
    return translations

def translate_issue(issue, original_content, original_language, issue_title, issue_body):
    title_translations = translate_content(issue_title, original_language)
    body_translations = translate_content(issue_body, original_language)
    updated_body = format_translations(title_translations, body_translations, original_content, original_language)
    
    if updated_body != issue.body:
        issue.edit(body=updated_body)
        return True
    
    return False

def should_translate(issue):
    labels = [label.name.lower() for label in issue.labels]
    
    if NEEDS_TRANSLATION_LABEL.lower() in labels:
        return True
    
    if TRANSLATED_LABEL.lower() in labels and NEEDS_TRANSLATION_LABEL.lower() in labels:
        return True
        
    return False

def main():
    if not all([GITHUB_TOKEN, REPO_NAME, ISSUE_NUMBER]):
        print("Missing required environment variables")
        return
    
    try:
        issue_number = int(ISSUE_NUMBER)
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        issue = repo.get_issue(number=issue_number)
        
        if not should_translate(issue):
            print(f"Issue #{issue_number} does not require translation at this time.")
            return
        
        original_content = get_original_content(issue.body)
        original_language = detect_language(original_content)
        issue_title = issue.title
        issue_body = get_original_content(issue.body)
        
        if translate_issue(issue, original_content, original_language, issue_title, issue_body):
            labels = [label.name.lower() for label in issue.labels]
            if TRANSLATED_LABEL.lower() not in labels:
                issue.add_to_labels(TRANSLATED_LABEL)
    
    except ValueError as ve:
        print(f"Invalid number format: {ve}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()