import sys
import os
from github import Github
import re

script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'src'))
sys.path.append(src_dir)

from utils.translation import translate_text

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
REPO_NAME = os.getenv("GITHUB_REPOSITORY", "").strip()
TRANSLATED_LABEL = "translated"
NEEDS_TRANSLATION_LABEL = "need translation"
PR_NUMBER = os.getenv("PR_NUMBER", "").strip()
COMMENT_ID = os.getenv("COMMENT_ID", "").strip()
ORIGINAL_CONTENT_MARKER = "Original Content:"
EVENT_NAME = os.getenv("GITHUB_EVENT_NAME", "").strip()

LANGUAGE_NAMES = {
    "ja": "日本語",
    "en": "English"
}

def get_original_content(content):
    if ORIGINAL_CONTENT_MARKER in content:
        parts = content.split(ORIGINAL_CONTENT_MARKER, 1)
        original = parts[1].lstrip()
        original = re.sub(r'^(<br>\s*)+', '', original)
        return original.strip()
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
            formatted_parts.append(f"## {translation}")

    for language, translation in body_translations.items():
        if translation and language != original_language:
            language_name = LANGUAGE_NAMES.get(language, language.capitalize())
            formatted_parts.append(
                f"<details>\n<summary><b>{language_name}</b></summary>\n\n{translation}\n</details><br>"
            )

    original_lang_name = LANGUAGE_NAMES.get(original_language, original_language.capitalize())
    formatted_parts.append(f"<b>{ORIGINAL_CONTENT_MARKER}</b>\n\n{original_content}")

    return "\n\n".join(formatted_parts).strip()

def translate_content(content, original_language):
    translations = {original_language: content}
    target_languages = get_target_languages(original_language)
    
    for language in target_languages:
        translation = translate_text(content, language)
        if translation:
            translations[language] = translation
    
    return translations

def split_quoted_and_reply_content(comment_body):
    """
    Splits the comment into quoted content (lines starting with '>') and reply content (all other lines).
    Preserves the original structure including empty lines between quoted and reply sections.
    Returns (quoted_content, reply_content) as strings.
    """
    lines = comment_body.splitlines()
    
    # Find the last quoted line to determine where quoted content ends
    last_quoted_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('>'):
            last_quoted_index = i
    
    if last_quoted_index == -1:
        # No quoted content found
        return "", comment_body.strip()
    
    # Extract quoted content (including any empty lines within the quoted block)
    quoted_lines = []
    for i in range(last_quoted_index + 1):
        quoted_lines.append(lines[i])
    
    # Extract reply content (everything after the quoted block, excluding leading empty lines)
    reply_lines = lines[last_quoted_index + 1:]
    # Remove leading empty lines from reply
    while reply_lines and not reply_lines[0].strip():
        reply_lines.pop(0)
    
    quoted_content = '\n'.join(quoted_lines).strip()
    reply_content = '\n'.join(reply_lines).strip()
    
    return quoted_content, reply_content

def translate_pr(pr, original_content, original_language, pr_title, pr_body):
    title_translations = translate_content(pr_title, original_language)
    body_translations = translate_content(pr_body, original_language)
    updated_body = format_translations(title_translations, body_translations, original_content, original_language)
    
    if updated_body != pr.body:
        pr.edit(body=updated_body)
        return True
    
    return False

def translate_pr_comment(comment):
    if not comment.body:
        return False
    
    current_content = comment.body.strip()
    
    # Check if this is already a translated comment
    if ORIGINAL_CONTENT_MARKER in current_content:
        # Extract the original content which includes quoted content with formatting
        original_full_content = get_original_content(current_content)
        quoted_content, reply_content = split_quoted_and_reply_content(original_full_content)
    else:
        # This is a fresh comment
        quoted_content, reply_content = split_quoted_and_reply_content(current_content)
        if not reply_content:
            reply_content = current_content
    
    original_language = detect_language(reply_content)
    translations = translate_content(reply_content, original_language)
    
    if translations:
        # When there's quoted content, we need to include it in the original content too
        if quoted_content:
            original_content_with_quote = f"{quoted_content}\n\n{reply_content}"
        else:
            original_content_with_quote = reply_content
            
        translated_reply = format_translations({}, translations, original_content_with_quote, original_language)
        if quoted_content:
            updated_body = f"{quoted_content}\n\n{translated_reply}"
        else:
            updated_body = translated_reply
        if updated_body != comment.body:
            comment.edit(body=updated_body)
            return True
    
    return False

def should_translate(pr):
    labels = [label.name.lower() for label in pr.labels]
    
    if NEEDS_TRANSLATION_LABEL.lower() in labels:
        return True
    
    if TRANSLATED_LABEL.lower() in labels and NEEDS_TRANSLATION_LABEL.lower() in labels:
        return True
        
    return False

def main():
    if not all([GITHUB_TOKEN, REPO_NAME, PR_NUMBER]):
        print("Missing required environment variables")
        return
    
    try:
        pr_number = int(PR_NUMBER)
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        pr = repo.get_pull(number=pr_number)
        
        if not should_translate(pr):
            print(f"PR #{pr_number} does not require translation at this time.")
            return
        
        if COMMENT_ID:
            comment_id = int(COMMENT_ID)
            # Check if it's a review comment or an issue comment
            if EVENT_NAME == 'pull_request_review_comment':
                comment = pr.get_review_comment(comment_id)
            else:
                comment = pr.get_issue_comment(comment_id)
            if translate_pr_comment(comment):
                labels = [label.name.lower() for label in pr.labels]
                if TRANSLATED_LABEL.lower() not in labels:
                    pr.add_to_labels(TRANSLATED_LABEL)
            return
        
        # Otherwise translate the PR body
        original_content = get_original_content(pr.body)
        original_language = detect_language(original_content)
        pr_title = pr.title
        pr_body = get_original_content(pr.body)
        
        pr_translated = translate_pr(pr, original_content, original_language, pr_title, pr_body)

        # Translate all comments on the PR
        comments = pr.get_issue_comments()
        comments_translated = False
        for comment in comments:
            if translate_pr_comment(comment):
                comments_translated = True

        # Translate all review comments on the PR
        review_comments = pr.get_review_comments()
        for comment in review_comments:
            if translate_pr_comment(comment):
                comments_translated = True

        if pr_translated or comments_translated:
            labels = [label.name.lower() for label in pr.labels]
            if TRANSLATED_LABEL.lower() not in labels:
                pr.add_to_labels(TRANSLATED_LABEL)
    
    except ValueError as ve:
        print(f"Invalid number format: {ve}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main() 