import os
import sys
import argparse
from pathlib import Path
from difflib import unified_diff
import re

script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'src'))
sys.path.insert(0, src_dir)

from utils.translation import translate_text

TARGET_LANGUAGES = ["en", "ja"]

def detect_source_language(text):
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

def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8-sig") as file:
            return file.read()

def get_translated_path(original_path, lang):
    path = Path(original_path)
    if path.name == "README.md":
        return path.parent / f"README.{lang}.md"
    else:
        # Replace 'docs' with 'docs.{lang}' in the path
        parts = path.parts
        try:
            idx = parts.index('docs')
            new_parts = list(parts)
            new_parts[idx] = f'docs.{lang}'
            return Path(*new_parts)
        except ValueError:
            # If 'docs' not in path, just prepend 'docs.{lang}'
            return Path(f'docs.{lang}') / path

def sync_translations(original_file):
    content = read_file(original_file)
    source_lang = detect_source_language(content)
    
    # Only translate if source language is different from target languages
    target_langs = [lang for lang in TARGET_LANGUAGES if lang != source_lang]
    
    if not target_langs:
        print(f"File {original_file} is already in target language(s), skipping translation")
        return
    
    for lang in target_langs:
        translated_file = get_translated_path(original_file, lang)
        translated_file.parent.mkdir(parents=True, exist_ok=True)
        needs_translation = True
        
        if translated_file.exists():
            existing_translation = read_file(translated_file)
            diff = list(unified_diff(
                existing_translation.splitlines(),
                content.splitlines(),
                lineterm=""
            ))
            needs_translation = bool(diff)
            
        if needs_translation:
            print(f"Translating {original_file} from {source_lang} to {lang}")
            translated_content = translate_text(content, lang)
            if translated_content:
                translated_file.write_text(translated_content, encoding='utf-8')

def find_markdown_files():
    markdown_files = []
    
    # Add README.md if it exists
    if os.path.exists('README.md'):
        markdown_files.append('README.md')
    
    # Add all .md files from docs directory
    if os.path.exists('docs'):
        for root, _, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    markdown_files.append(os.path.join(root, file))
    
    return markdown_files

def process_specific_files(file_list):
    if not file_list:
        return
    
    files = file_list.split(',')
    for file in files:
        file = file.strip()
        if file.endswith('.md'):
            if os.path.exists(file):
                print(f"Processing specific file: {file}")
                sync_translations(file)
            else:
                print(f"File not found: {file}")

def delete_translated_files(deleted_files):
    if not deleted_files:
        return
    files = deleted_files.split(',')
    for file in files:
        file = file.strip()
        if not file.endswith('.md'):
            continue
        for lang in TARGET_LANGUAGES:
            translated_path = get_translated_path(file, lang)
            if translated_path.exists():
                print(f"Deleting translated file: {translated_path}")
                os.remove(translated_path)

def main():
    parser = argparse.ArgumentParser(description='Translate markdown files')
    parser.add_argument('--initial-setup', action='store_true', help='Perform initial setup translation')
    parser.add_argument('--files', type=str, help='Comma-separated list of files to translate')
    parser.add_argument('--deleted-files', type=str, help='Comma-separated list of deleted files')
    args = parser.parse_args()

    if args.deleted_files:
        print(f"Deleting translated files for: {args.deleted_files}")
        delete_translated_files(args.deleted_files)

    if args.initial_setup:
        print("Performing initial setup translation")
        markdown_files = find_markdown_files()
        if not markdown_files:
            print("No markdown files found to translate")
            return
        print(f"Found {len(markdown_files)} markdown files to process")
        for file in markdown_files:
            sync_translations(file)
    elif args.files:
        print(f"Processing specific files: {args.files}")
        process_specific_files(args.files)
    else:
        markdown_files = find_markdown_files()
        if not markdown_files:
            print("No markdown files found to translate")
            return
        print(f"Found {len(markdown_files)} markdown files to process")
        for file in markdown_files:
            sync_translations(file)

if __name__ == "__main__":
    main()