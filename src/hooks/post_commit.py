import os
import sys
import argparse
from pathlib import Path
from difflib import unified_diff
import re
import hashlib
import json
from datetime import datetime, timedelta

script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'src'))
sys.path.insert(0, src_dir)

from utils.translation import translate_text

TARGET_LANGUAGES = ["en", "ja"]
COMMIT_HASH_FILE = ".translation_commits.json"

def detect_language(content):
    """Simple language detection based on character patterns"""
    # Count Japanese characters (Hiragana, Katakana, Kanji)
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', content))
    total_chars = len(re.sub(r'\s', '', content))
    
    if total_chars == 0:
        return "en"  # Default to English for empty files
    
    japanese_ratio = japanese_chars / total_chars
    return "ja" if japanese_ratio > 0.1 else "en"

def get_file_language(file_path):
    """Determine language based on file extension convention"""
    path = Path(file_path)
    
    # Special case for README.md - detect language from content
    if path.name == "README.md":
        if path.exists():
            content = read_file(file_path)
            return detect_language(content)
        return "en"  # Default README to English
    
    # Check for explicit language extensions
    if path.name.endswith('.ja.md'):
        return "ja"
    elif path.name.endswith('.en.md'):
        return "en"
    elif path.name.endswith('.md'):
        # For other .md files, detect language and rename
        if path.exists():
            content = read_file(file_path)
            detected_lang = detect_language(content)
            return detected_lang
        return "en"  # Default to English
    else:
        return None

def get_translated_path(original_path, target_lang):
    """Generate translated file path maintaining directory structure"""
    path = Path(original_path)
    
    # Special case for README.md
    if path.name == "README.md":
        if target_lang == "ja":
            return path.parent / "README.ja.md"
        else:
            return path.parent / "README.en.md"
    
    # Handle other files
    if target_lang == "ja":
        if path.name.endswith('.en.md'):
            # Convert filename.en.md to filename.ja.md
            stem = path.name[:-6]  # Remove .en.md
            return path.parent / f"{stem}.ja.md"
        elif path.name.endswith('.md'):
            # Convert filename.md to filename.ja.md
            stem = path.stem
            return path.parent / f"{stem}.ja.md"
    else:  # target_lang == "en"
        if path.name.endswith('.ja.md'):
            # Convert filename.ja.md to filename.en.md
            stem = path.name[:-6]  # Remove .ja.md
            return path.parent / f"{stem}.en.md"
        elif path.name.endswith('.md'):
            # Convert filename.md to filename.en.md
            stem = path.stem
            return path.parent / f"{stem}.en.md"
    
    return path

def rename_ambiguous_md_file(file_path):
    """Rename .md file to .en.md or .ja.md based on detected language"""
    path = Path(file_path)
    
    # Skip README.md and already explicit files
    if (path.name.upper() == "README.MD" or path.name.endswith('.en.md') or path.name.endswith('.ja.md')):
        return file_path
    
    if path.name.endswith('.md'):
        content = read_file(file_path)
        detected_lang = detect_language(content)
        
        # Create new name with explicit language
        stem = path.stem
        new_name = f"{stem}.{detected_lang}.md"
        new_path = path.parent / new_name
        
        # Rename the file
        print(f"Renaming {file_path} to {new_path} (detected: {detected_lang})")
        os.rename(file_path, new_path)
        return str(new_path)
    
    return file_path

def check_simultaneous_edits(changed_files):
    """Check if both language pairs were edited in the same changeset"""
    files_set = set(changed_files)
    skip_files = set()
    
    for file_path in changed_files:
        if file_path in skip_files:
            continue
            
        path = Path(file_path)
        
        # Special case for README.md
        if path.name == "README.md":
            readme_ja = path.parent / "README.ja.md"
            if str(readme_ja) in files_set:
                print(f"Simultaneous edit detected: {file_path} and {readme_ja}")
                skip_files.add(file_path)
                skip_files.add(str(readme_ja))
            continue
        
        # Check for paired files
        if path.name.endswith('.en.md'):
            stem = path.name[:-6]  # Remove .en.md
            pair_path = path.parent / f"{stem}.ja.md"
        elif path.name.endswith('.ja.md'):
            stem = path.name[:-6]  # Remove .ja.md
            pair_path = path.parent / f"{stem}.en.md"
        else:
            continue
            
        if str(pair_path) in files_set:
            print(f"Simultaneous edit detected: {file_path} and {pair_path}")
            skip_files.add(file_path)
            skip_files.add(str(pair_path))
    
    return skip_files

def get_file_hash(file_path):
    """Get hash of file content for change detection"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return None

def load_commit_history():
    """Load commit hash history"""
    if os.path.exists(COMMIT_HASH_FILE):
        try:
            with open(COMMIT_HASH_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_commit_history(history):
    """Save commit hash history"""
    with open(COMMIT_HASH_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def read_file(file_path):
    """Read file with encoding fallback"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8-sig") as file:
            return file.read()

def sync_translations(original_file, commit_history, current_commit_hash):
    """Sync translations with commit tracking"""
    if not os.path.exists(original_file):
        print(f"File {original_file} not found, skipping")
        return False
    
    # First, handle .md file renaming if needed
    processed_file = rename_ambiguous_md_file(original_file)
    
    source_lang = get_file_language(processed_file)
    if not source_lang:
        print(f"Cannot determine language for {processed_file}, skipping")
        return False
    
    # Get file hash for change detection
    current_hash = get_file_hash(processed_file)
    file_key = str(processed_file)
    
    # Skip hash checking for PR events - always translate on PR changes
    is_pr_event = os.environ.get('GITHUB_EVENT_NAME') == 'pull_request'
    
    # Check if file was already processed with this hash (only for non-PR events)
    if (not is_pr_event and 
        file_key in commit_history and 
        commit_history[file_key].get('hash') == current_hash and
        commit_history[file_key].get('commit') == current_commit_hash):
        print(f"File {processed_file} already processed with current hash, skipping")
        return False
    
    content = read_file(processed_file)
    target_langs = [lang for lang in TARGET_LANGUAGES if lang != source_lang]
    
    translated = False
    for lang in target_langs:
        translated_file = get_translated_path(original_file, lang)
        translated_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Always translate if source file changed
        print(f"Translating {original_file} ({source_lang}) → {translated_file} ({lang})")
        translated_content = translate_text(content, lang)
        
        if translated_content:
            translated_file.write_text(translated_content, encoding='utf-8')
            translated = True
            
            # Update commit history for translated file
            translated_key = str(translated_file)
            commit_history[translated_key] = {
                'hash': hashlib.md5(translated_content.encode()).hexdigest(),
                'commit': current_commit_hash,
                'timestamp': datetime.now().isoformat(),
                'source_file': file_key
            }
    
    # Update commit history for source file
    if translated:
        commit_history[file_key] = {
            'hash': current_hash,
            'commit': current_commit_hash,
            'timestamp': datetime.now().isoformat()
        }
    
    return translated

def find_markdown_files():
    """Find all markdown files in project"""
    markdown_files = []
    
    # Recursively find all .md, .en.md, and .ja.md files from project root
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.md'):  # Includes .en.md and .ja.md files
                file_path = os.path.join(root, file)
                # Skip hidden directories (.git, .github, etc.)
                if not any(part.startswith('.') for part in Path(file_path).parts):
                    markdown_files.append(file_path)
    
    return markdown_files

def process_specific_files(file_list, commit_history, current_commit_hash):
    """Process specific files with simultaneous edit detection"""
    if not file_list:
        return []
    
    files = [f.strip() for f in file_list.split(',') if f.strip().endswith('.md')]
    
    # Check for simultaneous edits
    skip_files = check_simultaneous_edits(files)
    
    if skip_files:
        print(f"Skipping translation for simultaneously edited files: {skip_files}")
    
    # Process files that weren't simultaneously edited
    processed = []
    for file in files:
        if file in skip_files:
            print(f"Skipping {file} due to simultaneous edit")
            continue
            
        if os.path.exists(file):
            print(f"Processing specific file: {file}")
            if sync_translations(file, commit_history, current_commit_hash):
                processed.append(file)
        else:
            print(f"File not found: {file}")
    
    return processed

def delete_translated_files(deleted_files):
    """Delete corresponding translated files"""
    if not deleted_files:
        return
    
    files = [f.strip() for f in deleted_files.split(',') if f.strip().endswith('.md')]
    
    for file in files:
        path = Path(file)
        
        # Special case for README.md
        if path.name == "README.md":
            # Delete README.ja.md if it exists
            readme_ja = path.parent / "README.ja.md"
            if readme_ja.exists():
                print(f"Deleting translated file: {readme_ja}")
                os.remove(readme_ja)
            continue
        
        source_lang = get_file_language(file)
        if not source_lang:
            continue
            
        # Delete corresponding translation
        target_lang = "ja" if source_lang == "en" else "en"
        translated_path = get_translated_path(file, target_lang)
        
        if translated_path.exists():
            print(f"Deleting translated file: {translated_path}")
            os.remove(translated_path)

def main():
    parser = argparse.ArgumentParser(description='Translate markdown files with enhanced tracking')
    parser.add_argument('--initial-setup', action='store_true', help='Perform initial setup translation')
    parser.add_argument('--files', type=str, help='Comma-separated list of files to translate')
    parser.add_argument('--deleted-files', type=str, help='Comma-separated list of deleted files')
    parser.add_argument('--commit-hash', type=str, help='Current commit hash', default='unknown')
    args = parser.parse_args()

    # Load commit history
    commit_history = load_commit_history()
    current_commit_hash = args.commit_hash

    # Handle deleted files first
    if args.deleted_files:
        print(f"Deleting translated files for: {args.deleted_files}")
        delete_translated_files(args.deleted_files)

    # Process translations
    if args.initial_setup:
        print("Performing initial setup translation")
        markdown_files = find_markdown_files()
        if not markdown_files:
            print("No markdown files found to translate")
            return
        
        print(f"Found {len(markdown_files)} markdown files to process")
        for file in markdown_files:
            sync_translations(file, commit_history, current_commit_hash)
            
    elif args.files:
        print(f"Processing specific files: {args.files}")
        process_specific_files(args.files, commit_history, current_commit_hash)
    else:
        markdown_files = find_markdown_files()
        if not markdown_files:
            print("No markdown files found to translate")
            return
            
        print(f"Found {len(markdown_files)} markdown files to process")
        for file in markdown_files:
            sync_translations(file, commit_history, current_commit_hash)

    # Save updated commit history
    save_commit_history(commit_history)

if __name__ == "__main__":
    main()