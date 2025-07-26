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

def get_file_language(file_path):
    """Determine language based on file extension convention"""
    path = Path(file_path)
    
    # Check for .ja.md extension
    if path.name.endswith('.ja.md'):
        return "ja"
    # All other .md files are English
    elif path.name.endswith('.md'):
        return "en"
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
            return path.parent / "README.md"
    
    # For other files, maintain directory structure
    if target_lang == "ja":
        # Convert filename.md to filename.ja.md
        stem = path.stem
        return path.parent / f"{stem}.ja.md"
    else:
        # Convert filename.ja.md to filename.md
        if path.name.endswith('.ja.md'):
            stem = path.name[:-6]  # Remove .ja.md
            return path.parent / f"{stem}.md"
        else:
            return path

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

# def check_simultaneous_updates(files, commit_history):
#     """Check if both language versions were updated recently"""
#     simultaneous_pairs = []
#     current_time = datetime.now()
#     
#     for file_path in files:
#         source_lang = get_file_language(file_path)
#         if not source_lang:
#             continue
#             
#         # Get the corresponding file in other language
#         other_lang = "ja" if source_lang == "en" else "en"
#         other_file = get_translated_path(file_path, other_lang)
#         
#         if not other_file.exists():
#             continue
#             
#         # Check if both files were modified within 100 seconds
#         file_time = commit_history.get(str(file_path), {}).get('timestamp')
#         other_time = commit_history.get(str(other_file), {}).get('timestamp')
#         
#         if file_time and other_time:
#             file_dt = datetime.fromisoformat(file_time)
#             other_dt = datetime.fromisoformat(other_time)
#             
#             if abs((file_dt - other_dt).total_seconds()) <= 100:
#                 simultaneous_pairs.append((file_path, str(other_file)))
#     
#     return simultaneous_pairs

# def create_conflict_pr_message(simultaneous_pairs):
#     """Create PR message for simultaneous updates"""
#     message = "# Translation Conflict Detected\n\n"
#     message += "Both language files were modified together. Please review and merge manually with appropriate tags.\n\n"
#     message += "## Affected Files:\n"
#     
#     for en_file, ja_file in simultaneous_pairs:
#         message += f"- {en_file} ↔ {ja_file}\n"
#     
#     message += "\n⚠️ **Manual review required to prevent overwrites**"
#     return message

def sync_translations(original_file, commit_history, current_commit_hash):
    """Sync translations with commit tracking"""
    if not os.path.exists(original_file):
        print(f"File {original_file} not found, skipping")
        return False
    
    source_lang = get_file_language(original_file)
    if not source_lang:
        print(f"Cannot determine language for {original_file}, skipping")
        return False
    
    # Get file hash for change detection
    current_hash = get_file_hash(original_file)
    file_key = str(original_file)
    
    # Check if file was already processed with this hash
    if (file_key in commit_history and 
        commit_history[file_key].get('hash') == current_hash and
        commit_history[file_key].get('commit') == current_commit_hash):
        print(f"File {original_file} already processed with current hash, skipping")
        return False
    
    content = read_file(original_file)
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
    
    # Add README.md if it exists
    if os.path.exists('README.md'):
        markdown_files.append('README.md')
    
    # Add README.ja.md if it exists
    if os.path.exists('README.ja.md'):
        markdown_files.append('README.ja.md')
    
    # Add all .md files from docs directory
    if os.path.exists('docs'):
        for root, _, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    markdown_files.append(os.path.join(root, file))
    
    return markdown_files

def process_specific_files(file_list, commit_history, current_commit_hash):
    """Process specific files"""
    if not file_list:
        return []
    
    files = [f.strip() for f in file_list.split(',') if f.strip().endswith('.md')]
    
    # # Check for simultaneous updates
    # simultaneous_pairs = check_simultaneous_updates(files, commit_history)
    # 
    # if simultaneous_pairs:
    #     print("⚠️ Simultaneous language updates detected!")
    #     print(create_conflict_pr_message(simultaneous_pairs))
    #     # Return conflict info instead of processing
    #     return simultaneous_pairs
    
    # Process files normally
    processed = []
    for file in files:
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
        result = process_specific_files(args.files, commit_history, current_commit_hash)
        
        # # Check if result contains simultaneous update conflicts
        # if result and isinstance(result[0], tuple):
        #     print("Creating PR for manual review due to simultaneous updates")
        #     # Exit with special code to signal workflow to create PR
        #     sys.exit(2)
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