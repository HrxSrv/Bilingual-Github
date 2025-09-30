#!/usr/bin/env python3
"""
Simplified post-commit translation script with essential features only.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from difflib import unified_diff
import re
import hashlib
import json
from datetime import datetime

# Add the src directory to the path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, src_dir)

from utils.markdown_translate import translate_text

TARGET_LANGUAGES = ["en", "ja"]
COMMIT_HASH_FILE = ".translation_commits.json"
INCREMENTAL_THRESHOLD = 20  # Default 20% threshold

def detect_language(content):
    """Simple language detection based on character patterns"""
    if not content or not content.strip():
        return "en"  # Default to English for empty files
    
    # Count Japanese characters (Hiragana, Katakana, Kanji)
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', content))
    total_chars = len(re.sub(r'\s', '', content))
    
    if total_chars == 0:
        return "en"
    
    japanese_ratio = japanese_chars / total_chars
    return "ja" if japanese_ratio > 0.1 else "en"

def get_file_language(file_path):
    """Determine language based on file extension and content"""
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
        # For other .md files, detect language
        if path.exists():
            content = read_file(file_path)
            return detect_language(content)
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
    if (path.name == "README.md" or 
        path.name.endswith('.en.md') or 
        path.name.endswith('.ja.md')):
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
        try:
            os.rename(file_path, new_path)
            return str(new_path)
        except Exception as e:
            print(f"Error renaming {file_path}: {e}")
            return file_path
    
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

def calculate_change_percentage(old_content, new_content):
    """Calculate percentage of lines changed between two versions"""
    if not old_content and not new_content:
        return 0  # Both empty
    
    if not old_content or not new_content:
        return 100  # Treat as complete change if either is missing
    
    # If contents are identical, no change
    if old_content.strip() == new_content.strip():
        return 0
    
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    
    # Use difflib to get actual changes
    diff = list(unified_diff(old_lines, new_lines, lineterm=''))
    
    # Count changed lines (lines starting with + or -)
    changed_lines = len([line for line in diff if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))])
    
    # Calculate percentage based on larger file
    total_lines = max(len(old_lines), len(new_lines))
    if total_lines == 0:
        return 0
    
    percentage = (changed_lines / total_lines) * 100
    print(f"Debug: old_lines={len(old_lines)}, new_lines={len(new_lines)}, changed_lines={changed_lines}, percentage={percentage:.1f}%")
    
    return percentage

def get_previous_version(file_path):
    """Get previous version of file from git history"""
    try:
        # Try to get file from previous commit
        result = subprocess.run(
            ['git', 'show', f'HEAD~1:{file_path}'],
            capture_output=True, text=True, cwd='.'
        )
        if result.returncode == 0:
            return result.stdout
    except Exception as e:
        print(f"Git lookup failed for {file_path}: {e}")
    
    return None

def get_previous_from_history(file_path, commit_history):
    """Get previous version from stored commit history"""
    file_key = str(file_path)
    if file_key in commit_history:
        return commit_history[file_key].get('content')
    return None

def incremental_translate(old_source, new_source, current_target, target_lang):
    """Perform incremental translation using LLM with 3-file context"""
    from utils.markdown_translate import OpenAI
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return None
    
    client = OpenAI(api_key=api_key)
    
    language_names = {
        'en': 'English',
        'ja': 'Japanese'
    }
    
    target_lang_name = language_names.get(target_lang, target_lang)
    
    prompt = f"""You are updating a {target_lang_name} translation. Analyze the changes between OLD_SOURCE and NEW_SOURCE, then update CURRENT_TARGET with minimal changes.

CRITICAL RULES:
1. ONLY modify sections that changed between OLD_SOURCE and NEW_SOURCE
2. Preserve unchanged translations EXACTLY as they are in CURRENT_TARGET
3. Maintain consistent terminology and style from CURRENT_TARGET
4. Preserve ALL markdown formatting (headers, lists, code blocks, tables, links)
5. Return ONLY the complete updated target file content, no explanations

OLD_SOURCE:
{old_source}

NEW_SOURCE:
{new_source}

CURRENT_TARGET:
{current_target}

Updated {target_lang_name} translation:"""

    try:
        print(f"Performing incremental translation to {target_lang_name}")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator specializing in incremental updates. You update {target_lang_name} translations by making minimal changes based on source file differences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
        )
        
        translated_content = response.choices[0].message.content.strip()
        
        if translated_content and len(translated_content) > 10:
            return translated_content
        else:
            print("Incremental translation returned insufficient content")
            return None
            
    except Exception as e:
        print(f"Error in incremental translation: {e}")
        return None

def smart_translate(old_source, new_source, current_target, target_lang, threshold=INCREMENTAL_THRESHOLD):
    """Hybrid translation with configurable threshold"""
    
    change_percentage = calculate_change_percentage(old_source, new_source)
    print(f"Change percentage: {change_percentage:.1f}%")
    
    if change_percentage >= threshold:
        # Large changes - use full translation
        print(f"Large changes detected ({change_percentage:.1f}% >= {threshold}%), using full translation")
        return translate_text(new_source, target_lang)
    else:
        # Small changes - use incremental translation
        print(f"Small changes detected ({change_percentage:.1f}% < {threshold}%), using incremental translation")
        result = incremental_translate(old_source, new_source, current_target, target_lang)
        
        # Fallback to full translation if incremental fails
        if not result:
            print("Incremental translation failed, falling back to full translation")
            return translate_text(new_source, target_lang)
        
        return result

def read_file(file_path):
    """Read file with simple encoding fallback"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8-sig") as file:
            return file.read()

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
        except Exception as e:
            print(f"Error loading commit history: {e}")
            return {}
    return {}

def save_commit_history(history):
    """Save commit hash history"""
    try:
        with open(COMMIT_HASH_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving commit history: {e}")

def sync_translations(original_file, commit_history, current_commit_hash, use_hybrid=True):
    """Sync translations with hybrid translation strategy"""
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
    
    # Check if file was already processed with this hash
    if (file_key in commit_history and 
        commit_history[file_key].get('hash') == current_hash and
        commit_history[file_key].get('commit') == current_commit_hash):
        print(f"File {processed_file} already processed with current hash, skipping")
        return False
    
    # Current source content
    new_source = read_file(processed_file)
    if not new_source.strip():
        print(f"File {processed_file} is empty, skipping")
        return False
    
    # Get previous version for hybrid translation
    old_source = None
    if use_hybrid:
        old_source = get_previous_version(processed_file)
        if not old_source:
            old_source = get_previous_from_history(processed_file, commit_history)
        
        # Skip files that haven't actually changed
        if old_source and old_source.strip() == new_source.strip():
            print(f"File {processed_file} has no actual changes, skipping translation")
            return False
        elif old_source:
            print(f"File {processed_file} has changes detected (old vs new content differs)")
        else:
            print(f"File {processed_file} - no previous version found, treating as new/changed")
            
    target_langs = [lang for lang in TARGET_LANGUAGES if lang != source_lang]
    
    translated = False
    for lang in target_langs:
        translated_file = get_translated_path(processed_file, lang)
        translated_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Translating {processed_file} ({source_lang}) → {translated_file} ({lang})")
        
        try:
            # Decide translation strategy
            if use_hybrid and old_source:
                # Get current translation if it exists
                current_target = None
                if translated_file.exists():
                    current_target = read_file(str(translated_file))
                
                if current_target:
                    # We have all 3 files - use hybrid approach
                    print(f"Using hybrid translation strategy")
                    translated_content = smart_translate(
                        old_source, new_source, current_target, lang
                    )
                else:
                    # No existing translation - fall back to full translation
                    print("No existing translation found, using full translation")
                    translated_content = translate_text(new_source, lang)
            else:
                # No hybrid or no previous version - use full translation
                print("Using full translation")
                translated_content = translate_text(new_source, lang)
            
            if translated_content and translated_content.strip():
                # Write translated content
                translated_file.write_text(translated_content, encoding='utf-8')
                translated = True
                
                # Update commit history for translated file
                translated_key = str(translated_file)
                commit_history[translated_key] = {
                    'hash': hashlib.md5(translated_content.encode()).hexdigest(),
                    'commit': current_commit_hash,
                    'timestamp': datetime.now().isoformat(),
                    'source_file': file_key,
                    'source_lang': source_lang,
                    'target_lang': lang
                }
                print(f"Successfully translated to {translated_file}")
            else:
                print(f"Translation failed or returned empty content for {lang}")
                
        except Exception as e:
            print(f"Error translating {processed_file} to {lang}: {e}")
            continue
    
    # Update commit history for source file (store content for next comparison)
    if translated:
        commit_history[file_key] = {
            'hash': current_hash,
            'commit': current_commit_hash,
            'timestamp': datetime.now().isoformat(),
            'language': source_lang,
            'content': new_source  # Store content for next hybrid translation
        }
    
    return translated

def find_markdown_files():
    """Find all markdown files in project, avoiding translation loops"""
    all_markdown_files = []
    
    # Recursively find all .md files from project root
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.relpath(os.path.join(root, file), '.')
                all_markdown_files.append(file_path)
    
    # Filter out translation files when their source exists to prevent loops
    filtered_files = []
    file_set = set(all_markdown_files)
    
    for file_path in all_markdown_files:
        path = Path(file_path)
        
        # Skip .ja.md files if corresponding .en.md exists
        if path.name.endswith('.ja.md'):
            stem = path.name[:-6]  # Remove .ja.md
            en_counterpart = str(path.parent / f"{stem}.en.md")
            
            if en_counterpart in file_set:
                print(f"Skipping {file_path} - corresponding English source {en_counterpart} exists")
                continue
        
        filtered_files.append(file_path)
    
    return filtered_files

def process_specific_files(file_list, commit_history, current_commit_hash, use_hybrid=True):
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
            if sync_translations(file, commit_history, current_commit_hash, use_hybrid):
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
            # Delete both README.ja.md and README.en.md
            for lang in TARGET_LANGUAGES:
                readme_lang = path.parent / f"README.{lang}.md"
                if readme_lang.exists():
                    print(f"Deleting translated file: {readme_lang}")
                    try:
                        os.remove(readme_lang)
                    except Exception as e:
                        print(f"Error deleting {readme_lang}: {e}")
            continue
        
        source_lang = get_file_language(file)
        if not source_lang:
            continue
            
        # Delete corresponding translation
        target_lang = "ja" if source_lang == "en" else "en"
        translated_path = get_translated_path(file, target_lang)
        
        if translated_path.exists():
            print(f"Deleting translated file: {translated_path}")
            try:
                os.remove(translated_path)
            except Exception as e:
                print(f"Error deleting {translated_path}: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Simplified markdown translation with hybrid support'
    )
    parser.add_argument(
        '--initial-setup', 
        action='store_true', 
        help='Perform initial setup translation'
    )
    parser.add_argument(
        '--files', 
        type=str, 
        help='Comma-separated list of files to translate'
    )
    parser.add_argument(
        '--deleted-files', 
        type=str, 
        help='Comma-separated list of deleted files'
    )
    parser.add_argument(
        '--commit-hash', 
        type=str, 
        help='Current commit hash', 
        default='unknown'
    )
    parser.add_argument(
        '--client-repo', 
        type=str, 
        help='Client repository name (owner/repo)',
        default=''
    )
    parser.add_argument(
        '--pr-number', 
        type=str, 
        help='Pull request number',
        default=''
    )
    parser.add_argument(
        '--no-hybrid', 
        action='store_true', 
        help='Disable hybrid translation, always use full translation'
    )
    
    args = parser.parse_args()

    print("=== Simplified Markdown Translation Starting ===")
    print(f"Client Repository: {args.client_repo}")
    print(f"PR Number: {args.pr_number}")
    print(f"Commit Hash: {args.commit_hash}")
    print(f"Hybrid Translation: {'Disabled' if args.no_hybrid else 'Enabled'}")
    
    # Load commit history
    commit_history = load_commit_history()
    current_commit_hash = args.commit_hash
    use_hybrid = not args.no_hybrid

    # Handle deleted files first
    if args.deleted_files:
        print(f"Processing deleted files: {args.deleted_files}")
        delete_translated_files(args.deleted_files)

    # Process translations
    translation_count = 0
    
    try:
        if args.initial_setup:
            print("Performing initial setup translation")
            markdown_files = find_markdown_files()
            if not markdown_files:
                print("No markdown files found to translate")
            else:
                print(f"Found {len(markdown_files)} markdown files to process")
                for file in markdown_files:
                    if sync_translations(file, commit_history, current_commit_hash, use_hybrid):
                        translation_count += 1
                        
        elif args.files:
            print(f"Processing specific files: {args.files}")
            processed_files = process_specific_files(
                args.files, commit_history, current_commit_hash, use_hybrid
            )
            translation_count = len(processed_files)
            
        else:
            print("Processing all markdown files")
            markdown_files = find_markdown_files()
            if not markdown_files:
                print("No markdown files found to translate")
            else:
                print(f"Found {len(markdown_files)} markdown files to process")
                for file in markdown_files:
                    if sync_translations(file, commit_history, current_commit_hash, use_hybrid):
                        translation_count += 1

        # Save updated commit history
        save_commit_history(commit_history)
        
        print(f"=== Translation Complete: {translation_count} files processed ===")
        
    except Exception as e:
        print(f"Error during translation process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()