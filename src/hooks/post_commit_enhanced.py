#!/usr/bin/env python3
"""
Enhanced post-commit translation script with integrated formatting and configuration support.
"""

import os
import sys
import argparse
import fnmatch
import subprocess
from pathlib import Path
from difflib import unified_diff
import re
import hashlib
import json
from datetime import datetime, timedelta

# Add the src directory to the path
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, src_dir)

from utils.markdown_translate import translate_text

TARGET_LANGUAGES = ["en", "ja"]
COMMIT_HASH_FILE = ".translation_commits.json"
TRANSLATION_IGNORE_FILE = ".translation_ignore"

DEFAULT_IGNORE_PATTERNS = [
    '.claude/**',
    '.cursor/**', 
    '.cline/**',
    '.vscode/**',
    'CLAUDE.md',
    'GEMINI.md',
    'node_modules/**',
    '.git/**',
    '.github/**'
]

def load_ignore_patterns(repo_root='.'):
    """Load .translation_ignore patterns from client repo"""
    ignore_file = Path(repo_root) / TRANSLATION_IGNORE_FILE
    patterns = []
    
    if ignore_file.exists():
        print(f"Loading ignore patterns from: {ignore_file}")
        try:
            with open(ignore_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
            print(f"Loaded {len(patterns)} ignore patterns")
        except Exception as e:
            print(f"Error reading {ignore_file}: {e}")
            patterns = DEFAULT_IGNORE_PATTERNS.copy()
    else:
        print("No .translation_ignore file found, using default patterns")
        patterns = DEFAULT_IGNORE_PATTERNS.copy()
    
    return patterns

def should_ignore_file(file_path, ignore_patterns):
    """Check if file should be ignored based on patterns"""
    file_path_str = str(file_path)
    
    for pattern in ignore_patterns:
        # Direct match
        if fnmatch.fnmatch(file_path_str, pattern):
            print(f"Ignoring {file_path_str} (matches pattern: {pattern})")
            return True
        
        # Handle directory patterns ending with /**
        if pattern.endswith('/**'):
            dir_pattern = pattern[:-3]  # Remove /**
            if file_path_str.startswith(dir_pattern + '/') or file_path_str == dir_pattern:
                print(f"Ignoring {file_path_str} (in directory: {dir_pattern})")
                return True
        
        # Handle patterns with path separators
        if '/' in pattern and fnmatch.fnmatch(file_path_str, pattern):
            print(f"Ignoring {file_path_str} (matches path pattern: {pattern})")
            return True
    
    return False

def load_threshold(repo_root='.'):
    """Load translation threshold from config"""
    config_file = Path(repo_root) / '.translation_config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                threshold = config.get('incremental_threshold', 20)
                print(f"Loaded incremental threshold: {threshold}%")
                return threshold
        except Exception as e:
            print(f"Error loading config: {e}")
    
    print("Using default incremental threshold: 20%")
    return 20  # Default 20%

def apply_formatting_fixes(file_path):
    """Apply formatting fixes to a markdown file"""
    if not os.path.exists(file_path):
        return False
    
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: Remove trailing whitespace from lines
        lines = content.splitlines()
        fixed_lines = [line.rstrip() for line in lines]
        content = '\n'.join(fixed_lines)
        
        # Fix 2: Ensure single newline at end of file
        if content and not content.endswith('\n'):
            content += '\n'
        elif content.endswith('\n\n'):
            # Remove extra newlines, keep only one
            content = content.rstrip('\n') + '\n'
        
        # Fix 3: Remove excessive blank lines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Applied formatting fixes to: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error formatting {file_path}: {e}")
        return False

def detect_language(content):
    """Enhanced language detection for markdown files"""
    if not content or not content.strip():
        return "en"  # Default to English for empty files
    
    # Remove code blocks and inline code to avoid false positives
    content_no_code = re.sub(r'```[\s\S]*?```', '', content)
    content_no_code = re.sub(r'`[^`]*`', '', content_no_code)
    
    # Count Japanese characters (Hiragana, Katakana, Kanji)
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', content_no_code))
    
    # Count total non-whitespace characters
    total_chars = len(re.sub(r'\s', '', content_no_code))
    
    if total_chars == 0:
        return "en"
    
    japanese_ratio = japanese_chars / total_chars
    
    # More sophisticated detection
    if japanese_ratio > 0.15:  # Higher threshold for more accuracy
        return "ja"
    elif japanese_ratio > 0.05:
        # Check for Japanese-specific patterns
        if re.search(r'[。、]', content_no_code):  # Japanese punctuation
            return "ja"
        if re.search(r'です|ます|である|だ。', content_no_code):  # Japanese sentence endings
            return "ja"
    
    return "en"

def get_file_language(file_path):
    """Determine language based on file extension and content"""
    path = Path(file_path)
    
    # Special case for README.md - detect language from content
    if path.name.upper() == "README.MD":
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
    if path.name.upper() == "README.MD":
        stem = path.name[:-3]  # Remove .md (3 characters)
        if target_lang == "ja":
            return path.parent / f"{stem}.ja.md"
        else:
            return path.parent / f"{stem}.en.md"
    
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
    if (path.name.upper() == "README.MD" or 
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

def load_commit_history(repo_root='.'):
    """Load commit hash history"""
    history_file = Path(repo_root) / COMMIT_HASH_FILE
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading commit history: {e}")
            return {}
    return {}

def save_commit_history(history, repo_root='.'):
    """Save commit hash history"""
    history_file = Path(repo_root) / COMMIT_HASH_FILE
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving commit history: {e}")

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

def was_file_changed_by_translation_commit(file_path):
    """Check if a file was last changed by a translation commit"""
    try:
        # Get the last commit that modified this file
        result = subprocess.run(['git', 'log', '-1', '--pretty=format:%s', '--', file_path], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            last_commit_msg = result.stdout.strip()
            # Check if it was a translation commit
            is_translation_commit = ('[skip-translation]' in last_commit_msg or 
                                   'Update markdown translations' in last_commit_msg or
                                   '🌐' in last_commit_msg)
            print(f"DEBUG: Last commit for {file_path}: '{last_commit_msg}' (translation: {is_translation_commit})")
            return is_translation_commit
    except Exception as e:
        print(f"DEBUG: Error checking commit history for {file_path}: {e}")
    return False

def check_simultaneous_edits_smart(changed_files):
    """Smart simultaneous edit detection that ignores translation commits"""
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
                # Check if the .ja.md file was changed by a translation commit
                if was_file_changed_by_translation_commit(str(readme_ja)):
                    print(f"DEBUG: {readme_ja} was changed by translation commit, not treating as simultaneous edit")
                    continue
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
            # Check if the pair file was changed by a translation commit
            if was_file_changed_by_translation_commit(str(pair_path)):
                print(f"DEBUG: {pair_path} was changed by translation commit, not treating as simultaneous edit")
                continue
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

def get_previous_version(file_path, repo_root='.'):
        """Get previous version of file from git history"""
        try:
            # Try to get file from previous commit
            result = subprocess.run(
                ['git', 'show', f'HEAD~1:{file_path}'],
                capture_output=True, text=True, cwd=repo_root
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

def get_current_translation(translated_file_path):
        """Get existing translation file content"""
        if os.path.exists(translated_file_path):
            try:
                with open(translated_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading translation file {translated_file_path}: {e}")
        
        return None

def incremental_translate(old_source, new_source, current_target, target_lang):
        """Perform incremental translation using LLM with 3-file context"""
        from utils.markdown_translate import OpenAI
        import os
        
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

def smart_translate(old_source, new_source, current_target, target_lang, incremental_threshold):
        """Hybrid translation with configurable threshold"""
        
        change_percentage = calculate_change_percentage(old_source, new_source)
        print(f"Change percentage: {change_percentage:.1f}%")
        
        if change_percentage >= incremental_threshold:
            # Large changes - use full translation
            print(f"Large changes detected ({change_percentage:.1f}% >= {incremental_threshold}%), using full translation")
            return translate_text(new_source, target_lang)
        else:
            # Small changes - use incremental translation
            print(f"Small changes detected ({change_percentage:.1f}% < {incremental_threshold}%), using incremental translation")
            result = incremental_translate(old_source, new_source, current_target, target_lang)
            
            # Fallback to full translation if incremental fails
            if not result:
                print("Incremental translation failed, falling back to full translation")
                return translate_text(new_source, target_lang)
            
            return result

def sync_translations(original_file, commit_history, current_commit_hash, ignore_patterns, incremental_threshold):
        """Sync translations with hybrid translation strategy"""
        if not os.path.exists(original_file):
            print(f"File {original_file} not found, skipping")
            return False

        # Check if file should be ignored
        if should_ignore_file(original_file, ignore_patterns):
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
        
        if is_pr_event:
            print(f"PR event detected, translating {processed_file} regardless of hash")
        else:
            # Check if file was already processed with this hash (only for non-PR events)
            print(f"DEBUG: Hash check for {processed_file}")
            print(f"DEBUG: file_key={file_key}")
            print(f"DEBUG: current_hash={current_hash}")
            print(f"DEBUG: file_key in commit_history: {file_key in commit_history}")
            if file_key in commit_history:
                stored_hash = commit_history[file_key].get('hash')
                stored_commit = commit_history[file_key].get('commit')
                print(f"DEBUG: stored_hash={stored_hash}")
                print(f"DEBUG: stored_commit={stored_commit}")
                print(f"DEBUG: current_commit_hash={current_commit_hash}")
                print(f"DEBUG: hash_match={stored_hash == current_hash}")
                print(f"DEBUG: commit_match={stored_commit == current_commit_hash}")
            
            if (file_key in commit_history and 
                commit_history[file_key].get('hash') == current_hash and
                commit_history[file_key].get('commit') == current_commit_hash):
                print(f"File {processed_file} has no actual changes, skipping translation")
                return False
        
        # Current source content
        new_source = read_file(processed_file)
        if not new_source.strip():
            print(f"File {processed_file} is empty, skipping")
            return False
        
        # Get previous version for hybrid translation
        old_source = get_previous_version(processed_file)
        if not old_source:
            old_source = get_previous_from_history(processed_file, commit_history)
        
        # Skip files that haven't actually changed (but not for PR events since workflow already filtered)
        if not is_pr_event and old_source and old_source.strip() == new_source.strip():
            print(f"File {processed_file} has no actual changes, skipping translation")
            return False
        elif is_pr_event:
            print(f"File {processed_file} has changes detected (PR event - trusting workflow filtering)")
        elif old_source:
            print(f"File {processed_file} has changes detected (old vs new content differs)")
        else:
            print(f"File {processed_file} - no previous version found, treating as new/changed")
            
        target_langs = [lang for lang in TARGET_LANGUAGES if lang != source_lang]
        
        translated = False
        for lang in target_langs:
            translated_file = get_translated_path(processed_file, lang)
            translated_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get current translation if it exists
            current_target = get_current_translation(str(translated_file))
            
            print(f"Translating {processed_file} ({source_lang}) → {translated_file} ({lang})")
            
            try:
                # Decide translation strategy
                if old_source and current_target:
                    # We have all 3 files - use hybrid approach
                    print(f"Using hybrid translation strategy")
                    translated_content = smart_translate(
                        old_source, new_source, current_target, lang, incremental_threshold
                    )
                else:
                    # Missing context - fall back to full translation
                    if not old_source:
                        print("No previous version found, using full translation")
                    if not current_target:
                        print("No existing translation found, using full translation")
                    translated_content = translate_text(new_source, lang)
                
                if translated_content and translated_content.strip():
                    # Write translated content
                    translated_file.write_text(translated_content, encoding='utf-8')
                    
                    # Apply formatting fixes immediately
                    apply_formatting_fixes(str(translated_file))
                    
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

def find_markdown_files(repo_root='.', ignore_patterns=None):
        """Find all markdown files in project, respecting ignore patterns and avoiding translation loops"""
        all_markdown_files = []
        
        # Recursively find all .md files from project root
        for root, dirs, files in os.walk(repo_root):
            # Skip ignored directories
            if ignore_patterns:
                dirs[:] = [d for d in dirs if not should_ignore_file(os.path.join(root, d), ignore_patterns)]
            
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.relpath(os.path.join(root, file), repo_root)
                    if not ignore_patterns or not should_ignore_file(file_path, ignore_patterns):
                        all_markdown_files.append(file_path)
        
        # Return all markdown files - both .en.md and .ja.md can be sources
        # The simultaneous edit detection will handle conflicts when both are changed
        filtered_files = all_markdown_files
        
        return filtered_files

def process_specific_files(file_list, commit_history, current_commit_hash, ignore_patterns, incremental_threshold):
        """Process specific files with simultaneous edit detection"""
        if not file_list:
            return []
        
        files = [f.strip() for f in file_list.split(',') if f.strip().endswith('.md')]
        
        # Check for simultaneous edits (when user manually edits both language files)
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
                if sync_translations(file, commit_history, current_commit_hash, ignore_patterns, incremental_threshold):
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
            if path.name.upper() == "README.MD":
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
        description='Enhanced markdown translation with integrated formatting'
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
    
    args = parser.parse_args()

    print("=== Enhanced Markdown Translation Starting ===")
    print(f"Client Repository: {args.client_repo}")
    print(f"PR Number: {args.pr_number}")
    print(f"Commit Hash: {args.commit_hash}")
    
    # Load configuration
    ignore_patterns = load_ignore_patterns('.')
    incremental_threshold = load_threshold('.')
    
    # Load commit history
    commit_history = load_commit_history('.')
    current_commit_hash = args.commit_hash

    # Handle deleted files first
    if args.deleted_files:
        print(f"Processing deleted files: {args.deleted_files}")
        delete_translated_files(args.deleted_files)

    # Process translations
    translation_count = 0
    
    try:
        if args.initial_setup:
            print("Performing initial setup translation")
            markdown_files = find_markdown_files('.', ignore_patterns)
            if not markdown_files:
                print("No markdown files found to translate")
            else:
                print(f"Found {len(markdown_files)} markdown files to process")
                for file in markdown_files:
                    if sync_translations(file, commit_history, current_commit_hash, ignore_patterns, incremental_threshold):
                        translation_count += 1
                        
        elif args.files:
            print(f"Processing specific files: {args.files}")
            processed_files = process_specific_files(
                args.files, commit_history, current_commit_hash, ignore_patterns, incremental_threshold
            )
            translation_count = len(processed_files)
            
        else:
            print("Processing all markdown files")
            markdown_files = find_markdown_files('.', ignore_patterns)
            if not markdown_files:
                print("No markdown files found to translate")
            else:
                print(f"Found {len(markdown_files)} markdown files to process")
                for file in markdown_files:
                    if sync_translations(file, commit_history, current_commit_hash, ignore_patterns, incremental_threshold):
                        translation_count += 1

        # Save updated commit history
        save_commit_history(commit_history, '.')
        
        print(f"=== Translation Complete: {translation_count} files processed ===")
        
    except Exception as e:
        print(f"Error during translation process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()