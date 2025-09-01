#!/usr/bin/env python3
"""
Enhanced post-commit translation script with integrated formatting and configuration support.
"""

import os
import sys
import argparse
import fnmatch
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

class TranslationConfig:
    """Handle translation configuration and ignore patterns"""
    
    def __init__(self, repo_root='.'):
        self.repo_root = Path(repo_root)
        self.ignore_patterns = self._load_ignore_patterns()
    
    def _load_ignore_patterns(self):
        """Load .translation_ignore patterns from client repo"""
        ignore_file = self.repo_root / TRANSLATION_IGNORE_FILE
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
    
    def should_ignore_file(self, file_path):
        """Check if file should be ignored based on patterns"""
        file_path_str = str(file_path)
        
        for pattern in self.ignore_patterns:
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

class MarkdownFormatter:
    """Handle markdown file formatting"""
    
    @staticmethod
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

class LanguageDetector:
    """Enhanced language detection for markdown files"""
    
    @staticmethod
    def detect_language(content):
        """Detect language based on character patterns and content"""
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

class TranslationManager:
    """Main translation management class"""
    
    def __init__(self, repo_root='.', client_repo=None, pr_number=None):
        self.repo_root = Path(repo_root)
        self.config = TranslationConfig(repo_root)
        self.formatter = MarkdownFormatter()
        self.detector = LanguageDetector()
        self.client_repo = client_repo
        self.pr_number = pr_number
        
    def get_file_language(self, file_path):
        """Determine language based on file extension and content"""
        path = Path(file_path)
        
        # Special case for README.md - detect language from content
        if path.name.upper() == "README.MD":
            if path.exists():
                content = self._read_file(file_path)
                return self.detector.detect_language(content)
            return "en"  # Default README to English
        
        # Check for explicit language extensions
        if path.name.endswith('.ja.md'):
            return "ja"
        elif path.name.endswith('.en.md'):
            return "en"
        elif path.name.endswith('.md'):
            # For other .md files, detect language
            if path.exists():
                content = self._read_file(file_path)
                return self.detector.detect_language(content)
            return "en"  # Default to English
        else:
            return None

    def get_translated_path(self, original_path, target_lang):
        """Generate translated file path maintaining directory structure"""
        path = Path(original_path)
        
        # Special case for README.md
        if path.name.upper() == "README.MD":
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

    def rename_ambiguous_md_file(self, file_path):
        """Rename .md file to .en.md or .ja.md based on detected language"""
        path = Path(file_path)
        
        # Skip README.md and already explicit files
        if (path.name.upper() == "README.MD" or 
            path.name.endswith('.en.md') or 
            path.name.endswith('.ja.md')):
            return file_path
        
        if path.name.endswith('.md'):
            content = self._read_file(file_path)
            detected_lang = self.detector.detect_language(content)
            
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

    def _read_file(self, file_path):
        """Read file with encoding fallback"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, return empty string
        print(f"Warning: Could not read {file_path} with any encoding")
        return ""

    def get_file_hash(self, file_path):
        """Get hash of file content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    def load_commit_history(self):
        """Load commit hash history"""
        history_file = self.repo_root / COMMIT_HASH_FILE
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading commit history: {e}")
                return {}
        return {}

    def save_commit_history(self, history):
        """Save commit hash history"""
        history_file = self.repo_root / COMMIT_HASH_FILE
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Error saving commit history: {e}")

    def check_simultaneous_edits(self, changed_files):
        """Check if both language pairs were edited in the same changeset"""
        files_set = set(changed_files)
        skip_files = set()
        
        for file_path in changed_files:
            if file_path in skip_files:
                continue
                
            path = Path(file_path)
            
            # Special case for README.md
            if path.name.upper() == "README.MD":
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

    def sync_translations(self, original_file, commit_history, current_commit_hash):
        """Sync translations with commit tracking and integrated formatting"""
        if not os.path.exists(original_file):
            print(f"File {original_file} not found, skipping")
            return False

        # Check if file should be ignored
        if self.config.should_ignore_file(original_file):
            return False
        
        # First, handle .md file renaming if needed
        processed_file = self.rename_ambiguous_md_file(original_file)
        
        source_lang = self.get_file_language(processed_file)
        if not source_lang:
            print(f"Cannot determine language for {processed_file}, skipping")
            return False
        
        # Get file hash for change detection
        current_hash = self.get_file_hash(processed_file)
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
        
        content = self._read_file(processed_file)
        if not content.strip():
            print(f"File {processed_file} is empty, skipping")
            return False
            
        target_langs = [lang for lang in TARGET_LANGUAGES if lang != source_lang]
        
        translated = False
        for lang in target_langs:
            translated_file = self.get_translated_path(processed_file, lang)
            translated_file.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Translating {processed_file} ({source_lang}) → {translated_file} ({lang})")
            try:
                translated_content = translate_text(content, lang)
                
                if translated_content and translated_content.strip():
                    # Write translated content
                    translated_file.write_text(translated_content, encoding='utf-8')
                    
                    # Apply formatting fixes immediately
                    self.formatter.apply_formatting_fixes(str(translated_file))
                    
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
        
        # Update commit history for source file
        if translated:
            commit_history[file_key] = {
                'hash': current_hash,
                'commit': current_commit_hash,
                'timestamp': datetime.now().isoformat(),
                'language': source_lang
            }
        
        return translated

    def find_markdown_files(self):
        """Find all markdown files in project, respecting ignore patterns"""
        markdown_files = []
        
        # Recursively find all .md files from project root
        for root, dirs, files in os.walk(self.repo_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self.config.should_ignore_file(os.path.join(root, d))]
            
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.relpath(os.path.join(root, file), self.repo_root)
                    if not self.config.should_ignore_file(file_path):
                        markdown_files.append(file_path)
        
        return markdown_files

    def process_specific_files(self, file_list, commit_history, current_commit_hash):
        """Process specific files with simultaneous edit detection"""
        if not file_list:
            return []
        
        files = [f.strip() for f in file_list.split(',') if f.strip().endswith('.md')]
        
        # Check for simultaneous edits
        skip_files = self.check_simultaneous_edits(files)
        
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
                if self.sync_translations(file, commit_history, current_commit_hash):
                    processed.append(file)
            else:
                print(f"File not found: {file}")
        
        return processed

    def delete_translated_files(self, deleted_files):
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
            
            source_lang = self.get_file_language(file)
            if not source_lang:
                continue
                
            # Delete corresponding translation
            target_lang = "ja" if source_lang == "en" else "en"
            translated_path = self.get_translated_path(file, target_lang)
            
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
    
    # Initialize translation manager
    manager = TranslationManager(
        repo_root='.',
        client_repo=args.client_repo,
        pr_number=args.pr_number
    )
    
    # Load commit history
    commit_history = manager.load_commit_history()
    current_commit_hash = args.commit_hash

    # Handle deleted files first
    if args.deleted_files:
        print(f"Processing deleted files: {args.deleted_files}")
        manager.delete_translated_files(args.deleted_files)

    # Process translations
    translation_count = 0
    
    try:
        if args.initial_setup:
            print("Performing initial setup translation")
            markdown_files = manager.find_markdown_files()
            if not markdown_files:
                print("No markdown files found to translate")
            else:
                print(f"Found {len(markdown_files)} markdown files to process")
                for file in markdown_files:
                    if manager.sync_translations(file, commit_history, current_commit_hash):
                        translation_count += 1
                        
        elif args.files:
            print(f"Processing specific files: {args.files}")
            processed_files = manager.process_specific_files(
                args.files, commit_history, current_commit_hash
            )
            translation_count = len(processed_files)
            
        else:
            print("Processing all markdown files")
            markdown_files = manager.find_markdown_files()
            if not markdown_files:
                print("No markdown files found to translate")
            else:
                print(f"Found {len(markdown_files)} markdown files to process")
                for file in markdown_files:
                    if manager.sync_translations(file, commit_history, current_commit_hash):
                        translation_count += 1

        # Save updated commit history
        manager.save_commit_history(commit_history)
        
        print(f"=== Translation Complete: {translation_count} files processed ===")
        
    except Exception as e:
        print(f"Error during translation process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()