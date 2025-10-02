#!/usr/bin/env python3
"""
Markdown Translation System - Core Translation Script

This script handles automated translation of markdown files between English and Japanese.
It supports both full and incremental translation modes based on file size and change percentage.
"""

import argparse
import logging
import sys
import fnmatch
import re
import time
import json
import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class MarkdownFile:
    """Represents a markdown file with translation metadata."""
    path: str
    language: str  # 'en', 'ja', or 'unknown'
    content: str
    diff_percentage: float
    file_size: int
    counterpart_path: str
    translation_mode: str  # 'full' or 'incremental'


@dataclass
class TranslationRequest:
    """Represents a translation request with all necessary context."""
    source_repo: str
    pr_number: int
    files: List[MarkdownFile]
    ignore_patterns: List[str]
    thresholds: Dict[str, Any]  # size_threshold, diff_threshold


@dataclass
class TranslationConfig:
    """Configuration settings for the translation system."""
    size_threshold: int = 1000  # characters
    diff_threshold: float = 0.3  # 30%
    llm_model: str = 'gpt-4'
    max_retries: int = 2
    log_level: str = 'INFO'
    supported_languages: List[str] = None
    llm_model: str = 'gpt-4'
    max_retries: int = 2
    log_level: str = 'INFO'
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ['en', 'ja']


class TranslationError(Exception):
    """Base exception for translation-related errors."""
    pass


class LanguageDetectionError(TranslationError):
    """Exception raised when language detection fails."""
    pass


class FileProcessingError(TranslationError):
    """Exception raised when file processing fails."""
    pass


class LLMError(TranslationError):
    """Exception raised when LLM API calls fail."""
    pass


class GitOperationError(TranslationError):
    """Exception raised when Git operations fail."""
    pass


@dataclass
class GitFileOperation:
    """Represents a Git file operation (read, write, rename, etc.)."""
    operation_type: str  # 'read', 'write', 'rename', 'commit'
    file_path: str
    new_file_path: Optional[str] = None  # For rename operations
    content: Optional[str] = None  # For write operations
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class LLMResponse:
    """Represents a response from the LLM API."""
    content: str
    model: str
    usage: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class LLMClient:
    """
    Client for interacting with Language Learning Models for translation.
    
    Supports OpenAI GPT models with retry logic and error handling.
    """
    
    def __init__(self, model: str = 'gpt-4', max_retries: int = 2, api_key: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            max_retries: Maximum number of retry attempts for failed requests
            api_key: OpenAI API key (if None, will try to get from environment)
        """
        self.model = model
        self.max_retries = max_retries
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.logger = logging.getLogger('markdown_translator.llm')
        
        if not self.api_key:
            raise LLMError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Import OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMError("OpenAI library not installed. Install with: pip install openai")
    
    def translate_full_content(self, content: str, source_language: str, target_language: str) -> LLMResponse:
        """
        Translate entire file content using full translation mode.
        
        This method sends the complete file content to the LLM for translation,
        maintaining markdown formatting and structure.
        
        Args:
            content: Complete markdown content to translate
            source_language: Source language code ('en' or 'ja')
            target_language: Target language code ('en' or 'ja')
            
        Returns:
            LLMResponse containing translated content
            
        Raises:
            LLMError: If translation fails after all retries
        """
        prompt = self._build_full_translation_prompt(content, source_language, target_language)
        
        return self._make_llm_request(prompt, f"full translation from {source_language} to {target_language}")
    
    def translate_incremental_content(self, original_content: str, modified_content: str, 
                                    existing_counterpart: str, source_language: str, 
                                    target_language: str) -> LLMResponse:
        """
        Translate content using incremental translation mode.
        
        This method sends the original file, changes, and existing counterpart
        to the LLM for partial translation of only the changed portions.
        
        Args:
            original_content: Original content before changes
            modified_content: Modified content with changes
            existing_counterpart: Existing translation counterpart content
            source_language: Source language code ('en' or 'ja')
            target_language: Target language code ('en' or 'ja')
            
        Returns:
            LLMResponse containing updated counterpart with translated changes
            
        Raises:
            LLMError: If translation fails after all retries
        """
        prompt = self._build_incremental_translation_prompt(
            original_content, modified_content, existing_counterpart, 
            source_language, target_language
        )
        
        return self._make_llm_request(prompt, f"incremental translation from {source_language} to {target_language}")
    
    def _build_full_translation_prompt(self, content: str, source_language: str, target_language: str) -> str:
        """
        Build prompt for full translation mode.
        
        Args:
            content: Content to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Formatted prompt for the LLM
        """
        lang_names = {'en': 'English', 'ja': 'Japanese'}
        source_name = lang_names.get(source_language, source_language)
        target_name = lang_names.get(target_language, target_language)
        
        return f"""You are a professional translator specializing in technical documentation. 
Please translate the following markdown content from {source_name} to {target_name}.

IMPORTANT INSTRUCTIONS:
1. Preserve all markdown formatting (headers, lists, code blocks, links, etc.)
2. Do not translate code blocks, URLs, or technical identifiers
3. Maintain the same document structure and organization
4. Ensure technical terms are translated appropriately for the target audience
5. Keep the same tone and style as the original
6. Return ONLY the translated content, no additional commentary

Content to translate:

{content}

Translated content:"""
    
    def _build_incremental_translation_prompt(self, original_content: str, modified_content: str,
                                            existing_counterpart: str, source_language: str, 
                                            target_language: str) -> str:
        """
        Build prompt for incremental translation mode.
        
        Args:
            original_content: Original content before changes
            modified_content: Modified content with changes
            existing_counterpart: Existing translation counterpart
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Formatted prompt for the LLM
        """
        lang_names = {'en': 'English', 'ja': 'Japanese'}
        source_name = lang_names.get(source_language, source_language)
        target_name = lang_names.get(target_language, target_language)
        
        return f"""You are a professional translator specializing in technical documentation.
I need you to update an existing {target_name} translation based on changes made to the {source_name} source document.

CONTEXT:
- Original {source_name} content: The content before changes
- Modified {source_name} content: The content after changes  
- Existing {target_name} translation: The current translation that needs updating

TASK:
Compare the original and modified {source_name} content to identify what changed, then update the existing {target_name} translation accordingly.

IMPORTANT INSTRUCTIONS:
1. Only translate the parts that changed in the source document
2. Preserve all markdown formatting (headers, lists, code blocks, links, etc.)
3. Do not translate code blocks, URLs, or technical identifiers
4. Maintain consistency with the existing translation style and terminology
5. Keep unchanged sections exactly as they are in the existing translation
6. Return the complete updated {target_name} document, not just the changes

ORIGINAL {source_name.upper()} CONTENT:
{original_content}

MODIFIED {source_name.upper()} CONTENT:
{modified_content}

EXISTING {target_name.upper()} TRANSLATION:
{existing_counterpart}

Updated {target_name} translation:"""
    
    def _make_llm_request(self, prompt: str, operation_description: str) -> LLMResponse:
        """
        Make a request to the LLM with retry logic and error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            operation_description: Description of the operation for logging
            
        Returns:
            LLMResponse containing the result
            
        Raises:
            LLMError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Attempting {operation_description} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # Lower temperature for more consistent translations
                    max_tokens=4000,  # Adjust based on expected response length
                )
                
                # Extract the response content
                content = response.choices[0].message.content
                
                if not content:
                    raise LLMError("Empty response from LLM")
                
                # Build usage information
                usage = {}
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                        'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                        'total_tokens': getattr(response.usage, 'total_tokens', 0)
                    }
                
                self.logger.info(f"Successfully completed {operation_description}")
                if usage:
                    self.logger.debug(f"Token usage: {usage}")
                
                return LLMResponse(
                    content=content.strip(),
                    model=self.model,
                    usage=usage,
                    success=True
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {operation_description}: {e}")
                
                # If this isn't the last attempt, wait before retrying
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        # All attempts failed
        error_msg = f"Failed to complete {operation_description} after {self.max_retries + 1} attempts. Last error: {last_error}"
        self.logger.error(error_msg)
        
        return LLMResponse(
            content="",
            model=self.model,
            usage={},
            success=False,
            error_message=error_msg
        )


class GitFileManager:
    """
    Manager for Git operations and file management.
    
    This class handles reading, writing, and committing markdown files,
    as well as file renaming operations required by the translation system.
    """
    
    def __init__(self, repository_path: str = ".", dry_run: bool = False):
        """
        Initialize the Git file manager.
        
        Args:
            repository_path: Path to the Git repository root
            dry_run: If True, operations will be logged but not executed
        """
        self.repository_path = Path(repository_path).resolve()
        self.dry_run = dry_run
        self.logger = logging.getLogger('markdown_translator.git')
        
        # Verify we're in a Git repository
        if not self._is_git_repository():
            raise GitOperationError(f"Directory {self.repository_path} is not a Git repository")
    
    def _is_git_repository(self) -> bool:
        """
        Check if the current directory is a Git repository.
        
        Returns:
            True if directory contains a .git folder, False otherwise
        """
        git_dir = self.repository_path / '.git'
        return git_dir.exists() and (git_dir.is_dir() or git_dir.is_file())
    
    def _run_git_command(self, command: List[str], check_output: bool = False) -> Tuple[bool, str]:
        """
        Run a Git command and return the result.
        
        Args:
            command: Git command as a list of strings
            check_output: If True, capture and return command output
            
        Returns:
            Tuple of (success: bool, output/error: str)
        """
        try:
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would execute git command: {' '.join(command)}")
                return True, "DRY RUN - command not executed"
            
            full_command = ['git'] + command
            self.logger.debug(f"Executing git command: {' '.join(full_command)}")
            
            if check_output:
                result = subprocess.run(
                    full_command,
                    cwd=self.repository_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return True, result.stdout.strip()
            else:
                result = subprocess.run(
                    full_command,
                    cwd=self.repository_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return True, ""
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Git command failed: {' '.join(full_command)}\nError: {e.stderr}"
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error running git command {' '.join(full_command)}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def read_file(self, file_path: str, encoding: str = 'utf-8') -> GitFileOperation:
        """
        Read a markdown file from the filesystem.
        
        Args:
            file_path: Path to the file to read (relative to repository root)
            encoding: File encoding (default: utf-8)
            
        Returns:
            GitFileOperation with the file content or error information
        """
        try:
            full_path = self.repository_path / file_path
            
            if not full_path.exists():
                return GitFileOperation(
                    operation_type='read',
                    file_path=file_path,
                    success=False,
                    error_message=f"File does not exist: {file_path}"
                )
            
            if not full_path.is_file():
                return GitFileOperation(
                    operation_type='read',
                    file_path=file_path,
                    success=False,
                    error_message=f"Path is not a file: {file_path}"
                )
            
            with open(full_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            self.logger.debug(f"Successfully read file: {file_path} ({len(content)} characters)")
            
            return GitFileOperation(
                operation_type='read',
                file_path=file_path,
                content=content,
                success=True
            )
            
        except UnicodeDecodeError as e:
            error_msg = f"Failed to decode file {file_path} with encoding {encoding}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='read',
                file_path=file_path,
                success=False,
                error_message=error_msg
            )
        except IOError as e:
            error_msg = f"Failed to read file {file_path}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='read',
                file_path=file_path,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error reading file {file_path}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='read',
                file_path=file_path,
                success=False,
                error_message=error_msg
            )
    
    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8', 
                   create_directories: bool = True) -> GitFileOperation:
        """
        Write content to a markdown file.
        
        Args:
            file_path: Path to the file to write (relative to repository root)
            content: Content to write to the file
            encoding: File encoding (default: utf-8)
            create_directories: If True, create parent directories if they don't exist
            
        Returns:
            GitFileOperation with success/error information
        """
        try:
            full_path = self.repository_path / file_path
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would write {len(content)} characters to {file_path}")
                return GitFileOperation(
                    operation_type='write',
                    file_path=file_path,
                    content=content,
                    success=True
                )
            
            # Create parent directories if needed
            if create_directories:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(full_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            self.logger.info(f"Successfully wrote file: {file_path} ({len(content)} characters)")
            
            return GitFileOperation(
                operation_type='write',
                file_path=file_path,
                content=content,
                success=True
            )
            
        except IOError as e:
            error_msg = f"Failed to write file {file_path}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='write',
                file_path=file_path,
                content=content,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error writing file {file_path}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='write',
                file_path=file_path,
                content=content,
                success=False,
                error_message=error_msg
            )
    
    def rename_file(self, old_path: str, new_path: str) -> GitFileOperation:
        """
        Rename a file and stage the change in Git.
        
        This function handles the file renaming logic required by requirements 2.2 and 2.3:
        - Rename .md files to .en.md or .ja.md based on detected language
        - Use Git mv to properly track the rename operation
        
        Args:
            old_path: Current file path (relative to repository root)
            new_path: New file path (relative to repository root)
            
        Returns:
            GitFileOperation with success/error information
        """
        try:
            old_full_path = self.repository_path / old_path
            new_full_path = self.repository_path / new_path
            
            # Validate source file exists
            if not old_full_path.exists():
                return GitFileOperation(
                    operation_type='rename',
                    file_path=old_path,
                    new_file_path=new_path,
                    success=False,
                    error_message=f"Source file does not exist: {old_path}"
                )
            
            # Check if target already exists
            if new_full_path.exists():
                return GitFileOperation(
                    operation_type='rename',
                    file_path=old_path,
                    new_file_path=new_path,
                    success=False,
                    error_message=f"Target file already exists: {new_path}"
                )
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would rename {old_path} to {new_path}")
                return GitFileOperation(
                    operation_type='rename',
                    file_path=old_path,
                    new_file_path=new_path,
                    success=True
                )
            
            # Create target directory if needed
            new_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try Git mv first (if file is tracked)
            success, output = self._run_git_command(['mv', old_path, new_path])
            
            if success:
                self.logger.info(f"Successfully renamed file using git mv: {old_path} -> {new_path}")
                return GitFileOperation(
                    operation_type='rename',
                    file_path=old_path,
                    new_file_path=new_path,
                    success=True
                )
            else:
                # Git mv failed, try regular file system move and then add/remove
                self.logger.debug(f"Git mv failed, trying filesystem move: {output}")
                
                # Move file using filesystem
                shutil.move(str(old_full_path), str(new_full_path))
                
                # Add new file to Git
                add_success, add_output = self._run_git_command(['add', new_path])
                if not add_success:
                    # Try to rollback the move
                    try:
                        shutil.move(str(new_full_path), str(old_full_path))
                    except:
                        pass
                    return GitFileOperation(
                        operation_type='rename',
                        file_path=old_path,
                        new_file_path=new_path,
                        success=False,
                        error_message=f"Failed to add renamed file to Git: {add_output}"
                    )
                
                # Remove old file from Git (if it was tracked)
                rm_success, rm_output = self._run_git_command(['rm', '--cached', old_path])
                # Note: rm --cached might fail if file wasn't tracked, which is OK
                
                self.logger.info(f"Successfully renamed file using filesystem move: {old_path} -> {new_path}")
                return GitFileOperation(
                    operation_type='rename',
                    file_path=old_path,
                    new_file_path=new_path,
                    success=True
                )
                
        except Exception as e:
            error_msg = f"Unexpected error renaming file {old_path} to {new_path}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='rename',
                file_path=old_path,
                new_file_path=new_path,
                success=False,
                error_message=error_msg
            )
    
    def commit_changes(self, file_paths: List[str], commit_message: str) -> GitFileOperation:
        """
        Commit changes to the specified files.
        
        Args:
            file_paths: List of file paths to commit (relative to repository root)
            commit_message: Commit message
            
        Returns:
            GitFileOperation with success/error information
        """
        try:
            if not file_paths:
                return GitFileOperation(
                    operation_type='commit',
                    file_path="",
                    success=False,
                    error_message="No files specified for commit"
                )
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would commit {len(file_paths)} files with message: {commit_message}")
                return GitFileOperation(
                    operation_type='commit',
                    file_path=", ".join(file_paths),
                    success=True
                )
            
            # Add files to staging area
            for file_path in file_paths:
                add_success, add_output = self._run_git_command(['add', file_path])
                if not add_success:
                    return GitFileOperation(
                        operation_type='commit',
                        file_path=file_path,
                        success=False,
                        error_message=f"Failed to add file to staging: {add_output}"
                    )
            
            # Commit the changes
            commit_success, commit_output = self._run_git_command(['commit', '-m', commit_message])
            
            if commit_success:
                self.logger.info(f"Successfully committed {len(file_paths)} files: {commit_message}")
                return GitFileOperation(
                    operation_type='commit',
                    file_path=", ".join(file_paths),
                    success=True
                )
            else:
                return GitFileOperation(
                    operation_type='commit',
                    file_path=", ".join(file_paths),
                    success=False,
                    error_message=f"Failed to commit changes: {commit_output}"
                )
                
        except Exception as e:
            error_msg = f"Unexpected error committing files {file_paths}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='commit',
                file_path=", ".join(file_paths),
                success=False,
                error_message=error_msg
            )
    
    def get_file_status(self, file_path: str) -> Dict[str, Any]:
        """
        Get Git status information for a file.
        
        Args:
            file_path: Path to the file (relative to repository root)
            
        Returns:
            Dictionary with status information:
            - exists: bool - whether file exists
            - tracked: bool - whether file is tracked by Git
            - modified: bool - whether file has uncommitted changes
            - staged: bool - whether file is staged for commit
        """
        try:
            full_path = self.repository_path / file_path
            
            status_info = {
                'exists': full_path.exists(),
                'tracked': False,
                'modified': False,
                'staged': False
            }
            
            if not status_info['exists']:
                return status_info
            
            # Check if file is tracked
            success, output = self._run_git_command(['ls-files', file_path], check_output=True)
            if success and output.strip():
                status_info['tracked'] = True
            
            # Check file status
            success, output = self._run_git_command(['status', '--porcelain', file_path], check_output=True)
            if success and output.strip():
                status_line = output.strip()
                # Parse Git status output
                if len(status_line) >= 2:
                    index_status = status_line[0]
                    worktree_status = status_line[1]
                    
                    # Check if staged
                    if index_status in ['A', 'M', 'D', 'R', 'C']:
                        status_info['staged'] = True
                    
                    # Check if modified
                    if worktree_status in ['M', 'D']:
                        status_info['modified'] = True
            
            return status_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get status for file {file_path}: {e}")
            return {
                'exists': False,
                'tracked': False,
                'modified': False,
                'staged': False
            }
    
    def handle_file_rename_for_language_detection(self, file_path: str, detected_language: str) -> GitFileOperation:
        """
        Handle file renaming based on language detection.
        
        This function implements requirements 2.2 and 2.3:
        - WHEN the language is English THEN the system SHALL rename the file to .en.md
        - WHEN the language is Japanese THEN the system SHALL rename the file to .ja.md
        
        Args:
            file_path: Path to the .md file (relative to repository root)
            detected_language: Detected language ('en' or 'ja')
            
        Returns:
            GitFileOperation with success/error information and new file path
        """
        try:
            # Only process plain .md files (not already language-specific)
            if not file_path.endswith('.md') or file_path.endswith('.en.md') or file_path.endswith('.ja.md'):
                return GitFileOperation(
                    operation_type='rename',
                    file_path=file_path,
                    new_file_path=file_path,
                    success=True  # No rename needed
                )
            
            # Special handling for README.md (requirement 2.5)
            if Path(file_path).name.lower() == 'readme.md':
                return GitFileOperation(
                    operation_type='rename',
                    file_path=file_path,
                    new_file_path=file_path,
                    success=True  # Keep README.md as-is
                )
            
            # Determine new file path based on detected language
            if detected_language == 'en':
                new_file_path = file_path.replace('.md', '.en.md')
            elif detected_language == 'ja':
                new_file_path = file_path.replace('.md', '.ja.md')
            else:
                # Default to English if language is unknown (requirement 8.1)
                self.logger.warning(f"Unknown language '{detected_language}' for {file_path}, defaulting to English")
                new_file_path = file_path.replace('.md', '.en.md')
            
            # Perform the rename
            return self.rename_file(file_path, new_file_path)
            
        except Exception as e:
            error_msg = f"Failed to handle file rename for {file_path}: {e}"
            self.logger.error(error_msg)
            return GitFileOperation(
                operation_type='rename',
                file_path=file_path,
                new_file_path=file_path,
                success=False,
                error_message=error_msg
            )


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration for the translation system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('markdown_translator')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the translation script.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Automated markdown translation system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --repo owner/repo --pr 123 --files file1.md file2.md
  %(prog)s --config config.json --files *.md
  %(prog)s --help
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--repository',
        required=True,
        help='Source repository in format owner/repo'
    )
    
    parser.add_argument(
        '--pr-number',
        type=int,
        required=True,
        help='Pull request number'
    )
    
    parser.add_argument(
        '--files',
        type=str,
        required=True,
        help='JSON string containing list of file metadata objects'
    )
    
    parser.add_argument(
        '--ignore-patterns',
        type=str,
        default='[]',
        help='JSON string containing list of ignore patterns'
    )
    
    parser.add_argument(
        '--working-dir',
        type=str,
        default='.',
        help='Working directory path for the client repository'
    )
    
    parser.add_argument(
        '--retry',
        action='store_true',
        help='Indicates this is a retry attempt'
    )
    
    # Optional configuration arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--size-threshold',
        type=int,
        default=1000,
        help='File size threshold for translation mode selection (default: 1000)'
    )
    
    parser.add_argument(
        '--diff-threshold',
        type=float,
        default=0.3,
        help='Diff percentage threshold for translation mode selection (default: 0.3)'
    )
    
    parser.add_argument(
        '--llm-model',
        type=str,
        default='gpt-4',
        help='LLM model to use for translation (default: gpt-4)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum number of retries for failed operations (default: 2)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--ignore-file',
        type=str,
        default='.md_ignore',
        help='Path to ignore patterns file (default: .md_ignore)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without making actual changes'
    )
    
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> TranslationConfig:
    """
    Load configuration from arguments and optional config file.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        TranslationConfig instance with merged settings
    """
    config = TranslationConfig(
        size_threshold=getattr(args, 'size_threshold', 1000),
        diff_threshold=getattr(args, 'diff_threshold', 0.3),
        llm_model=getattr(args, 'llm_model', 'gpt-4'),
        max_retries=getattr(args, 'max_retries', 2),
        log_level=getattr(args, 'log_level', 'INFO')
    )
    
    # Load additional configuration from file if provided
    if hasattr(args, 'config') and args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Update config with file values
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        
            except (json.JSONDecodeError, IOError) as e:
                raise TranslationError(f"Failed to load configuration file: {e}")
    
    return config


def detect_language(content: str) -> str:
    """
    Detect the language of markdown content using character frequency analysis.
    
    This function analyzes the character composition of the text to determine
    if it's primarily English or Japanese. It uses Unicode ranges for Japanese
    characters (Hiragana, Katakana, and Kanji) to make the determination.
    
    Args:
        content: The text content to analyze
        
    Returns:
        'ja' for Japanese, 'en' for English, or 'en' as default for edge cases
        
    Raises:
        LanguageDetectionError: If content analysis fails unexpectedly
    """
    try:
        if not content or not content.strip():
            # Empty or whitespace-only content defaults to English
            return 'en'
        
        # Remove markdown syntax and code blocks to focus on actual text content
        cleaned_content = _clean_markdown_for_analysis(content)
        
        if not cleaned_content.strip():
            # If no text remains after cleaning, default to English
            return 'en'
        
        # Count Japanese characters using Unicode ranges
        japanese_chars = 0
        total_text_chars = 0
        
        for char in cleaned_content:
            # Skip whitespace and punctuation for analysis
            if char.isspace() or not char.isprintable():
                continue
                
            total_text_chars += 1
            
            # Check if character is in Japanese Unicode ranges
            char_code = ord(char)
            
            # Hiragana: U+3040-U+309F
            # Katakana: U+30A0-U+30FF  
            # CJK Unified Ideographs (Kanji): U+4E00-U+9FAF
            # CJK Extension A: U+3400-U+4DBF
            if (0x3040 <= char_code <= 0x309F or    # Hiragana
                0x30A0 <= char_code <= 0x30FF or    # Katakana
                0x4E00 <= char_code <= 0x9FAF or    # CJK Unified Ideographs
                0x3400 <= char_code <= 0x4DBF):     # CJK Extension A
                japanese_chars += 1
        
        # If no meaningful text characters found, default to English
        if total_text_chars == 0:
            return 'en'
        
        # Calculate Japanese character ratio
        japanese_ratio = japanese_chars / total_text_chars
        
        # Use 10% threshold - if more than 10% of characters are Japanese, 
        # classify as Japanese content
        threshold = 0.1
        
        return 'ja' if japanese_ratio > threshold else 'en'
        
    except Exception as e:
        # Log the error but don't fail - default to English as per requirements
        logger = logging.getLogger('markdown_translator')
        logger.warning(f"Language detection failed for content (length: {len(content)}): {e}")
        return 'en'


def _clean_markdown_for_analysis(content: str) -> str:
    """
    Clean markdown content to focus on actual text for language analysis.
    
    Removes code blocks, inline code, URLs, and other markdown syntax
    that shouldn't influence language detection.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Cleaned text content suitable for language analysis
    """
    import re
    
    # Remove code blocks (both ``` and ~~~ style)
    content = re.sub(r'```[\s\S]*?```', '', content)
    content = re.sub(r'~~~[\s\S]*?~~~', '', content)
    
    # Remove inline code
    content = re.sub(r'`[^`]*`', '', content)
    
    # Remove URLs
    content = re.sub(r'https?://[^\s\)]+', '', content)
    
    # Remove markdown links but keep the text
    content = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', content)
    
    # Remove markdown images
    content = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', content)
    
    # Remove HTML tags
    content = re.sub(r'<[^>]*>', '', content)
    
    # Remove markdown headers (# symbols)
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    
    # Remove markdown emphasis markers but keep the text
    content = re.sub(r'\*\*([^\*]*)\*\*', r'\1', content)  # Bold
    content = re.sub(r'\*([^\*]*)\*', r'\1', content)      # Italic
    content = re.sub(r'__([^_]*)__', r'\1', content)       # Bold
    content = re.sub(r'_([^_]*)_', r'\1', content)         # Italic
    
    # Remove list markers
    content = re.sub(r'^\s*[-\*\+]\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
    
    # Remove blockquote markers
    content = re.sub(r'^\s*>\s*', '', content, flags=re.MULTILINE)
    
    return content


def detect_language_from_filename(file_path: str) -> Optional[str]:
    """
    Detect language from filename patterns.
    
    Args:
        file_path: Path to the file
        
    Returns:
        'en' for English files, 'ja' for Japanese files, None if undetermined
    """
    path_lower = file_path.lower()
    
    if path_lower.endswith('.en.md'):
        return 'en'
    elif path_lower.endswith('.ja.md'):
        return 'ja'
    else:
        return None


def get_counterpart_path(file_path: str, detected_language: Optional[str] = None) -> str:
    """
    Determine the counterpart file path for a given markdown file.
    
    This function handles the mapping between English and Japanese markdown files:
    - .en.md files map to .ja.md files
    - .ja.md files map to .en.md files  
    - README.md files have special handling to create both .en.md and .ja.md counterparts
    - Plain .md files are mapped based on detected language
    
    Args:
        file_path: Path to the source markdown file
        detected_language: Optional detected language ('en' or 'ja') for .md files
        
    Returns:
        Path to the counterpart file
        
    Raises:
        FileProcessingError: If file path format is invalid or unsupported
    """
    try:
        file_path = file_path.strip()
        if not file_path:
            raise FileProcessingError("Empty file path provided")
        
        # Convert to Path object for easier manipulation
        path_obj = Path(file_path)
        
        # Handle .en.md files -> .ja.md
        if file_path.endswith('.en.md'):
            return file_path.replace('.en.md', '.ja.md')
        
        # Handle .ja.md files -> .en.md
        elif file_path.endswith('.ja.md'):
            return file_path.replace('.ja.md', '.en.md')
        
        # Handle README.md special case
        elif path_obj.name.lower() == 'readme.md':
            # For README.md, we need to determine which counterpart to create
            # This will be handled by the caller to create both .en.md and .ja.md
            parent_dir = path_obj.parent
            if detected_language == 'en':
                return str(parent_dir / 'README.ja.md')
            elif detected_language == 'ja':
                return str(parent_dir / 'README.en.md')
            else:
                # Default to creating English counterpart if language unknown
                return str(parent_dir / 'README.en.md')
        
        # Handle plain .md files based on detected language
        elif file_path.endswith('.md'):
            if not detected_language:
                raise FileProcessingError(f"Cannot determine counterpart for {file_path}: language detection required")
            
            base_path = file_path[:-3]  # Remove .md extension
            
            if detected_language == 'en':
                return f"{base_path}.ja.md"
            elif detected_language == 'ja':
                return f"{base_path}.en.md"
            else:
                raise FileProcessingError(f"Unsupported language '{detected_language}' for file {file_path}")
        
        else:
            raise FileProcessingError(f"Unsupported file format: {file_path}")
            
    except Exception as e:
        if isinstance(e, FileProcessingError):
            raise
        else:
            raise FileProcessingError(f"Failed to determine counterpart path for {file_path}: {e}")


def get_all_counterpart_paths(file_path: str, detected_language: Optional[str] = None) -> List[str]:
    """
    Get all counterpart paths for a given file, handling README.md special case.
    
    For most files, this returns a single counterpart path.
    For README.md files, this returns both .en.md and .ja.md counterparts.
    
    Args:
        file_path: Path to the source markdown file
        detected_language: Optional detected language for .md files
        
    Returns:
        List of counterpart file paths
        
    Raises:
        FileProcessingError: If file path format is invalid or unsupported
    """
    try:
        path_obj = Path(file_path)
        
        # Special handling for README.md - create both counterparts
        if path_obj.name.lower() == 'readme.md':
            parent_dir = path_obj.parent
            return [
                str(parent_dir / 'README.en.md'),
                str(parent_dir / 'README.ja.md')
            ]
        
        # For all other files, return single counterpart
        else:
            return [get_counterpart_path(file_path, detected_language)]
            
    except Exception as e:
        if isinstance(e, FileProcessingError):
            raise
        else:
            raise FileProcessingError(f"Failed to get counterpart paths for {file_path}: {e}")


def validate_file_pair(source_path: str, counterpart_path: str) -> bool:
    """
    Validate that two files form a valid translation pair.
    
    Checks that the files have compatible naming patterns and are in the same directory.
    
    Args:
        source_path: Path to the source file
        counterpart_path: Path to the counterpart file
        
    Returns:
        True if the files form a valid pair, False otherwise
    """
    try:
        source_obj = Path(source_path)
        counterpart_obj = Path(counterpart_path)
        
        # Files must be in the same directory
        if source_obj.parent != counterpart_obj.parent:
            return False
        
        # Both must be markdown files
        if not (source_path.endswith('.md') and counterpart_path.endswith('.md')):
            return False
        
        # Check valid pairing patterns
        source_name = source_obj.name.lower()
        counterpart_name = counterpart_obj.name.lower()
        
        # README.md special cases
        if source_name == 'readme.md':
            return counterpart_name in ['readme.en.md', 'readme.ja.md']
        elif source_name in ['readme.en.md', 'readme.ja.md']:
            return counterpart_name in ['readme.md', 'readme.en.md', 'readme.ja.md'] and counterpart_name != source_name
        
        # Standard .en.md <-> .ja.md pairs
        elif source_name.endswith('.en.md'):
            expected_counterpart = source_name.replace('.en.md', '.ja.md')
            return counterpart_name == expected_counterpart
        elif source_name.endswith('.ja.md'):
            expected_counterpart = source_name.replace('.ja.md', '.en.md')
            return counterpart_name == expected_counterpart
        
        # Plain .md files paired with language-specific versions
        elif source_name.endswith('.md') and not source_name.endswith(('.en.md', '.ja.md')):
            base_name = source_name[:-3]  # Remove .md
            return counterpart_name in [f"{base_name}.en.md", f"{base_name}.ja.md"]
        
        return False
        
    except Exception:
        return False


def detect_file_pairs_in_directory(directory_path: str) -> List[tuple[str, str]]:
    """
    Detect existing file pairs in a directory.
    
    Scans a directory for markdown files and identifies translation pairs.
    
    Args:
        directory_path: Path to the directory to scan
        
    Returns:
        List of tuples containing (source_file, counterpart_file) pairs
        
    Raises:
        FileProcessingError: If directory cannot be accessed
    """
    try:
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            raise FileProcessingError(f"Directory does not exist: {directory_path}")
        
        if not dir_path.is_dir():
            raise FileProcessingError(f"Path is not a directory: {directory_path}")
        
        # Get all markdown files in directory
        md_files = []
        for file_path in dir_path.glob('*.md'):
            if file_path.is_file():
                md_files.append(str(file_path))
        
        pairs = []
        processed_files = set()
        
        for file_path in md_files:
            if file_path in processed_files:
                continue
            
            file_name = Path(file_path).name.lower()
            
            # Handle README.md special case
            if file_name == 'readme.md':
                # Look for README.en.md and README.ja.md counterparts
                readme_en = str(dir_path / 'README.en.md')
                readme_ja = str(dir_path / 'README.ja.md')
                
                if readme_en in md_files:
                    pairs.append((file_path, readme_en))
                    processed_files.add(readme_en)
                
                if readme_ja in md_files:
                    pairs.append((file_path, readme_ja))
                    processed_files.add(readme_ja)
                
                processed_files.add(file_path)
            
            # Handle .en.md files
            elif file_name.endswith('.en.md'):
                counterpart_path = file_path.replace('.en.md', '.ja.md')
                if counterpart_path in md_files:
                    pairs.append((file_path, counterpart_path))
                    processed_files.add(counterpart_path)
                processed_files.add(file_path)
            
            # Handle .ja.md files (if not already processed as counterpart)
            elif file_name.endswith('.ja.md'):
                counterpart_path = file_path.replace('.ja.md', '.en.md')
                if counterpart_path in md_files:
                    pairs.append((counterpart_path, file_path))  # Put .en.md first by convention
                    processed_files.add(counterpart_path)
                processed_files.add(file_path)
            
            # Handle plain .md files (not README.md)
            elif file_name.endswith('.md'):
                base_name = file_name[:-3]
                en_counterpart = str(dir_path / f'{base_name}.en.md')
                ja_counterpart = str(dir_path / f'{base_name}.ja.md')
                
                if en_counterpart in md_files:
                    pairs.append((file_path, en_counterpart))
                    processed_files.add(en_counterpart)
                
                if ja_counterpart in md_files:
                    pairs.append((file_path, ja_counterpart))
                    processed_files.add(ja_counterpart)
                
                processed_files.add(file_path)
        
        return pairs
        
    except Exception as e:
        if isinstance(e, FileProcessingError):
            raise
        else:
            raise FileProcessingError(f"Failed to detect file pairs in {directory_path}: {e}")


def is_readme_file(file_path: str) -> bool:
    """
    Check if a file is a README file (case-insensitive).
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a README file, False otherwise
    """
    file_name = Path(file_path).name.lower()
    return file_name in ['readme.md', 'readme.en.md', 'readme.ja.md']


def get_base_filename(file_path: str) -> str:
    """
    Get the base filename without language suffix and extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Base filename (e.g., 'document' from 'document.en.md')
    """
    file_name = Path(file_path).name
    
    # Handle README special case
    if file_name.lower().startswith('readme'):
        return 'README'
    
    # Remove language suffixes and .md extension
    if file_name.endswith('.en.md'):
        return file_name[:-6]  # Remove .en.md
    elif file_name.endswith('.ja.md'):
        return file_name[:-6]  # Remove .ja.md
    elif file_name.endswith('.md'):
        return file_name[:-3]  # Remove .md
    else:
        return file_name


def load_ignore_patterns(ignore_file_path: str) -> List[str]:
    """
    Load ignore patterns from a .md_ignore file.
    
    The ignore file format follows gitignore-style patterns:
    - One pattern per line
    - Lines starting with # are comments
    - Empty lines are ignored
    - Patterns can use glob-style wildcards (* and ?)
    - Patterns starting with / are treated as absolute from repository root
    - Patterns not starting with / can match anywhere in the path
    
    Args:
        ignore_file_path: Path to the .md_ignore file
        
    Returns:
        List of ignore patterns (empty list if file doesn't exist or is empty)
        
    Raises:
        FileProcessingError: If the ignore file exists but cannot be read
    """
    try:
        ignore_path = Path(ignore_file_path)
        
        # If ignore file doesn't exist, return empty list (no patterns to ignore)
        if not ignore_path.exists():
            return []
        
        if not ignore_path.is_file():
            raise FileProcessingError(f"Ignore file path is not a file: {ignore_file_path}")
        
        patterns = []
        
        with open(ignore_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Strip whitespace
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Validate pattern (basic check for obviously invalid patterns)
                if len(line) > 1000:  # Reasonable limit for pattern length
                    logger = logging.getLogger('markdown_translator')
                    logger.warning(f"Ignoring overly long pattern at line {line_num} in {ignore_file_path}")
                    continue
                
                patterns.append(line)
        
        logger = logging.getLogger('markdown_translator')
        logger.debug(f"Loaded {len(patterns)} ignore patterns from {ignore_file_path}")
        
        return patterns
        
    except IOError as e:
        raise FileProcessingError(f"Failed to read ignore file {ignore_file_path}: {e}")
    except Exception as e:
        raise FileProcessingError(f"Unexpected error loading ignore patterns from {ignore_file_path}: {e}")


def normalize_path_for_matching(file_path: str) -> str:
    """
    Normalize a file path for pattern matching.
    
    Converts path separators to forward slashes and removes leading slashes
    to ensure consistent matching across different operating systems.
    
    Args:
        file_path: File path to normalize
        
    Returns:
        Normalized path suitable for pattern matching
    """
    if not file_path:
        return ''
    
    # Convert to Path object and back to string to normalize separators
    normalized = str(Path(file_path)).replace('\\', '/')
    
    # Handle special case where Path('.') returns '.'
    if normalized == '.':
        return ''
    
    # Remove leading slash if present (for relative path matching)
    if normalized.startswith('/'):
        normalized = normalized[1:]
    
    return normalized


def matches_ignore_pattern(file_path: str, pattern: str) -> bool:
    """
    Check if a file path matches an ignore pattern.
    
    Supports gitignore-style pattern matching:
    - * matches any sequence of characters (except /)
    - ? matches any single character (except /)
    - ** matches any sequence of characters (including /)
    - Patterns starting with / match from repository root
    - Patterns not starting with / can match anywhere in the path
    - Patterns ending with / only match directories
    
    Args:
        file_path: File path to check (should be relative to repository root)
        pattern: Ignore pattern to match against
        
    Returns:
        True if the file path matches the pattern, False otherwise
    """
    try:
        # Normalize inputs
        normalized_path = normalize_path_for_matching(file_path)
        pattern = pattern.strip()
        
        if not pattern:
            return False
        
        # Handle directory-only patterns (ending with /)
        if pattern.endswith('/'):
            # For directory patterns, check if the file is inside a matching directory
            pattern = pattern[:-1]  # Remove trailing slash
            
            # Check if any directory in the path matches the pattern
            path_parts = normalized_path.split('/')
            
            # For each directory level, check if it matches the pattern
            for i in range(len(path_parts) - 1):  # Exclude the filename
                # Get the directory name at this level
                dir_name = path_parts[i]
                
                # Check if this directory name matches the pattern
                # For relative patterns, we need to check suffix matching too
                if pattern.startswith('/'):
                    # Absolute pattern - only match at specific level
                    abs_pattern = pattern[1:]
                    if i == 0 and _glob_match_with_doublestar(dir_name, abs_pattern):
                        return True
                else:
                    # Relative pattern - can match at any level
                    if _glob_match_with_doublestar(dir_name, pattern):
                        return True
            
            return False
        
        # Handle regular file/directory patterns
        return _match_pattern_against_path(normalized_path, pattern)
        
    except Exception as e:
        # If pattern matching fails, log warning and don't ignore the file
        logger = logging.getLogger('markdown_translator')
        logger.warning(f"Pattern matching failed for path '{file_path}' with pattern '{pattern}': {e}")
        return False


def _match_pattern_against_path(normalized_path: str, pattern: str) -> bool:
    """
    Internal helper to match a normalized path against a pattern.
    
    Args:
        normalized_path: Normalized file path
        pattern: Pattern to match (without trailing slash)
        
    Returns:
        True if path matches pattern, False otherwise
    """
    # Handle absolute patterns (starting with /)
    if pattern.startswith('/'):
        # Remove leading slash and match from root
        pattern = pattern[1:]
        return _glob_match_with_doublestar(normalized_path, pattern)
    
    # Handle relative patterns - can match anywhere in the path
    else:
        # Try matching the full path first
        if _glob_match_with_doublestar(normalized_path, pattern):
            return True
        
        # For relative patterns, try matching against path suffixes
        # But be careful with directory patterns and path boundaries
        path_parts = normalized_path.split('/')
        
        # Try matching each suffix of the path
        for i in range(len(path_parts)):
            suffix_path = '/'.join(path_parts[i:])
            if _glob_match_with_doublestar(suffix_path, pattern):
                return True
        
        return False


def _glob_match_with_doublestar(path: str, pattern: str) -> bool:
    """
    Perform glob matching with support for ** (double star) patterns.
    
    Args:
        path: Path to match
        pattern: Glob pattern (may contain ** for recursive matching)
        
    Returns:
        True if path matches pattern, False otherwise
    """
    # Handle ** patterns by converting to regex
    if '**' in pattern:
        # Convert glob pattern to regex
        regex_pattern = _glob_to_regex(pattern)
        try:
            return bool(re.match(regex_pattern, path))
        except re.error:
            # If regex compilation fails, fall back to simple fnmatch
            return fnmatch.fnmatch(path, pattern.replace('**', '*'))
    
    # Use standard fnmatch for simple patterns
    # fnmatch treats * as not matching /, which is what we want
    # On Windows, fnmatch might be case-insensitive, so we need to ensure case sensitivity
    import os
    if os.name == 'nt':  # Windows
        # Use case-sensitive matching by converting to regex
        regex_pattern = _glob_to_regex(pattern)
        try:
            return bool(re.match(regex_pattern, path))
        except re.error:
            return fnmatch.fnmatch(path, pattern)
    else:
        return fnmatch.fnmatch(path, pattern)


def _glob_to_regex(pattern: str) -> str:
    """
    Convert a glob pattern with ** support to a regex pattern.
    
    Args:
        pattern: Glob pattern
        
    Returns:
        Equivalent regex pattern
    """
    # Split pattern by ** to handle each part separately
    parts = pattern.split('**')
    
    if len(parts) == 1:
        # No ** in pattern, handle normally
        escaped = re.escape(pattern)
        escaped = escaped.replace(r'\*', '[^/]*')  # * matches anything except /
        escaped = escaped.replace(r'\?', '[^/]')   # ? matches single char except /
        return f'^{escaped}$'
    
    # Handle ** patterns
    regex_parts = []
    
    for i, part in enumerate(parts):
        if i == 0:
            # First part
            if part:
                escaped_part = re.escape(part)
                escaped_part = escaped_part.replace(r'\*', '[^/]*')
                escaped_part = escaped_part.replace(r'\?', '[^/]')
                regex_parts.append(escaped_part)
        elif i == len(parts) - 1:
            # Last part
            if part:
                escaped_part = re.escape(part)
                escaped_part = escaped_part.replace(r'\*', '[^/]*')
                escaped_part = escaped_part.replace(r'\?', '[^/]')
                # ** before last part should match any path including empty
                if part.startswith('/'):
                    # If last part starts with /, ** can match empty or any path
                    regex_parts.append(f'(?:.*{escaped_part}|{escaped_part[1:]})')
                else:
                    # If last part doesn't start with /, we need to add the separator
                    regex_parts.append(f'(?:.*/|){escaped_part}')
            else:
                # Pattern ends with **, match anything
                regex_parts.append('.*')
        else:
            # Middle part
            if part:
                escaped_part = re.escape(part)
                escaped_part = escaped_part.replace(r'\*', '[^/]*')
                escaped_part = escaped_part.replace(r'\?', '[^/]')
                regex_parts.append(f'.*{escaped_part}')
    
    # Join all parts
    regex = ''.join(regex_parts)
    
    # Anchor the pattern to match the full string
    return f'^{regex}$'


def should_ignore_file(file_path: str, ignore_patterns: List[str]) -> bool:
    """
    Check if a file should be ignored based on ignore patterns.
    
    Args:
        file_path: Path to the file (relative to repository root)
        ignore_patterns: List of ignore patterns to check against
        
    Returns:
        True if the file should be ignored, False otherwise
    """
    if not ignore_patterns:
        return False
    
    for pattern in ignore_patterns:
        if matches_ignore_pattern(file_path, pattern):
            logger = logging.getLogger('markdown_translator')
            logger.debug(f"File '{file_path}' matches ignore pattern '{pattern}'")
            return True
    
    return False


def filter_files_by_ignore_patterns(file_paths: List[str], ignore_patterns: List[str]) -> List[str]:
    """
    Filter a list of file paths by removing those that match ignore patterns.
    
    Args:
        file_paths: List of file paths to filter
        ignore_patterns: List of ignore patterns
        
    Returns:
        Filtered list of file paths (files that should NOT be ignored)
    """
    if not ignore_patterns:
        return file_paths.copy()
    
    filtered_files = []
    ignored_count = 0
    
    for file_path in file_paths:
        if should_ignore_file(file_path, ignore_patterns):
            ignored_count += 1
        else:
            filtered_files.append(file_path)
    
    logger = logging.getLogger('markdown_translator')
    if ignored_count > 0:
        logger.info(f"Filtered out {ignored_count} files based on ignore patterns")
    
    return filtered_files


def calculate_file_size(content: str) -> int:
    """
    Calculate the size of file content in characters.
    
    This function counts the total number of characters in the file content,
    which is used as one of the criteria for determining translation mode.
    
    Args:
        content: The file content as a string
        
    Returns:
        Number of characters in the content
        
    Raises:
        ValueError: If content is None
    """
    if content is None:
        raise ValueError("Content cannot be None")
    
    return len(content)


def calculate_diff_percentage(original_content: str, modified_content: str) -> float:
    """
    Calculate the percentage of content that has changed between two versions.
    
    This function compares the original and modified content to determine
    what percentage of the file has been changed. It uses a line-by-line
    comparison to calculate the diff ratio.
    
    Args:
        original_content: The original file content
        modified_content: The modified file content
        
    Returns:
        Percentage of content changed (0.0 to 1.0)
        
    Raises:
        ValueError: If either content parameter is None
    """
    if original_content is None or modified_content is None:
        raise ValueError("Content parameters cannot be None")
    
    # Handle edge cases
    if original_content == modified_content:
        return 0.0
    
    if not original_content and modified_content:
        return 1.0  # 100% change (new file)
    
    if original_content and not modified_content:
        return 1.0  # 100% change (file deleted/emptied)
    
    # Split content into lines for comparison
    original_lines = original_content.splitlines()
    modified_lines = modified_content.splitlines()
    
    # Use difflib to calculate similarity ratio
    import difflib
    
    # Create a SequenceMatcher to compare the lines
    matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
    
    # Get the similarity ratio (0.0 to 1.0)
    similarity_ratio = matcher.ratio()
    
    # Convert similarity to diff percentage
    diff_percentage = 1.0 - similarity_ratio
    
    return diff_percentage


def determine_translation_mode(file_content: str, original_content: str, config: TranslationConfig) -> str:
    """
    Determine the appropriate translation mode based on file size and diff percentage.
    
    This function implements the logic from requirements 5.1 and 5.3:
    - If file is above size threshold OR diff exceeds threshold → full mode
    - If file is below size threshold AND diff is under threshold → incremental mode
    
    Args:
        file_content: Current content of the file to be translated
        original_content: Original content before changes (for diff calculation)
        config: Translation configuration containing thresholds
        
    Returns:
        'full' for full translation mode, 'incremental' for incremental mode
        
    Raises:
        ValueError: If content parameters are None or config is invalid
        TranslationError: If mode determination fails
    """
    try:
        if file_content is None:
            raise ValueError("file_content cannot be None")
        
        if config is None:
            raise ValueError("config cannot be None")
        
        # Calculate file size
        file_size = calculate_file_size(file_content)
        
        # Calculate diff percentage (if original content is provided)
        diff_percentage = 0.0
        if original_content is not None:
            diff_percentage = calculate_diff_percentage(original_content, file_content)
        else:
            # If no original content, treat as 100% change (new file)
            diff_percentage = 1.0
        
        # Apply threshold logic from requirements 5.1 and 5.3
        size_threshold = config.size_threshold
        diff_threshold = config.diff_threshold
        
        # Requirement 5.1: If file is above length threshold Y OR diff amount exceeds X% → full mode
        if file_size > size_threshold or diff_percentage > diff_threshold:
            return 'full'
        
        # Requirement 5.3: If file is below length threshold Y AND diff amount is under X% → incremental mode
        else:
            return 'incremental'
            
    except ValueError:
        raise
    except Exception as e:
        raise TranslationError(f"Failed to determine translation mode: {e}")


def get_translation_mode_info(file_content: str, original_content: str, config: TranslationConfig) -> Dict[str, Any]:
    """
    Get detailed information about translation mode determination.
    
    This function provides comprehensive information about why a particular
    translation mode was selected, including the calculated metrics.
    
    Args:
        file_content: Current content of the file to be translated
        original_content: Original content before changes (for diff calculation)
        config: Translation configuration containing thresholds
        
    Returns:
        Dictionary containing:
        - mode: Selected translation mode ('full' or 'incremental')
        - file_size: Calculated file size in characters
        - diff_percentage: Calculated diff percentage (0.0 to 1.0)
        - size_threshold: Size threshold from config
        - diff_threshold: Diff threshold from config
        - size_exceeds_threshold: Boolean indicating if size exceeds threshold
        - diff_exceeds_threshold: Boolean indicating if diff exceeds threshold
        - reason: Human-readable explanation of mode selection
        
    Raises:
        ValueError: If content parameters are None or config is invalid
        TranslationError: If mode determination fails
    """
    try:
        if file_content is None:
            raise ValueError("file_content cannot be None")
        
        if config is None:
            raise ValueError("config cannot be None")
        
        # Calculate metrics
        file_size = calculate_file_size(file_content)
        
        diff_percentage = 0.0
        if original_content is not None:
            diff_percentage = calculate_diff_percentage(original_content, file_content)
        else:
            diff_percentage = 1.0
        
        # Get thresholds
        size_threshold = config.size_threshold
        diff_threshold = config.diff_threshold
        
        # Check threshold conditions
        size_exceeds_threshold = file_size > size_threshold
        diff_exceeds_threshold = diff_percentage > diff_threshold
        
        # Determine mode
        mode = determine_translation_mode(file_content, original_content, config)
        
        # Generate reason
        if mode == 'full':
            reasons = []
            if size_exceeds_threshold:
                reasons.append(f"file size ({file_size} chars) exceeds threshold ({size_threshold} chars)")
            if diff_exceeds_threshold:
                reasons.append(f"diff percentage ({diff_percentage:.1%}) exceeds threshold ({diff_threshold:.1%})")
            reason = f"Full translation mode selected because: {' and '.join(reasons)}"
        else:
            reason = f"Incremental translation mode selected because file size ({file_size} chars) ≤ threshold ({size_threshold} chars) and diff percentage ({diff_percentage:.1%}) ≤ threshold ({diff_threshold:.1%})"
        
        return {
            'mode': mode,
            'file_size': file_size,
            'diff_percentage': diff_percentage,
            'size_threshold': size_threshold,
            'diff_threshold': diff_threshold,
            'size_exceeds_threshold': size_exceeds_threshold,
            'diff_exceeds_threshold': diff_exceeds_threshold,
            'reason': reason
        }
        
    except ValueError:
        raise
    except Exception as e:
        raise TranslationError(f"Failed to get translation mode info: {e}")


def translate_markdown_file(file_path: str, target_language: str, config: TranslationConfig, 
                          original_content: Optional[str] = None, 
                          existing_counterpart_content: Optional[str] = None) -> str:
    """
    Translate a markdown file to the target language.
    
    This function implements the core translation logic, choosing between full
    and incremental translation modes based on file size and change percentage.
    
    Args:
        file_path: Path to the source markdown file
        target_language: Target language code ('en' or 'ja')
        config: Translation configuration
        original_content: Original content before changes (for incremental mode)
        existing_counterpart_content: Existing counterpart content (for incremental mode)
        
    Returns:
        Translated content as a string
        
    Raises:
        TranslationError: If translation fails
        FileProcessingError: If file cannot be read
        LLMError: If LLM API calls fail
    """
    logger = logging.getLogger('markdown_translator')
    
    try:
        # Read the source file content
        with open(file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Detect source language
        source_language = detect_language_from_filename(file_path)
        if not source_language:
            source_language = detect_language(current_content)
        
        logger.info(f"Translating {file_path} from {source_language} to {target_language}")
        
        # Determine translation mode
        translation_mode = determine_translation_mode(current_content, original_content or "", config)
        mode_info = get_translation_mode_info(current_content, original_content or "", config)
        
        logger.info(f"Using {translation_mode} translation mode")
        logger.debug(f"Mode selection reason: {mode_info['reason']}")
        
        # Initialize LLM client
        llm_client = LLMClient(model=config.llm_model, max_retries=config.max_retries)
        
        # Perform translation based on mode
        if translation_mode == 'full':
            response = llm_client.translate_full_content(
                content=current_content,
                source_language=source_language,
                target_language=target_language
            )
        else:  # incremental mode
            if existing_counterpart_content is None:
                logger.warning("Incremental mode requested but no existing counterpart provided, falling back to full translation")
                response = llm_client.translate_full_content(
                    content=current_content,
                    source_language=source_language,
                    target_language=target_language
                )
            else:
                response = llm_client.translate_incremental_content(
                    original_content=original_content or "",
                    modified_content=current_content,
                    existing_counterpart=existing_counterpart_content,
                    source_language=source_language,
                    target_language=target_language
                )
        
        # Check if translation was successful
        if not response.success:
            raise LLMError(f"Translation failed: {response.error_message}")
        
        logger.info(f"Translation completed successfully for {file_path}")
        return response.content
        
    except FileNotFoundError:
        raise FileProcessingError(f"Source file not found: {file_path}")
    except IOError as e:
        raise FileProcessingError(f"Failed to read source file {file_path}: {e}")
    except (LLMError, TranslationError):
        raise
    except Exception as e:
        raise TranslationError(f"Unexpected error during translation of {file_path}: {e}")


def translate_file_pair(source_file_path: str, counterpart_file_path: str, config: TranslationConfig,
                       original_source_content: Optional[str] = None, 
                       git_manager: Optional[GitFileManager] = None) -> str:
    """
    Translate a source file and create/update its counterpart.
    
    This function handles the complete translation workflow for a file pair,
    including reading existing counterparts and determining the appropriate
    translation mode.
    
    Args:
        source_file_path: Path to the source markdown file
        counterpart_file_path: Path to the counterpart file to create/update
        config: Translation configuration
        original_source_content: Original source content before changes (for incremental mode)
        
    Returns:
        Translated content that was written to the counterpart file
        
    Raises:
        TranslationError: If translation fails
        FileProcessingError: If file operations fail
    """
    logger = logging.getLogger('markdown_translator')
    
    try:
        # Determine target language from counterpart file path
        target_language = detect_language_from_filename(counterpart_file_path)
        if not target_language:
            # Try to infer from source file
            source_language = detect_language_from_filename(source_file_path)
            if source_language == 'en':
                target_language = 'ja'
            elif source_language == 'ja':
                target_language = 'en'
            else:
                # Default to Japanese if we can't determine
                target_language = 'ja'
        
        logger.info(f"Processing file pair: {source_file_path} -> {counterpart_file_path}")
        
        # Read existing counterpart content if it exists
        existing_counterpart_content = None
        
        if git_manager:
            # Use Git file manager if provided
            status = git_manager.get_file_status(counterpart_file_path)
            if status['exists']:
                try:
                    existing_counterpart_content = read_markdown_file_with_git(git_manager, counterpart_file_path)
                    logger.debug(f"Found existing counterpart file: {counterpart_file_path}")
                except FileProcessingError as e:
                    logger.warning(f"Could not read existing counterpart file {counterpart_file_path}: {e}")
        else:
            # Fallback to direct file access
            counterpart_path = Path(counterpart_file_path)
            if counterpart_path.exists():
                try:
                    with open(counterpart_path, 'r', encoding='utf-8') as f:
                        existing_counterpart_content = f.read()
                    logger.debug(f"Found existing counterpart file: {counterpart_file_path}")
                except IOError as e:
                    logger.warning(f"Could not read existing counterpart file {counterpart_file_path}: {e}")
        
        # Perform translation
        translated_content = translate_markdown_file(
            file_path=source_file_path,
            target_language=target_language,
            config=config,
            original_content=original_source_content,
            existing_counterpart_content=existing_counterpart_content
        )
        
        # Apply markdown linting to translated content (requirement 6.1)
        linted_content = apply_markdown_linting(counterpart_file_path, translated_content)
        
        # Write linted translated content to counterpart file
        if git_manager:
            # Use Git file manager if provided
            write_markdown_file_with_git(git_manager, counterpart_file_path, linted_content)
        else:
            # Fallback to direct file access
            counterpart_path = Path(counterpart_file_path)
            counterpart_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(counterpart_path, 'w', encoding='utf-8') as f:
                f.write(linted_content)
        
        logger.info(f"Successfully wrote translated content to {counterpart_file_path}")
        return linted_content
        
    except (TranslationError, FileProcessingError):
        raise
    except Exception as e:
        raise TranslationError(f"Failed to process file pair {source_file_path} -> {counterpart_file_path}: {e}")


def check_both_counterparts_modified(file_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Check for file pairs where both counterparts are modified in the same PR.
    
    This function implements requirement 3.4: WHEN both .en.md and .ja.md files 
    are modified in the same PR THEN the system SHALL skip translation for that file pair.
    
    Args:
        file_paths: List of all file paths being processed in the PR
        
    Returns:
        List of tuples containing (en_file, ja_file) pairs that should be skipped
    """
    logger = logging.getLogger('markdown_translator')
    
    # Group files by their base name to find pairs
    file_groups = {}
    skip_pairs = []
    
    for file_path in file_paths:
        try:
            path_obj = Path(file_path)
            file_name = path_obj.name.lower()
            
            # Handle different file patterns
            if file_name.endswith('.en.md'):
                base_name = file_name[:-6]  # Remove .en.md
                key = f"{path_obj.parent}/{base_name}"
                if key not in file_groups:
                    file_groups[key] = {}
                file_groups[key]['en'] = file_path
                
            elif file_name.endswith('.ja.md'):
                base_name = file_name[:-6]  # Remove .ja.md
                key = f"{path_obj.parent}/{base_name}"
                if key not in file_groups:
                    file_groups[key] = {}
                file_groups[key]['ja'] = file_path
                
            elif file_name == 'readme.md':
                # README.md files are handled separately - they don't form skip pairs
                continue
                
            elif file_name.endswith('.md'):
                # Plain .md files don't form skip pairs since they get renamed
                continue
                
        except Exception as e:
            logger.warning(f"Failed to analyze file path {file_path}: {e}")
            continue
    
    # Find pairs where both en and ja files are present
    for key, files in file_groups.items():
        if 'en' in files and 'ja' in files:
            skip_pairs.append((files['en'], files['ja']))
            logger.info(f"Found both counterparts modified, will skip translation: {files['en']} <-> {files['ja']}")
    
    return skip_pairs


def process_translation_request(request: TranslationRequest, config: TranslationConfig,
                              git_manager: Optional[GitFileManager] = None) -> Dict[str, Any]:
    """
    Process a complete translation request for multiple files.
    
    This function orchestrates the translation of multiple markdown files,
    applying ignore patterns, checking for skip conditions, and handling errors gracefully.
    Implements parallel processing for multiple files (requirement 8.5).
    
    Args:
        request: TranslationRequest containing files and metadata
        config: Translation configuration
        git_manager: Git file manager for file operations
        
    Returns:
        Dictionary containing processing results:
        - processed_files: List of successfully processed files
        - failed_files: List of files that failed processing
        - skipped_files: List of files that were skipped
        - total_files: Total number of files in request
        
    Raises:
        TranslationError: If the entire request fails
    """
    import concurrent.futures
    import threading
    
    logger = logging.getLogger('markdown_translator')
    
    try:
        logger.info(f"Processing translation request for {len(request.files)} files")
        
        # Check for file pairs where both counterparts are modified (requirement 3.4)
        all_file_paths = [f.path for f in request.files]
        skip_pairs = check_both_counterparts_modified(all_file_paths)
        skip_files = set()
        for en_file, ja_file in skip_pairs:
            skip_files.add(en_file)
            skip_files.add(ja_file)
        
        processed_files = []
        failed_files = []
        skipped_files = []
        
        # Filter files that should be processed
        files_to_process = []
        for markdown_file in request.files:
            file_path = markdown_file.path
            
            # Check if file should be skipped due to both counterparts being modified
            if file_path in skip_files:
                logger.info(f"Skipping file (both counterparts modified): {file_path}")
                skipped_files.append({
                    'file_path': file_path,
                    'reason': 'both_counterparts_modified'
                })
                continue
            
            # Check if file should be ignored by patterns
            if should_ignore_file(file_path, request.ignore_patterns):
                logger.info(f"Skipping ignored file: {file_path}")
                skipped_files.append({
                    'file_path': file_path,
                    'reason': 'matched_ignore_pattern'
                })
                continue
            
            files_to_process.append(markdown_file)
        
        if not files_to_process:
            logger.info("No files to process after filtering")
            return {
                'processed_files': processed_files,
                'failed_files': failed_files,
                'skipped_files': skipped_files,
                'total_files': len(request.files)
            }
        
        # Process files in parallel (requirement 8.5)
        logger.info(f"Processing {len(files_to_process)} files in parallel")
        
        # Thread-safe collections for results
        processed_lock = threading.Lock()
        failed_lock = threading.Lock()
        
        def process_single_file(markdown_file: MarkdownFile) -> None:
            """Process a single markdown file and its counterparts."""
            file_logger = logging.getLogger(f'markdown_translator.worker')
            
            try:
                file_path = markdown_file.path
                file_logger.debug(f"Processing file: {file_path}")
                
                # Get counterpart path(s)
                counterpart_paths = get_all_counterpart_paths(file_path, markdown_file.language)
                
                # Process each counterpart
                for counterpart_path in counterpart_paths:
                    try:
                        # Skip if counterpart is the same as source (shouldn't happen, but safety check)
                        if counterpart_path == file_path:
                            continue
                        
                        # Translate and create/update counterpart
                        translated_content = translate_file_pair(
                            source_file_path=file_path,
                            counterpart_file_path=counterpart_path,
                            config=config,
                            original_source_content=None,  # TODO: Get from git diff in future tasks
                            git_manager=git_manager
                        )
                        
                        # Thread-safe result collection
                        with processed_lock:
                            processed_files.append({
                                'source_file': file_path,
                                'counterpart_file': counterpart_path,
                                'translation_mode': markdown_file.translation_mode,
                                'content_length': len(translated_content)
                            })
                        
                        file_logger.info(f"Successfully processed: {file_path} -> {counterpart_path}")
                        
                    except Exception as e:
                        file_logger.error(f"Failed to process counterpart {counterpart_path} for {file_path}: {e}")
                        with failed_lock:
                            failed_files.append({
                                'source_file': file_path,
                                'counterpart_file': counterpart_path,
                                'error': str(e)
                            })
                
            except Exception as e:
                file_logger.error(f"Failed to process file {markdown_file.path}: {e}")
                with failed_lock:
                    failed_files.append({
                        'source_file': markdown_file.path,
                        'counterpart_file': None,
                        'error': str(e)
                    })
        
        # Use ThreadPoolExecutor for parallel processing
        # Limit concurrent threads to avoid overwhelming the LLM API
        max_workers = min(len(files_to_process), 3)  # Max 3 concurrent translations
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            futures = [executor.submit(process_single_file, markdown_file) 
                      for markdown_file in files_to_process]
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
            
            # Check for any exceptions in the futures
            for future in futures:
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as e:
                    logger.error(f"Parallel processing task failed: {e}")
        
        # Log summary
        logger.info(f"Translation request completed:")
        logger.info(f"  Processed: {len(processed_files)} file pairs")
        logger.info(f"  Failed: {len(failed_files)} files")
        logger.info(f"  Skipped: {len(skipped_files)} files")
        
        return {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'skipped_files': skipped_files,
            'total_files': len(request.files)
        }
        
    except Exception as e:
        raise TranslationError(f"Failed to process translation request: {e}")


def auto_lint_markdown(content: str) -> str:
    """
    Apply automatic linting and formatting fixes to markdown content.
    
    This function implements basic markdown linting requirements:
    - Remove redundant extra spaces (trailing whitespace and multiple consecutive spaces)
    - Ensure proper line endings (LF)
    - Ensure single final newline
    
    Args:
        content: Raw markdown content to lint and format
        
    Returns:
        Linted and formatted markdown content
        
    Raises:
        ValueError: If content is None
    """
    if content is None:
        raise ValueError("Content cannot be None")
    
    logger = logging.getLogger('markdown_translator')
    logger.debug("Starting markdown auto-lint process")
    
    # Start with the original content
    linted_content = content
    
    # Apply basic linting rules
    linted_content = _fix_line_endings(linted_content)
    linted_content = _remove_redundant_spaces(linted_content)
    linted_content = _ensure_single_final_newline(linted_content)
    
    logger.debug("Markdown auto-lint process completed")
    return linted_content


def _remove_redundant_spaces(content: str) -> str:
    """
    Remove redundant extra spaces from markdown content.
    
    This function implements basic space cleanup:
    - Remove trailing whitespace from lines
    - Remove multiple consecutive spaces within lines (except in code blocks)
    - Preserve intentional spacing in code blocks and inline code
    
    Args:
        content: Markdown content to process
        
    Returns:
        Content with redundant spaces removed
    """
    lines = content.splitlines(keepends=True)
    processed_lines = []
    in_code_block = False
    
    for line in lines:
        # Track code block state
        stripped_line = line.strip()
        
        # Check for code block delimiters
        if stripped_line.startswith('```') or stripped_line.startswith('~~~'):
            if not in_code_block:
                # Starting a code block
                in_code_block = True
            elif in_code_block:
                # Ending a code block
                in_code_block = False
        
        # Process the line based on whether we're in a code block
        if in_code_block:
            # In code blocks, only remove trailing whitespace but preserve line ending
            processed_line = line.rstrip() + ('\n' if line.endswith('\n') else '')
        else:
            # Outside code blocks, remove trailing whitespace and compress multiple spaces
            line_without_ending = line.rstrip('\r\n')
            line_ending = line[len(line_without_ending):]
            processed_line = line_without_ending.rstrip()
            
            # Simple approach: compress multiple spaces to single spaces
            # but preserve leading indentation and inline code
            import re
            
            # Handle leading whitespace separately
            leading_match = re.match(r'^(\s*)', processed_line)
            leading_spaces = leading_match.group(1) if leading_match else ''
            rest_of_line = processed_line[len(leading_spaces):]
            
            # Simple compression of multiple spaces (preserve inline code roughly)
            if '`' in rest_of_line:
                # Basic preservation of inline code - don't compress spaces inside backticks
                parts = rest_of_line.split('`')
                for i in range(len(parts)):
                    if i % 2 == 0:  # Outside code (even indices)
                        parts[i] = re.sub(r' {2,}', ' ', parts[i])
                    # Odd indices are inside code, leave as-is
                rest_of_line = '`'.join(parts)
            else:
                # No inline code, safe to compress all multiple spaces
                rest_of_line = re.sub(r' {2,}', ' ', rest_of_line)
            
            processed_line = leading_spaces + rest_of_line + line_ending
        
        processed_lines.append(processed_line)
    
    return ''.join(processed_lines)


def _fix_line_endings(content: str) -> str:
    """
    Ensure proper line endings throughout the content.
    
    This function implements requirement 6.3:
    - Standardize line endings to LF (Unix-style)
    - Remove any CR characters that might be present
    
    Args:
        content: Markdown content to process
        
    Returns:
        Content with standardized line endings
    """
    # Replace CRLF with LF, then replace any remaining CR with LF
    content = content.replace('\r\n', '\n')
    content = content.replace('\r', '\n')
    
    return content





def _ensure_single_final_newline(content: str) -> str:
    """
    Ensure the content ends with exactly one newline character.
    
    This function implements part of requirement 6.3 (proper line endings):
    - Remove any trailing whitespace at the end of the document
    - Ensure exactly one newline at the end of the file
    
    Args:
        content: Markdown content to process
        
    Returns:
        Content with proper final newline
    """
    # Remove all trailing whitespace and newlines
    content = content.rstrip()
    
    # Add exactly one newline at the end
    if content:  # Only add newline if content is not empty
        content += '\n'
    
    return content


def apply_markdown_linting(file_path: str, content: str) -> str:
    """
    Apply markdown linting to content and optionally write back to file.
    
    This function implements requirement 6.1: "WHEN translation is completed 
    THEN the system SHALL run auto-lint fixes on all processed markdown files"
    
    Args:
        file_path: Path to the markdown file (for logging purposes)
        content: Markdown content to lint
        
    Returns:
        Linted markdown content
        
    Raises:
        TranslationError: If linting fails
    """
    logger = logging.getLogger('markdown_translator')
    
    try:
        logger.debug(f"Applying markdown linting to {file_path}")
        
        # Apply auto-lint fixes
        linted_content = auto_lint_markdown(content)
        
        # Log if changes were made
        if linted_content != content:
            logger.info(f"Applied markdown formatting fixes to {file_path}")
        else:
            logger.debug(f"No formatting changes needed for {file_path}")
        
        return linted_content
        
    except Exception as e:
        raise TranslationError(f"Failed to apply markdown linting to {file_path}: {e}")


def read_markdown_file_with_git(git_manager: GitFileManager, file_path: str) -> str:
    """
    Read a markdown file using the Git file manager.
    
    Args:
        git_manager: GitFileManager instance
        file_path: Path to the file to read
        
    Returns:
        File content as string
        
    Raises:
        FileProcessingError: If file cannot be read
    """
    operation = git_manager.read_file(file_path)
    
    if not operation.success:
        raise FileProcessingError(f"Failed to read file {file_path}: {operation.error_message}")
    
    return operation.content or ""


def write_markdown_file_with_git(git_manager: GitFileManager, file_path: str, content: str) -> None:
    """
    Write a markdown file using the Git file manager.
    
    Args:
        git_manager: GitFileManager instance
        file_path: Path to the file to write
        content: Content to write
        
    Raises:
        FileProcessingError: If file cannot be written
    """
    operation = git_manager.write_file(file_path, content)
    
    if not operation.success:
        raise FileProcessingError(f"Failed to write file {file_path}: {operation.error_message}")


def rename_markdown_file_with_git(git_manager: GitFileManager, old_path: str, new_path: str) -> None:
    """
    Rename a markdown file using the Git file manager.
    
    Args:
        git_manager: GitFileManager instance
        old_path: Current file path
        new_path: New file path
        
    Raises:
        FileProcessingError: If file cannot be renamed
    """
    operation = git_manager.rename_file(old_path, new_path)
    
    if not operation.success:
        raise FileProcessingError(f"Failed to rename file {old_path} to {new_path}: {operation.error_message}")


def commit_markdown_files_with_git(git_manager: GitFileManager, file_paths: List[str], 
                                 commit_message: str) -> None:
    """
    Commit markdown files using the Git file manager.
    
    Args:
        git_manager: GitFileManager instance
        file_paths: List of file paths to commit
        commit_message: Commit message
        
    Raises:
        GitOperationError: If commit fails
    """
    operation = git_manager.commit_changes(file_paths, commit_message)
    
    if not operation.success:
        raise GitOperationError(f"Failed to commit files {file_paths}: {operation.error_message}")


def process_file_with_language_detection_and_rename(git_manager: GitFileManager, 
                                                  file_path: str) -> Tuple[str, str]:
    """
    Process a .md file with language detection and automatic renaming.
    
    This function implements requirements 2.1, 2.2, and 2.3:
    - Detect language of .md files
    - Rename to .en.md or .ja.md based on detected language
    - Handle special cases like README.md
    
    Args:
        git_manager: GitFileManager instance
        file_path: Path to the .md file to process
        
    Returns:
        Tuple of (final_file_path, detected_language)
        
    Raises:
        FileProcessingError: If file processing fails
        GitOperationError: If Git operations fail
    """
    logger = logging.getLogger('markdown_translator')
    
    try:
        # Read the file content
        content = read_markdown_file_with_git(git_manager, file_path)
        
        # Detect language from filename first
        detected_language = detect_language_from_filename(file_path)
        
        if not detected_language:
            # Detect language from content (requirement 2.1)
            detected_language = detect_language(content)
            logger.info(f"Detected language '{detected_language}' for file {file_path}")
            
            # Handle file renaming based on detected language
            rename_operation = git_manager.handle_file_rename_for_language_detection(
                file_path, detected_language
            )
            
            if rename_operation.success and rename_operation.new_file_path != file_path:
                logger.info(f"Renamed file {file_path} to {rename_operation.new_file_path}")
                return rename_operation.new_file_path, detected_language
            elif not rename_operation.success:
                raise GitOperationError(f"Failed to rename file: {rename_operation.error_message}")
        
        return file_path, detected_language
        
    except (FileProcessingError, GitOperationError):
        raise
    except Exception as e:
        raise FileProcessingError(f"Failed to process file {file_path}: {e}")


def create_or_update_counterpart_file(git_manager: GitFileManager, source_file_path: str, 
                                    counterpart_file_path: str, translated_content: str) -> None:
    """
    Create or update a counterpart file with translated content.
    
    This function implements requirements 3.1, 3.2, and 3.3:
    - Create or update corresponding .ja.md/.en.md files
    - Handle README.md counterparts
    - Apply markdown linting to the content
    
    Args:
        git_manager: GitFileManager instance
        source_file_path: Path to the source file
        counterpart_file_path: Path to the counterpart file to create/update
        translated_content: Translated content to write
        
    Raises:
        FileProcessingError: If file operations fail
        TranslationError: If linting fails
    """
    logger = logging.getLogger('markdown_translator')
    
    try:
        # Apply markdown linting to translated content (requirement 6.1)
        linted_content = apply_markdown_linting(counterpart_file_path, translated_content)
        
        # Write the linted content to the counterpart file
        write_markdown_file_with_git(git_manager, counterpart_file_path, linted_content)
        
        logger.info(f"Successfully created/updated counterpart file: {counterpart_file_path}")
        
    except (FileProcessingError, TranslationError):
        raise
    except Exception as e:
        raise FileProcessingError(f"Failed to create/update counterpart file {counterpart_file_path}: {e}")


def main():
    """
    Main entry point for the translation script.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_configuration(args)
        
        # Set up logging
        logger = setup_logging(config.log_level)
        
        logger.info("Starting markdown translation system")
        logger.info(f"Repository: {args.repository}")
        logger.info(f"PR Number: {args.pr_number}")
        logger.info(f"Working Directory: {args.working_dir}")
        
        if args.retry:
            logger.info("RETRY MODE - Attempting to recover from previous failure")
        
        # Change to working directory
        if args.working_dir != '.':
            os.chdir(args.working_dir)
            logger.info(f"Changed working directory to: {os.getcwd()}")
        
        # Parse JSON inputs
        try:
            files_data = json.loads(args.files)
            ignore_patterns = json.loads(args.ignore_patterns)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON inputs: {e}")
            sys.exit(1)
        
        logger.info(f"Files to process: {len(files_data)}")
        logger.info(f"Ignore patterns: {len(ignore_patterns)}")
        logger.info(f"Configuration: {config}")
        
        # Initialize Git file manager
        try:
            dry_run = getattr(args, 'dry_run', False)
            git_manager = GitFileManager(repository_path=".", dry_run=dry_run)
            logger.info("Git file manager initialized successfully")
        except GitOperationError as e:
            logger.error(f"Failed to initialize Git file manager: {e}")
            if not getattr(args, 'dry_run', False):
                sys.exit(1)
            else:
                logger.warning("Continuing in dry-run mode without Git manager")
                git_manager = None
        
        # Process each file from JSON data
        markdown_files = []
        for file_data in files_data:
            try:
                file_path = file_data['path']
                
                # Check if file should be ignored
                if should_ignore_file(file_path, ignore_patterns):
                    logger.info(f"Skipping ignored file: {file_path}")
                    continue
                
                # Check if file exists and read content using Git manager
                try:
                    if git_manager:
                        content = read_markdown_file_with_git(git_manager, file_path)
                    else:
                        # Dry run mode - use dummy content
                        content = "# Sample Content\nThis is sample markdown content for testing."
                except FileProcessingError as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    continue
                
                # Use provided metadata or detect language
                language = file_data.get('language')
                if not language:
                    language = detect_language_from_filename(file_path)
                    if not language:
                        language = detect_language(content)
                
                # Use provided diff percentage or calculate
                diff_percentage = file_data.get('diff_percentage', 0.0)
                
                # Determine translation mode
                translation_mode = determine_translation_mode(content, "", config)
                if file_data.get('size', len(content)) > config.size_threshold or diff_percentage > config.diff_threshold:
                    translation_mode = 'full'
                else:
                    translation_mode = 'incremental'
                
                # Get counterpart path
                counterpart_paths = get_all_counterpart_paths(file_path, language)
                
                # Create MarkdownFile object
                markdown_file = MarkdownFile(
                    path=file_path,
                    language=language,
                    content=content,
                    diff_percentage=diff_percentage,
                    file_size=file_data.get('size', len(content)),
                    counterpart_path=counterpart_paths[0] if counterpart_paths else "",
                    translation_mode=translation_mode
                )
                
                markdown_files.append(markdown_file)
                
            except Exception as e:
                logger.error(f"Failed to process file {file_data.get('path', 'unknown')}: {e}")
        
        if not markdown_files:
            logger.info("No files to process after filtering")
            return
        
        # Create translation request
        request = TranslationRequest(
            source_repo=args.repository,
            pr_number=args.pr_number,
            files=markdown_files,
            ignore_patterns=ignore_patterns,
            thresholds={
                'size_threshold': config.size_threshold,
                'diff_threshold': config.diff_threshold
            }
        )
        
        # Perform actual translation
        results = process_translation_request(request, config, git_manager)
        
        # Log results
        if results['failed_files']:
            logger.warning(f"Some files failed to process: {len(results['failed_files'])}")
            for failed in results['failed_files']:
                logger.warning(f"  {failed['source_file']}: {failed['error']}")
        
        if results['processed_files']:
            logger.info("Successfully processed files:")
            processed_counterpart_files = []
            for processed in results['processed_files']:
                logger.info(f"  {processed['source_file']} -> {processed['counterpart_file']}")
                processed_counterpart_files.append(processed['counterpart_file'])
            
            # Commit the translated files (requirement 8.6)
            if processed_counterpart_files:
                try:
                    commit_message = f"Auto-translate markdown files for PR #{args.pr_number}"
                    commit_markdown_files_with_git(git_manager, processed_counterpart_files, commit_message)
                    logger.info(f"Successfully committed {len(processed_counterpart_files)} translated files")
                except GitOperationError as e:
                    logger.warning(f"Failed to commit translated files: {e}")
                    raise
        
        # Exit with error if any files failed and this is not a retry
        if results['failed_files'] and not args.retry:
            logger.error("Some files failed to process")
            sys.exit(1)
        
        logger.info("Translation script completed successfully")
        
    except TranslationError as e:
        logger = logging.getLogger('markdown_translator')
        logger.error(f"Translation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger('markdown_translator')
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()