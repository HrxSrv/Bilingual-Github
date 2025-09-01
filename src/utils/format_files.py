#!/usr/bin/env python3
"""
Standalone formatting utility for markdown files.
Applied as a final step in the translation process.
"""

import sys
import os
import re
from pathlib import Path


def apply_formatting_fixes(file_path):
    """Apply comprehensive formatting fixes to a markdown file"""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return False
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content:
            print(f"Skipping empty file: {file_path}")
            return False
        
        original_content = content
        
        # Fix 1: Remove trailing whitespace from all lines
        lines = content.splitlines()
        fixed_lines = [line.rstrip() for line in lines]
        content = '\n'.join(fixed_lines)
        
        # Fix 2: Ensure proper end-of-file handling
        if content and not content.endswith('\n'):
            content += '\n'
        elif content.endswith('\n\n'):
            # Remove excessive newlines at end, keep exactly one
            content = content.rstrip('\n') + '\n'
        
        # Fix 3: Normalize excessive blank lines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix 4: Clean up markdown-specific formatting issues
        content = fix_markdown_formatting(content)
        
        # Write back if any changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Applied formatting fixes to: {file_path}")
            return True
        else:
            print(f"No formatting changes needed for: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error formatting {file_path}: {e}")
        return False


def fix_markdown_formatting(content):
    """Apply markdown-specific formatting fixes"""
    
    # Fix spacing around headers
    content = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', content)
    content = re.sub(r'(#{1,6}.*)\n([^\n#])', r'\1\n\n\2', content)
    
    # Fix spacing around horizontal rules
    content = re.sub(r'\n([-*_]{3,})\n', r'\n\n\1\n\n', content)
    
    # Fix spacing around code blocks
    content = re.sub(r'\n(```[^\n]*)\n', r'\n\n\1\n', content)
    content = re.sub(r'\n(```)\n', r'\n\1\n\n', content)
    
    # Clean up any excessive spacing we may have created
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Fix list formatting - ensure proper spacing
    content = re.sub(r'\n([*\-+]\s)', r'\n\n\1', content)
    content = re.sub(r'\n(\d+\.\s)', r'\n\n\1', content)
    
    # Clean up excessive spacing again after list fixes
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content


def format_multiple_files(file_paths):
    """Format multiple markdown files"""
    formatted_count = 0
    
    for file_path in file_paths:
        file_path = file_path.strip()
        if not file_path:
            continue
            
        # Only process markdown files
        if not file_path.endswith('.md'):
            print(f"Skipping non-markdown file: {file_path}")
            continue
            
        if apply_formatting_fixes(file_path):
            formatted_count += 1
    
    print(f"Formatting complete: {formatted_count} files modified")
    return formatted_count


def main():
    """Main function for standalone execution"""
    if len(sys.argv) < 2:
        print("Usage: python format_files.py <file1.md> [file2.md] ...")
        print("   or: python format_files.py --all")
        sys.exit(1)
    
    if sys.argv[1] == "--all":
        # Find all markdown files in current directory and subdirectories
        markdown_files = []
        for root, dirs, files in os.walk('.'):
            # Skip common ignored directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'dist', 'build']]
            
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.relpath(os.path.join(root, file))
                    markdown_files.append(file_path)
        
        if not markdown_files:
            print("No markdown files found")
            sys.exit(0)
            
        print(f"Found {len(markdown_files)} markdown files to format")
        format_multiple_files(markdown_files)
        
    else:
        # Process specific files provided as arguments
        file_paths = sys.argv[1:]
        format_multiple_files(file_paths)


if __name__ == "__main__":
    main()