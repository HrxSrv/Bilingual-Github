#!/usr/bin/env python3
"""
Simple file formatting utility for markdown files
"""

import sys
import os
import re


def format_markdown_file(file_path):
    """Apply basic formatting fixes to a markdown file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
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
            print(f"Formatted: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
        
    except Exception as e:
        print(f"Error formatting {file_path}: {e}")
        return False


def main():
    """Format multiple markdown files"""
    if len(sys.argv) < 2:
        print("Usage: python format_files.py <file1.md> [file2.md] ...")
        sys.exit(1)
    
    files = sys.argv[1:]
    formatted_count = 0
    
    for file_path in files:
        if file_path.endswith('.md'):
            if format_markdown_file(file_path):
                formatted_count += 1
        else:
            print(f"Skipping non-markdown file: {file_path}")
    
    print(f"Formatted {formatted_count} files")


if __name__ == "__main__":
    main()