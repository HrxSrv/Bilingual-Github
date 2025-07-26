import os

HOOK_SCRIPT = """#!/bin/bash
python3 src/hooks/post_commit.py
"""

def install_hooks():
    hook_path = ".git/hooks/post-commit"
    with open(hook_path, "w") as hook_file:
        hook_file.write(HOOK_SCRIPT)
    os.chmod(hook_path, 0o775)
    print("Post-commit hook installed.")

if __name__ == "__main__":
    install_hooks()
