#!/usr/bin/env python3
"""
Script to update model saving format in the codebase to use the native Keras format
without the deprecated save_format parameter.
"""
import os
import re
import sys

def update_file(file_path):
    """Update model.save calls in a file to use the native Keras format."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match model.save with save_format parameter
    pattern = r'model\.save\(([^,]+),\s*save_format\s*=\s*[\'"]keras[\'"]\)'
    
    # Replace with model.save without the save_format parameter
    replacement = r'model.save(\1)  # The .keras extension automatically uses the native Keras format'
    
    # Perform the replacement
    new_content = re.sub(pattern, replacement, content)
    
    # Only write to the file if changes were made
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    
    return False

def find_and_update_files(directory, extensions=None):
    """Find and update all Python files in the directory."""
    if extensions is None:
        extensions = ['.py']
    
    updated_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                if update_file(file_path):
                    updated_files.append(file_path)
    
    return updated_files

if __name__ == "__main__":
    # Directory to search for files
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    
    print(f"Searching for Python files in {directory}")
    updated_files = find_and_update_files(directory)
    
    if updated_files:
        print(f"Updated {len(updated_files)} files:")
        for file in updated_files:
            print(f"  - {file}")
    else:
        print("No files were updated.")
