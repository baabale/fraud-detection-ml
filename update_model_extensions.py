#!/usr/bin/env python3
"""
Script to update model file extensions from .h5 to .keras across the codebase.
This helps eliminate the HDF5 format deprecation warnings in TensorFlow/Keras.
"""
import os
import re
import sys

def update_file(file_path):
    """Update .h5 extensions to .keras in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match .h5 file extensions in model paths
    pattern = r'([\'"].*_model.*?)\.h5([\'"])'
    
    # Replace with .keras extension
    replacement = r'\1.keras\2'
    
    # Perform the replacement
    new_content = re.sub(pattern, replacement, content)
    
    # Also handle non-quoted paths
    pattern2 = r'([\s\(][a-zA-Z0-9_/\\.-]*_model.*?)\.h5([\s\),])'
    replacement2 = r'\1.keras\2'
    new_content = re.sub(pattern2, replacement2, new_content)
    
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
