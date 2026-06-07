#!/usr/bin/env python3
import os
import sys
import json
import argparse

def extract_ipynb_code(filepath):
    """Extracts only the code cells from a Jupyter Notebook."""
    code_blocks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    # 'source' can be a list of strings or a single string
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        code_blocks.append("".join(source))
                    else:
                        code_blocks.append(source)
        return "\n\n".join(code_blocks)
    except Exception as e:
        return f"# Error parsing notebook: {e}"

def extract_py_code(filepath):
    """Reads standard python files."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# Error reading python file: {e}"

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Append code from .py and .ipynb files to a single target file.")
    parser.add_argument("source_dir", help="The target directory to scan.")
    parser.add_argument("output_file", help="The file where the code will be appended.")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_file = args.output_file

    if not os.path.isdir(source_dir):
        print(f"Error: The directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Open the output file in append mode ('a')
    with open(output_file, 'a', encoding='utf-8') as out_file:
        # Walk through the directory and all child directories
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py') or file.endswith('.ipynb'):
                    filepath = os.path.join(root, file)
                    
                    # Calculate the relative path (Child Directory + File Name)
                    rel_path = os.path.relpath(filepath, source_dir)

                    # Write the header format
                    out_file.write(f"\n\n{'='*50}\n")
                    out_file.write(f"### FILE: {rel_path} ###\n")
                    out_file.write(f"{'='*50}\n\n")

                    # Extract content based on file type
                    if file.endswith('.ipynb'):
                        content = extract_ipynb_code(filepath)
                    else:
                        content = extract_py_code(filepath)

                    # Write the code and add a trailing newline
                    out_file.write(content)
                    out_file.write("\n")

    print(f"Success! All .py and .ipynb code has been appended to '{output_file}'.")

if __name__ == "__main__":
    main()
