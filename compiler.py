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
    
    # New argument for depth toggle
    parser.add_argument(
        "-d", "--depth", 
        type=int, 
        default=-1, 
        help="Maximum folder depth. 0 = source directory only. 1 = source + immediate subfolders. Default is infinite."
    )
    
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    output_file = args.output_file
    max_depth = args.depth

    if not os.path.isdir(source_dir):
        print(f"Error: The directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Open the output file in append mode ('a')
    with open(output_file, 'a', encoding='utf-8') as out_file:
        # Walk through the directory
        for root, dirs, files in os.walk(source_dir):
            
            # Calculate current depth
            if max_depth != -1:
                rel_path = os.path.relpath(root, source_dir)
                if rel_path == '.':
                    current_depth = 0
                else:
                    # Count the directory separators to determine depth
                    current_depth = len(rel_path.split(os.sep))
                
                # If we've reached the max depth, clear the 'dirs' list.
                # This stops os.walk from descending into child directories.
                if current_depth >= max_depth:
                    dirs[:] = [] 

            for file in files:
                if file.endswith('.py') or file.endswith('.ipynb'):
                    filepath = os.path.join(root, file)
                    
                    # Calculate the relative path for the header
                    rel_filepath = os.path.relpath(filepath, source_dir)

                    # Write the header format
                    out_file.write(f"\n\n{'='*50}\n")
                    out_file.write(f"### FILE: {rel_filepath} ###\n")
                    out_file.write(f"{'='*50}\n\n")

                    # Extract content based on file type
                    if file.endswith('.ipynb'):
                        content = extract_ipynb_code(filepath)
                    else:
                        content = extract_py_code(filepath)

                    # Write the code and add a trailing newline
                    out_file.write(content)
                    out_file.write("\n")

    print(f"Success! Code appended to '{output_file}'.")

if __name__ == "__main__":
    main()
