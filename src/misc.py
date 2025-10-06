import csv
import numpy as np
import os


def read_txt_file(file_path) -> str:
    """Read txt file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_csv_file(file_path: str, n_lines: int = 5) -> str:
    """
    Reads the first 'n' lines of a CSV file using the csv module and returns them as text.

    Args:
        file_path (str): The path to the CSV file.
        n_lines (int): The number of lines to read.

    Returns:
        str: The first 'n' lines of the CSV file as text.
    """
    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i >= n_lines:
                    break
                result.append(','.join(row) + '\n')  # Join row elements with commas
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    return '\n'.join(result)


def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def get_extension(path: str) -> str:
    """
    Gets the file extension using os.path.splitext and returns it in lowercase.
    """
    root, ext = os.path.splitext(path)
    return ext[1:].lower()  # Remove the leading dot and convert to lowercase
