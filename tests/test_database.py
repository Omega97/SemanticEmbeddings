""" semanticsearch/tests/test_database.py
"""
import os
from src.database import Database
from src import misc


def test_load_txt_file():
    """Test loading a .txt file from the data/ directory."""
    # Assume there's a data/ folder at the project root
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Initialize the database
    db = Database(data_dir=data_dir)

    # List available documents
    docs = db.list_documents()
    if not docs:
        print("No .txt or .csv files found in data/ directory.")
        return

    # Pick the first document
    first_doc_path = docs[0]
    print(f"\nLoading document: {first_doc_path}")

    # Read its content
    content = db.get_document(first_doc_path)
    print(f"\nContent preview (first 200 chars):\n{content[:200]}...\n")

    # Optional: iterate through all documents
    print("\nIterating through all documents:")
    for doc in db.iter_documents():
        print(f"  - {doc['path']}: {len(doc['text'])} characters")


def test_paths():
    """Test loading a txt file from the data/ directory."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Initialize the database
    db = Database(data_dir=data_dir)

    # search
    for root, _, files in os.walk(db.folder_path):
        for filename in files:
            if misc.get_extension(filename) in db.get_extensions():
                rel_path = db.get_relative_path(root, filename)
                print(rel_path)


if __name__ == "__main__":
    # test_load_txt_file()
    test_paths()
