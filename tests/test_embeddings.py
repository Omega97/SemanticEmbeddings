import os
import numpy as np
from src.database import Database
from src.models import EmbeddingModel
from src.embeddings import Embeddings


ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, '..', 'data')
EMBED_DIR = os.path.join(ROOT_DIR, '..', 'embeddings')


def test_embeddings_basic():
    """Test basic embedding generation and loading."""

    # Initialize components
    print(f'\nSetting up ')
    db = Database(data_dir=DATA_DIR)
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    embeddings = Embeddings(embeddings_dir=EMBED_DIR, database=db, embedding_model=model)

    # Verify embeddings were created
    print(f'\nChecking embedding matrix')
    names, matrix = embeddings.to_matrix()
    print(f'Matrix shape: {matrix.shape}')
    assert len(names) == 2
    assert matrix.shape == (2, 384)  # all-MiniLM-L6-v2 outputs 384-dim vectors

    # Check embedding file exists
    print(f'\nVerifying data file')
    embed_file = embeddings.get_data_file_name()
    print(embed_file)
    assert os.path.exists(embed_file)

    print("✅ Basic embedding test passed.")


def test_embeddings_recompute():
    """Test recompute_embeddings=True forces full re-embedding."""

    db = Database(data_dir=DATA_DIR)
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

    # Run with recompute
    emb = Embeddings(embeddings_dir=EMBED_DIR, database=db, embedding_model=model,
                     recompute_embeddings=True)

    print("✅ Recompute embedding test passed.")


if __name__ == "__main__":
    # test_embeddings_basic()
    test_embeddings_recompute()
