import os
import tempfile
import numpy as np
from src.database import Database
from src.models import EmbeddingModel
from src.embeddings import Embeddings


def create_test_data(data_dir: str):
    """Create sample .txt files for testing."""
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("This is the first test document.")
    with open(os.path.join(data_dir, "doc2.txt"), "w", encoding="utf-8") as f:
        f.write("This is the second test document.")


def test_embeddings_basic():
    """Test basic embedding generation and loading."""
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        embed_dir = os.path.join(temp_dir, "embeddings")

        # Create test data
        create_test_data(data_dir)

        # Initialize components
        db = Database(folder_path=data_dir)
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
        embeddings = Embeddings(dir_path=embed_dir, database=db, embedding_model=model)

        # Verify embeddings were created
        names, matrix = embeddings.to_matrix()
        assert len(names) == 2
        assert matrix.shape == (2, 384)  # all-MiniLM-L6-v2 outputs 384-dim vectors
        assert set(names) == {"doc1.txt", "doc2.txt"}

        # Check embedding file exists
        embed_file = embeddings.get_data_file_name()
        assert os.path.exists(embed_file)

        print("✅ Basic embedding test passed.")


def test_embeddings_incremental():
    """Test that new documents trigger new embeddings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        embed_dir = os.path.join(temp_dir, "embeddings")

        # Step 1: Start with one document
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "doc1.txt"), "w", encoding="utf-8") as f:
            f.write("Initial document.")

        db = Database(folder_path=data_dir)
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
        embeddings = Embeddings(dir_path=embed_dir, database=db, embedding_model=model)

        names, _ = embeddings.to_matrix()
        assert len(names) == 1
        assert "doc1.txt" in names

        # Step 2: Add a second document
        with open(os.path.join(data_dir, "doc2.txt"), "w", encoding="utf-8") as f:
            f.write("New document added later.")

        # Re-initialize (simulates next run)
        db2 = Database(folder_path=data_dir)
        embeddings2 = Embeddings(dir_path=embed_dir, database=db2, embedding_model=model)

        names2, matrix2 = embeddings2.to_matrix()
        assert len(names2) == 2
        assert matrix2.shape == (2, 384)
        assert set(names2) == {"doc1.txt", "doc2.txt"}

        print("✅ Incremental embedding test passed.")


def test_embeddings_recompute():
    """Test recompute_embeddings=True forces full re-embedding."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        embed_dir = os.path.join(temp_dir, "embeddings")

        create_test_data(data_dir)

        db = Database(folder_path=data_dir)
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

        # First run
        emb1 = Embeddings(dir_path=embed_dir, database=db, embedding_model=model)
        _, mat1 = emb1.to_matrix()

        # Second run with recompute
        emb2 = Embeddings(dir_path=embed_dir, database=db, embedding_model=model, recompute_embeddings=True)
        _, mat2 = emb2.to_matrix()

        # Should have same shape and (approximately) same values
        assert mat1.shape == mat2.shape
        assert np.allclose(mat1, mat2, atol=1e-6)

        print("✅ Recompute embedding test passed.")


if __name__ == "__main__":
    test_embeddings_basic()
    test_embeddings_incremental()
    test_embeddings_recompute()
    print("🎉 All Embeddings tests passed!")
