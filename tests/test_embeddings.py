import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.database import Database
from src.models import EmbeddingModel
from src.embeddings import Embeddings
from src.misc import cosine_similarity


# ----- Default Parameters -----
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
    print(emb)
    print("✅ Recompute embedding test passed.")


def test_embeddings_similarity_matrix():
    """Test embedding generation and visualize pairwise cosine similarity as a heatmap."""

    print("\n" + "=" * 60)
    print("TEST: EMBEDDING SIMILARITY MATRIX")
    print("=" * 60)

    # Initialize components
    db = Database(data_dir=DATA_DIR)
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    embeddings = Embeddings(
        embeddings_dir=EMBED_DIR,
        database=db,
        embedding_model=model,
        recompute_embeddings=False  # Use cached if available
    )

    # Get aligned names and embedding matrix
    names, matrix = embeddings.to_matrix()
    n_docs = len(names)

    print(f"Found {n_docs} document(s):")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")

    # Compute cosine similarity matrix
    print(f"\nComputing {n_docs}x{n_docs} cosine similarity matrix...")
    similarity_matrix = cosine_similarity(matrix, matrix)

    # === Plot Heatmap ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=[os.path.basename(n) for n in names],
        yticklabels=[os.path.basename(n) for n in names],
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title("Document Embedding Similarity Matrix\n(all-MiniLM-L6-v2)")
    plt.xlabel("Document")
    plt.ylabel("Document")
    plt.tight_layout()

    # # Save plot
    plot_path = os.path.join(EMBED_DIR, "similarity_heatmap.png")
    os.makedirs(EMBED_DIR, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Similarity heatmap saved to: {plot_path}")
    plt.close()

    print("Similarity matrix test passed!")
    print("=" * 60)


if __name__ == "__main__":
    # test_embeddings_basic()
    # test_embeddings_recompute()
    test_embeddings_similarity_matrix()
