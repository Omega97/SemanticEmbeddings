import os
from src import EmbeddingModel
from src.database import Database


def main():
    # Define data directory (relative to project root)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    if not os.path.exists(data_dir):
        print(f"Error: 'data/' directory not found at {data_dir}")
        print("Please create it and add .txt or .csv files.")
        return

    # Initialize database
    print("🔍 Loading documents from 'data/'...")
    db = Database(folder_path=data_dir)
    docs = list(db.iter_documents())

    if not docs:
        print("No readable documents found in 'data/'. Supported: .txt, .csv")
        return

    print(f"✅ Found {len(docs)} document(s).")

    # Extract texts and paths
    paths = [doc['path'] for doc in docs]
    texts = [doc['text'] for doc in docs]

    # Initialize embedding model
    print("🧠 Loading embedding model...")
    model = EmbeddingModel()  # uses all-MiniLM-L6-v2 by default

    # Generate embeddings
    print("🧮 Generating embeddings...")
    embeddings = model.encode(texts)  # shape: (N, 384)

    # Summary
    print(f"\n📊 Embedding shape: {embeddings.shape}")
    print(f"   → {embeddings.shape[0]} documents")
    print(f"   → {embeddings.shape[1]} dimensions per embedding")

    # Optional: Save embeddings (uncomment if needed)
    # np.save('embeddings.npy', embeddings)
    # with open('paths.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(paths))
    # print("\n💾 Embeddings saved to 'embeddings.npy' and paths to 'paths.txt'")

    # Example: Show first document preview + first 5 dims of its embedding
    print(f"\n📄 Example (first document):")
    print(f"   Path: {paths[0]}")
    print(f"   Text preview: {texts[0][:100]}{'...' if len(texts[0]) > 100 else ''}")
    print(f"   Embedding (first 5 dims): {embeddings[0][:5]}")

    return embeddings, paths


if __name__ == "__main__":
    main()
