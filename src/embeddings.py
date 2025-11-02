import numpy as np
import os
from typing import List, Tuple
from src.database import Database
from src.models import EmbeddingModel


class Embeddings:
    """
    Manages persistent storage and incremental updates of document embeddings.

    Loads existing embeddings from disk (model-specific filename), identifies missing
    documents, computes new embeddings only for those, and saves the full set back.
    Embeddings are stored as a dictionary mapping relative file paths to vectors.
    Supports extension, removal, retrieval, and conversion to a matrix format.
    """
    def __init__(
        self,
        embeddings_dir: str,
        database: Database,
        embedding_model: EmbeddingModel,
        recompute_embeddings: bool = False
    ):
        """
        Initialize the Embeddings manager.

        Args:
            embeddings_dir (str): Directory to store/load embedding files.
            database (Database): Document source.
            embedding_model (EmbeddingModel): Model to generate embeddings.
            recompute_embeddings (bool): If True, ignores existing embeddings and recomputes all.
        """
        self.embeddings_path = embeddings_dir
        self.database = database
        self.embedding_model = embedding_model
        self.recompute_embeddings = recompute_embeddings
        self.data = {}  # {relative_path: embedding_vector}
        self.all_paths = []
        self.missing_embeddings = []

        self._load()

    def get_data_file_name(self) -> str:
        """Generate a model-specific embedding filename."""
        model_name = self.embedding_model.model_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.embeddings_path, f"_embeddings_by_{model_name}.npz")

    def _setup(self):
        """Ensure the embedding directory exists."""
        os.makedirs(self.embeddings_path, exist_ok=True)
        # Do NOT create an empty .npz file here — loading handles missing files gracefully.

    def _load_file_paths_from_database(self):
        """Load current document list from the database."""
        self.all_paths = list(self.database.list_documents())

    def _load_embeddings_from_file(self):
        """Load existing embeddings from disk, matching only current documents."""
        file_path = self.get_data_file_name()
        if not os.path.exists(file_path):
            print(f"No existing embeddings found at {file_path}. Starting fresh.")
            self.data = {}
            return

        try:
            print(f"Loading embeddings from {file_path}")
            loaded_data = np.load(file_path, allow_pickle=True)
            # Keep only entries that are still in the database
            self.data = {
                key: loaded_data[key]
                for key in loaded_data.keys()
                if key in self.all_paths
            }
            print(f"Loaded {len(self.data)} embeddings.")
        except Exception as e:
            print(f"Failed to load embeddings ({e}). Starting fresh.")
            self.data = {}

    def _find_missing_embeddings(self):
        """Identify documents that lack embeddings."""
        existing_keys = set(self.data.keys())
        all_paths_set = set(self.all_paths)
        self.missing_embeddings = list(all_paths_set - existing_keys)

    def _compute_missing_embeddings(self):
        """Encode and store embeddings for missing documents."""
        if not self.missing_embeddings:
            print("No new documents to embed.")
            return

        print(f"Computing embeddings for {len(self.missing_embeddings)} new document(s)...")
        texts = [self.database.get_document(path) for path in self.missing_embeddings]
        embeddings = self.embedding_model.encode(texts)
        self.extend(self.missing_embeddings, embeddings)
        self._save()

    def _load(self):
        """Orchestrate full loading workflow."""
        self._setup()
        self._load_file_paths_from_database()

        if self.recompute_embeddings:
            self.data = {}
            self.missing_embeddings = self.all_paths
        else:
            self._load_embeddings_from_file()
            self._find_missing_embeddings()

        self._compute_missing_embeddings()

    def _save(self):
        """Save current embeddings to disk."""
        np.savez(file=self.get_data_file_name(), **self.data)
        print(f"Saved {len(self.data)} embeddings to {self.get_data_file_name()}")

    def extend(self, names: List[str], embeddings: np.ndarray):
        """Add new embeddings to the store."""
        if len(names) != embeddings.shape[0]:
            raise ValueError("Number of names must match number of embedding rows.")

        for name, emb in zip(names, embeddings):
            if name in self.data:
                print(f"Warning: Overwriting existing embedding for '{name}'.")
            self.data[name] = emb

        print(f"Extended store to {len(self.data)} entries.")

    def remove(self, names: List[str]):
        """Remove specified documents from the embedding store."""
        missing = [name for name in names if name not in self.data]
        if missing:
            raise KeyError(f"Cannot remove non-existent entries: {missing}")

        for name in names:
            del self.data[name]
        print(f"Removed {len(names)} entries.")

    def get(self, name: str) -> np.ndarray:
        """Retrieve embedding for a specific document."""
        if name not in self.data:
            raise KeyError(f"No embedding found for '{name}'.")
        return self.data[name]

    def get_names(self) -> List[str]:
        """Get list of all embedded document paths."""
        return list(self.data.keys())

    def to_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Return aligned list of paths and embedding matrix.
        Order is insertion-ordered (Python 3.7+ dict guarantee).
        """
        names = list(self.data.keys())
        embeddings = np.array(list(self.data.values()))
        return names, embeddings
