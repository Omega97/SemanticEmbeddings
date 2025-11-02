""" semanticsearch/src/embedding.py

This module contains the EmbeddingModel class, which is used to convert
text inputs into vector embeddings. The class uses the SentenceTransformer
library to encode the text inputs.

Useful models:
- all-MiniLM-L6-v2 (small & efficient)
- paraphrase-MiniLM-L6-v2 (small & efficient)
- stsb-distilroberta-base-v2 (small)
- paraphrase-mpnet-base-v2 (medium)
- paraphrase-TinyBERT-L6-v2 (medium)
- paraphrase-distilroberta-base-v1 (medium)
- paraphrase-multilingual-mpnet-base-v2 (multilingual)
- paraphrase-xlm-r-multilingual-v1 (multilingual)
"""
import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


# ----- Default Parameters -----
ROOT_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = os.path.join(ROOT_DIR, '..', 'model_cache')


class EmbeddingModel:
    """
    Wraps a SentenceTransformer model to generate normalized text embeddings.

    Automatically caches downloaded models locally to avoid repeated downloads.
    Supports loading from cache only or falling back to download if needed.
    Default model: 'all-MiniLM-L6-v2' (384-dimensional embeddings).
    """
    def __init__(self, model_name=DEFAULT_MODEL_NAME, use_cache_only=False):
        self.model_name = model_name
        self.use_cache_only = use_cache_only
        self.model = None
        self._load()

    def _load(self):
        cache_folder = DEFAULT_CACHE_DIR
        os.makedirs(cache_folder, exist_ok=True)

        print(f'Loading model "{self.model_name}"...')

        try:
            # Try loading from cache only first with local_files_only=True
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_folder,
                local_files_only=True
            )
            print(f'Loaded from cache: {cache_folder}')
        except Exception:
            print('Cache not found. Downloading model (one-time only)...')
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_folder
            )
            print(f'Downloaded and cached in: {cache_folder}')

    def encode(self, texts: List[str], normalize_embeddings=True) -> np.ndarray:
        """
        Converts a list of text inputs into vector embeddings.

        Args:
            texts (list of str): List of text inputs.
            normalize_embeddings (bool): Whether to normalize the embeddings.

        Returns:
            np.ndarray: Array, the embeddings matrix (shape: [num_texts, embedding_dim]).
        """
        return np.array(self.model.encode(texts,
                                          normalize_embeddings=normalize_embeddings,
                                          convert_to_numpy=True))

    def __repr__(self):
        return f'{type(self)}({self.model_name})'
