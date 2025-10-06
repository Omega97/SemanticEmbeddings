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
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


# ----- Default Parameters -----
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    The EmbeddingModel class is used to convert text inputs into vector embeddings.
    """
    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        """
        Initialize the embedding model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        print(f'Loading embedding model {model_name}')
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

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
