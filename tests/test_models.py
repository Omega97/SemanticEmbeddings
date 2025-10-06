from src.models import EmbeddingModel


def test_model(query):
    print('Loading model...')
    model = EmbeddingModel()

    print('Computing Embedding...')
    query_emb = model.encode([query])[0]
    print(query_emb)


if __name__ == "__main__":
    test_model(query="The full story behind the invention")
