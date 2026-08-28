from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from chunker import create_chunks


PDF_PATH = "data/documents/sample.pdf"

MODELS = {
    "BGE-M3": "BAAI/bge-m3",
    "Multilingual-E5-large": "intfloat/multilingual-e5-large",
    "BGE-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2"
}


def create_embedding_index(model_name, model_path, chunks):

    print("\n===================================")
    print("MODEL:", model_name)
    print("===================================")

    print("Loading model...")

    model = SentenceTransformer(model_path)

    print("Model loaded successfully!")

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print("Embedding dimension:", embeddings.shape[1])

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    print("FAISS index created!")
    print("Vectors stored:", index.ntotal)

    return model, index, embeddings


if __name__ == "__main__":

    print("Reading PDF and creating chunks...")

    chunks = create_chunks(PDF_PATH)

    print("Number of chunks:", len(chunks))

    # Test one model at a time.
    # Change this name when you want to test another model.

    model_name = "BGE-M3"
    model_path = MODELS[model_name]

    model, index, embeddings = create_embedding_index(
        model_name,
        model_path,
        chunks
    )

    print("\n===== EMBEDDING PIPELINE COMPLETED =====")