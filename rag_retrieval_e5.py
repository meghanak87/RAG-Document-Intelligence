from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load document
with open("data/sample.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Create chunks
chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]

print("Number of chunks:", len(chunks))

# Load E5 model
print("\nLoading Multilingual-E5-large...")

model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Multilingual-E5-large loaded successfully!")

# Create document embeddings
print("\nCreating document embeddings...")

document_embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)

print("Document embeddings created!")
print("Embedding shape:", document_embeddings.shape)

# Create FAISS index
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)

print("FAISS index created!")
print("Vectors stored:", index.ntotal)

# Query
query = "What are the three main stages of a RAG system?"

print("\nQuery:")
print(query)

# Create query embedding
query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)

# Search
top_k = 3

scores, indices = index.search(
    query_embedding,
    top_k
)

# Display results
print("\n===== E5 RETRIEVED RESULTS =====")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):
    print(f"\nRank {rank}")
    print("Similarity score:", round(float(score), 4))
    print("Chunk:")
    print(chunks[idx])

print("\n===== E5 TEST COMPLETED =====")