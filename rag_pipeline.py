from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from chunker import create_chunks
from google import genai
import os


# ==============================
# SETTINGS
# ==============================

PDF_PATH = "data/documents/sample.pdf"

MODELS = {
    "BGE-M3": "BAAI/bge-m3",
    "Multilingual-E5-large": "intfloat/multilingual-e5-large",
    "BGE-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2"
}


# ==============================
# GEMINI
# ==============================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is not set.")
    exit()

client = genai.Client(api_key=api_key)


# ==============================
# LOAD PDF
# ==============================

print("Reading PDF...")

chunks = create_chunks(PDF_PATH)

print("Number of chunks:", len(chunks))


# ==============================
# SELECT MODEL
# ==============================

print("\nAvailable Embedding Models:")

model_names = list(MODELS.keys())

for i, name in enumerate(model_names, start=1):
    print(f"{i}. {name}")

choice = int(input("\nSelect embedding model (1-4): "))

model_name = model_names[choice - 1]
model_path = MODELS[model_name]

print("\nSelected model:", model_name)


# ==============================
# LOAD MODEL
# ==============================

print("Loading embedding model...")

model = SentenceTransformer(model_path)

print("Model loaded successfully!")


# ==============================
# CREATE DOCUMENT EMBEDDINGS
# ==============================

texts = [chunk["text"] for chunk in chunks]

print("\nCreating document embeddings...")

document_embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

document_embeddings = np.asarray(
    document_embeddings,
    dtype="float32"
)

print(
    "Embedding dimension:",
    document_embeddings.shape[1]
)


# ==============================
# FAISS
# ==============================

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)

print("FAISS index created!")
print("Vectors stored:", index.ntotal)


# ==============================
# QUESTIONS
# ==============================

while True:

    question = input(
        "\nAsk a question about the PDF "
        "(or type 'exit'): "
    )

    if question.lower() == "exit":
        break


    # ==============================
    # QUERY EMBEDDING
    # ==============================

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # ==============================
    # RETRIEVAL
    # ==============================

    k = min(3, len(chunks))

    scores, indices = index.search(
        query_embedding,
        k
    )


    # ==============================
    # BUILD CONTEXT
    # ==============================

    retrieved_chunks = []

    for rank, index_number in enumerate(
        indices[0],
        start=1
    ):

        chunk = chunks[index_number]

        retrieved_chunks.append(
            f"""
Source: sample.pdf
Page: {chunk['page']}
Rank: {rank}
Similarity: {scores[0][rank-1]:.4f}

Content:
{chunk['text']}
"""
        )


    context = "\n".join(retrieved_chunks)


    print("\n===== RETRIEVED CONTEXT =====")
    print(context)


    # ==============================
    # GEMINI ANSWER
    # ==============================

    prompt = f"""
You are an answer-generation system in a RAG application.

Answer the user's question using ONLY the
retrieved context below.

If the answer is not present in the context,
say: "The information is not available in the document."

Give a short and specific answer.

Retrieved context:
{context}

User question:
{question}
"""


    print("\nGenerating answer...")

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    print("\n===== FINAL ANSWER =====")
    print(response.text)

    print("\n===== SOURCE =====")
    print("Document: sample.pdf")
    print("Retrieved page(s):", 
          sorted(set(chunks[i]["page"] for i in indices[0])))


print("\n===== RAG PIPELINE COMPLETED =====")