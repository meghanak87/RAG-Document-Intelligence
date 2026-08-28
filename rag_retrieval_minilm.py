from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import time


# 1. Load document
with open("data/sample.txt", "r", encoding="utf-8") as file:
    document = file.read()


# 2. Create chunks
chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]

print("Number of chunks:", len(chunks))


# 3. Load questions
with open("data/evaluation/questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

print("Number of questions:", len(questions))


# 4. Load MiniLM
print("\nLoading all-MiniLM-L6-v2...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("MiniLM loaded successfully!")


# 5. Create document embeddings
print("\nCreating document embeddings...")

start_time = time.time()

document_embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)

embedding_time = time.time() - start_time

print(
    "Embedding dimension:",
    document_embeddings.shape[1]
)

print(
    "Document embedding time:",
    round(embedding_time, 3),
    "seconds"
)


# 6. Create FAISS index
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)

print("FAISS vectors:", index.ntotal)


# 7. Relevance function
def is_relevant(chunk, answer):

    answer_words = [
        word.lower()
        for word in answer.split()
        if len(word) > 4
    ]

    matches = sum(
        1
        for word in answer_words
        if word in chunk.lower()
    )

    return matches >= 2


# 8. Evaluation
precision_at_1_count = 0
recall_at_3_count = 0

reciprocal_ranks = []
retrieval_times = []


# 9. Test all questions
for question_number, item in enumerate(
    questions,
    start=1
):

    query = item["question"]
    answer = item["answer"]

    start_time = time.time()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        3
    )

    retrieval_time = time.time() - start_time

    retrieval_times.append(retrieval_time)

    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]


    # Find relevant ranks
    relevant_positions = []

    for position, retrieved_chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        if is_relevant(
            retrieved_chunk,
            answer
        ):
            relevant_positions.append(position)


    # Precision@1
    if 1 in relevant_positions:
        precision_at_1_count += 1


    # Recall@3
    if len(relevant_positions) > 0:
        recall_at_3_count += 1


    # MRR
    if len(relevant_positions) > 0:

        first_relevant_rank = min(
            relevant_positions
        )

        reciprocal_ranks.append(
            1 / first_relevant_rank
        )

    else:

        reciprocal_ranks.append(0)


    print(
        f"Question {question_number}: "
        f"Top-1 score = {scores[0][0]:.4f} | "
        f"Relevant rank = "
        f"{relevant_positions if relevant_positions else 'None'}"
    )


# 10. Final results
total_questions = len(questions)

precision_at_1 = (
    precision_at_1_count /
    total_questions
)

recall_at_3 = (
    recall_at_3_count /
    total_questions
)

mrr = sum(reciprocal_ranks) / total_questions

average_retrieval_time = (
    sum(retrieval_times) /
    len(retrieval_times)
)


# 11. Display results
print("\n---------- RESULTS ----------")

print("Model: all-MiniLM-L6-v2")

print(
    "Precision@1:",
    round(precision_at_1, 4)
)

print(
    "Recall@3:",
    round(recall_at_3, 4)
)

print(
    "MRR:",
    round(mrr, 4)
)

print(
    "Average retrieval time:",
    round(average_retrieval_time, 4),
    "seconds"
)

print(
    "Embedding dimension:",
    document_embeddings.shape[1]
)

print("\n===================================")
print("MINILM TEST COMPLETED")
print("===================================")