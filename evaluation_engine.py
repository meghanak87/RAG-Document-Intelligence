import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.%-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_matches(expected, actual):
    expected = normalize_text(expected)
    actual = normalize_text(actual)

    if not expected or not actual:
        return False

    if expected in actual or actual in expected:
        return True

    expected_words = set(expected.split())
    actual_words = set(actual.split())

    overlap = len(expected_words & actual_words) / len(expected_words)

    return overlap >= 0.70


def evaluate_model(model_name, model_path, chunks, evaluation_questions):

    print("\n" + "=" * 70)
    print("EVALUATING:", model_name)
    print("=" * 70)

    start_loading = time.time()

    model = SentenceTransformer(model_path)

    model_load_time = time.time() - start_loading

    texts = [chunk["text"] for chunk in chunks]

    embedding_start = time.time()

    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    document_embeddings = np.asarray(
        document_embeddings,
        dtype="float32"
    )

    embedding_time = time.time() - embedding_start

    dimension = document_embeddings.shape[1]

    top1_correct = 0
    top3_correct = 0
    answer_correct = 0

    similarities = []
    question_results = []

    for item in evaluation_questions:

        question = item["question"]
        expected_answer = item["answer"]
        expected_text = item.get("source_text", "")

        question_embedding = model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype="float32"
        )

        search_start = time.time()

        scores = np.dot(
            document_embeddings,
            question_embedding[0]
        )

        ranked_indices = np.argsort(scores)[::-1]

        retrieval_time = time.time() - search_start

        top1_index = int(ranked_indices[0])

        top3_indices = ranked_indices[
            :min(3, len(ranked_indices))
        ]

        top1_chunk = chunks[top1_index]
        top1_text = top1_chunk["text"]

        source_found_top1 = False
        source_found_top3 = False

        if expected_text:

            expected_normalized = normalize_text(
                expected_text
            )

            retrieved_normalized = normalize_text(
                top1_text
            )

            if (
                expected_normalized in retrieved_normalized
                or retrieved_normalized in expected_normalized
            ):
                source_found_top1 = True

            for idx in top3_indices:

                candidate = normalize_text(
                    chunks[int(idx)]["text"]
                )

                if (
                    expected_normalized in candidate
                    or candidate in expected_normalized
                ):
                    source_found_top3 = True
                    break

        is_answer_correct = answer_matches(
            expected_answer,
            top1_text
        )

        if source_found_top1:
            top1_correct += 1

        if source_found_top3:
            top3_correct += 1

        if is_answer_correct:
            answer_correct += 1

        similarities.append(
            float(scores[top1_index])
        )

        question_results.append({
            "question": question,
            "expected_answer": expected_answer,
            "retrieved_context": top1_text,
            "similarity": float(scores[top1_index]),
            "top1_correct": source_found_top1,
            "top3_correct": source_found_top3,
            "answer_correct": is_answer_correct,
            "page": top1_chunk["page"],
            "retrieval_time": retrieval_time
        })

    total_questions = len(evaluation_questions)

    if total_questions == 0:
        return None

    top1_accuracy = (
        top1_correct / total_questions
    ) * 100

    top3_accuracy = (
        top3_correct / total_questions
    ) * 100

    answer_accuracy = (
        answer_correct / total_questions
    ) * 100

    average_similarity = sum(similarities) / len(similarities)

    average_retrieval_time = (
        sum(
            q["retrieval_time"]
            for q in question_results
        )
        / total_questions
    )

    speed_score = max(
        0,
        100 - (average_retrieval_time * 100)
    )

    overall_score = (
        answer_accuracy * 0.40
        + top1_accuracy * 0.25
        + top3_accuracy * 0.15
        + (average_similarity * 100) * 0.10
        + speed_score * 0.10
    )

    result = {
        "model": model_name,
        "dimension": dimension,
        "model_load_time": model_load_time,
        "embedding_time": embedding_time,
        "retrieval_time": average_retrieval_time,
        "answer_accuracy": answer_accuracy,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "similarity": average_similarity,
        "overall_score": overall_score,
        "question_results": question_results
    }

    print("Answer Accuracy:", round(answer_accuracy, 2), "%")
    print("Top-1 Accuracy:", round(top1_accuracy, 2), "%")
    print("Top-3 Accuracy:", round(top3_accuracy, 2), "%")
    print("Average Similarity:", round(average_similarity, 4))
    print("Overall Score:", round(overall_score, 2))

    return result


def evaluate_all_models(
    chunks,
    evaluation_questions,
    models
):

    results = []

    for model_name, model_path in models.items():

        result = evaluate_model(
            model_name,
            model_path,
            chunks,
            evaluation_questions
        )

        if result:
            results.append(result)

    results.sort(
        key=lambda x: x["overall_score"],
        reverse=True
    )

    return results