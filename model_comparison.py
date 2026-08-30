import re
import time
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# EMBEDDING MODELS
# =========================================================

MODELS = {
    "BGE-M3": "BAAI/bge-m3",
    "Multilingual-E5-large": "intfloat/multilingual-e5-large",
    "BGE-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "Nomic-Embed": "nomic-ai/nomic-embed-text-v1.5",
}


# =========================================================
# MODEL LOADING
# =========================================================

@lru_cache(maxsize=8)
def load_model(model_name):
    model_id = MODELS[model_name]

    # Nomic uses custom model code
    if model_name == "Nomic-Embed":
        return SentenceTransformer(
            model_id,
            trust_remote_code=True,
        )

    return SentenceTransformer(model_id)


# =========================================================
# TEXT HELPERS
# =========================================================

def _text(chunk):
    if isinstance(chunk, dict):
        return str(chunk.get("text", "")).strip()

    return str(chunk).strip()


def _page(chunk):
    if isinstance(chunk, dict):
        return chunk.get("page", 1)

    return 1


# =========================================================
# EMBEDDING
# =========================================================

def _encode(model, texts, model_name, is_query=False):
    """
    Generate normalized embeddings.

    E5:
        query: / passage:

    Nomic:
        search_query: / search_document:

    Other models:
        no special prefix.
    """

    texts = [str(x) for x in texts]

    if model_name == "Multilingual-E5-large":

        prefix = "query: " if is_query else "passage: "

        texts = [
            prefix + text
            for text in texts
        ]

    elif model_name == "Nomic-Embed":

        prefix = (
            "search_query: "
            if is_query
            else "search_document: "
        )

        texts = [
            prefix + text
            for text in texts
        ]

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# =========================================================
# DOCUMENT EMBEDDING CACHE
# =========================================================

@lru_cache(maxsize=32)
def _cached_document_embeddings(
    model_name,
    chunks_key,
    texts_tuple,
):

    model = load_model(model_name)

    embeddings = _encode(
        model,
        list(texts_tuple),
        model_name,
        is_query=False,
    )

    return np.asarray(
        embeddings,
        dtype="float32",
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_with_model(
    model_name,
    question,
    chunks,
    top_k=3,
):

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    valid = [
        (i, _text(c))
        for i, c in enumerate(chunks)
        if _text(c)
    ]

    if not valid:
        return None

    original_ids = [
        item[0]
        for item in valid
    ]

    texts = [
        item[1]
        for item in valid
    ]

    model = load_model(model_name)

    # Different documents receive different cache keys.
    chunks_key = hash(tuple(texts))

    # -----------------------------------------------------
    # Document embeddings
    # -----------------------------------------------------

    t0 = time.perf_counter()

    doc_embeddings = _cached_document_embeddings(
        model_name,
        chunks_key,
        tuple(texts),
    )

    embedding_time = (
        time.perf_counter() - t0
    )

    # -----------------------------------------------------
    # FAISS similarity search
    # -----------------------------------------------------

    index = faiss.IndexFlatIP(
        doc_embeddings.shape[1]
    )

    index.add(doc_embeddings)

    t1 = time.perf_counter()

    q_embedding = np.asarray(
        _encode(
            model,
            [question],
            model_name,
            is_query=True,
        ),
        dtype="float32",
    )

    k = min(
        top_k,
        len(texts),
    )

    scores, ids = index.search(
        q_embedding,
        k,
    )

    retrieval_time = (
        time.perf_counter() - t1
    )

    retrieved = []

    for rank, local_id in enumerate(
        ids[0],
        start=1,
    ):

        local_id = int(local_id)

        if local_id < 0:
            continue

        original_id = original_ids[
            local_id
        ]

        chunk = chunks[
            original_id
        ]

        retrieved.append(
            {
                "rank": rank,
                "score": float(
                    scores[0][rank - 1]
                ),
                "page": _page(chunk),
                "text": _text(chunk),
            }
        )

    if not retrieved:
        return None

    return {
        "model": model_name,
        "dimension": int(
            doc_embeddings.shape[1]
        ),
        "similarity": retrieved[0]["score"],
        "top3_score": float(
            np.mean(
                [
                    item["score"]
                    for item in retrieved
                ]
            )
        ),
        "embedding_time": embedding_time,
        "retrieval_time": retrieval_time,
        "total_time": (
            embedding_time
            + retrieval_time
        ),
        "page": retrieved[0]["page"],
        "source": retrieved[0]["text"],
        "retrieved": retrieved,
    }


# =========================================================
# ANSWER NORMALIZATION
# =========================================================

def normalize_answer(text):

    text = str(
        text or ""
    ).lower()

    text = text.replace(
        "–",
        "-"
    ).replace(
        "—",
        "-"
    )

    text = re.sub(
        r"https?://",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9+#.%/&-]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


# =========================================================
# EVIDENCE MATCHING
# =========================================================

def answer_match(
    source_text,
    expected_answer,
):

    expected = normalize_answer(
        expected_answer
    )

    source = normalize_answer(
        source_text
    )

    if not expected or not source:
        return False

    # Exact evidence match
    if expected in source:
        return True

    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "what",
        "which",
        "who",
        "where",
        "when",
        "this",
        "that",
    }

    expected_tokens = [
        token
        for token in expected.split()
        if len(token) > 1
        and token not in stop
    ]

    source_tokens = set(
        token
        for token in source.split()
        if len(token) > 1
        and token not in stop
    )

    if not expected_tokens:
        return False

    overlap = (
        sum(
            token in source_tokens
            for token in expected_tokens
        )
        / len(expected_tokens)
    )

    return overlap >= 0.70


# =========================================================
# SINGLE MODEL EVALUATION
# =========================================================

def evaluate_model(
    model_name,
    chunks,
    questions,
):

    valid_questions = [
        q
        for q in questions
        if str(
            q.get("question", "")
        ).strip()
        and str(
            q.get("answer", "")
        ).strip()
    ]

    if not valid_questions:
        return None

    similarities = []
    top3_similarities = []
    times = []

    top1_matches = []
    top3_matches = []

    details = []

    for item in valid_questions:

        question = item["question"]
        expected = item["answer"]

        result = retrieve_with_model(
            model_name,
            question,
            chunks,
            top_k=3,
        )

        if result is None:

            details.append(
                {
                    "question": question,
                    "expected": expected,
                    "similarity": 0.0,
                    "top3_similarity": 0.0,
                    "page": None,
                    "time": 0.0,
                    "source": (
                        "No evidence retrieved."
                    ),
                    "evidence_match": False,
                    "top3_evidence_match": False,
                }
            )

            top1_matches.append(False)
            top3_matches.append(False)

            continue

        top1_similarity = float(
            result["similarity"]
        )

        top3_similarity = float(
            result["top3_score"]
        )

        total_time = float(
            result["total_time"]
        )

        top1_match = answer_match(
            result["source"],
            expected,
        )

        # Check whether ANY of the top-3
        # retrieved chunks contain evidence.
        top3_match = any(
            answer_match(
                retrieved["text"],
                expected,
            )
            for retrieved
            in result["retrieved"]
        )

        similarities.append(
            top1_similarity
        )

        top3_similarities.append(
            top3_similarity
        )

        times.append(
            total_time
        )

        top1_matches.append(
            top1_match
        )

        top3_matches.append(
            top3_match
        )

        details.append(
            {
                "question": question,
                "expected": expected,
                "similarity": top1_similarity,
                "top3_similarity": top3_similarity,
                "page": result["page"],
                "time": total_time,
                "source": result["source"],
                "evidence_match": top1_match,
                "top3_evidence_match": top3_match,
            }
        )

    question_count = len(
        valid_questions
    )

    top1_accuracy = (
        sum(top1_matches)
        / question_count
        if question_count
        else 0.0
    )

    top3_accuracy = (
        sum(top3_matches)
        / question_count
        if question_count
        else 0.0
    )

    return {
        "model": model_name,

        # Evidence-based accuracy
        "accuracy": round(
            top1_accuracy * 100,
            2,
        ),

        "top3_accuracy": round(
            top3_accuracy * 100,
            2,
        ),

        # Keep old keys for compatibility
        "avg_similarity": round(
            float(
                np.mean(similarities)
            )
            if similarities
            else 0.0,
            4,
        ),

        "top3_similarity": round(
            float(
                np.mean(
                    top3_similarities
                )
            )
            if top3_similarities
            else 0.0,
            4,
        ),

        "avg_time": round(
            float(
                np.mean(times)
            )
            if times
            else 0.0,
            4,
        ),

        "details": details,
        "question_count": question_count,
        "top1_matches": top1_matches,
        "top3_matches": top3_matches,
        "win_count": 0,
    }


# =========================================================
# PER-QUESTION WINNER CALCULATION
# =========================================================

def calculate_question_winners(
    results,
):

    if not results:
        return []

    question_count = max(
        len(
            result.get(
                "details",
                []
            )
        )
        for result in results
    )

    winners = []

    for question_index in range(
        question_count
    ):

        candidates = []

        for result in results:

            details = result.get(
                "details",
                []
            )

            if question_index >= len(
                details
            ):
                continue

            detail = details[
                question_index
            ]

            candidates.append(
                {
                    "model": result["model"],
                    "evidence": (
                        1
                        if detail.get(
                            "evidence_match",
                            False,
                        )
                        else 0
                    ),
                    "top3_evidence": (
                        1
                        if detail.get(
                            "top3_evidence_match",
                            False,
                        )
                        else 0
                    ),
                    "similarity": float(
                        detail.get(
                            "similarity",
                            0.0,
                        )
                    ),
                    "top3_similarity": float(
                        detail.get(
                            "top3_similarity",
                            0.0,
                        )
                    ),
                    "time": float(
                        detail.get(
                            "time",
                            float("inf"),
                        )
                    ),
                }
            )

        if not candidates:
            continue

        # Priority:
        # 1. Correct evidence
        # 2. Top-3 evidence
        # 3. Similarity
        # 4. Top-3 similarity
        # 5. Faster processing
        winner = max(
            candidates,
            key=lambda x: (
                x["evidence"],
                x["top3_evidence"],
                x["similarity"],
                x["top3_similarity"],
                -x["time"],
            ),
        )

        question_text = ""

        for result in results:

            details = result.get(
                "details",
                []
            )

            if question_index < len(
                details
            ):
                question_text = details[
                    question_index
                ].get(
                    "question",
                    "",
                )
                break

        winners.append(
            {
                "question_number": (
                    question_index + 1
                ),
                "question": question_text,
                "best_model": winner["model"],
                "similarity": round(
                    winner["similarity"],
                    4,
                ),
            }
        )

    return winners


# =========================================================
# COMPLETE MODEL COMPARISON
# =========================================================

def evaluate_all_models(
    chunks,
    questions,
):

    results = []

    for model_name in MODELS:

        result = evaluate_model(
            model_name,
            chunks,
            questions,
        )

        if result is not None:
            results.append(result)

    if not results:
        return results

    # -----------------------------------------------------
    # Determine best model for each question
    # -----------------------------------------------------

    question_winners = (
        calculate_question_winners(
            results
        )
    )

    # -----------------------------------------------------
    # Count model wins
    # -----------------------------------------------------

    win_counts = {
        model_name: 0
        for model_name in MODELS
    }

    for winner in question_winners:

        model_name = winner[
            "best_model"
        ]

        if model_name in win_counts:
            win_counts[
                model_name
            ] += 1

    # Add win count to every model result
    for result in results:

        result["win_count"] = (
            win_counts.get(
                result["model"],
                0,
            )
        )

    # -----------------------------------------------------
    # Overall winner
    # -----------------------------------------------------

    overall_winner = max(
        results,
        key=lambda r: (
            r.get(
                "win_count",
                0,
            ),
            r.get(
                "accuracy",
                0.0,
            ),
            r.get(
                "top3_accuracy",
                0.0,
            ),
            r.get(
                "avg_similarity",
                0.0,
            ),
            r.get(
                "top3_similarity",
                0.0,
            ),
            -r.get(
                "avg_time",
                float("inf"),
            ),
        ),
    )

    # Store comparison-level information
    for result in results:

        result["question_winners"] = (
            question_winners
        )

        result["win_counts"] = (
            win_counts
        )

        result["overall_best_model"] = (
            overall_winner["model"]
        )

    return results


# =========================================================
# RECOMMENDATION
# =========================================================

def recommend_model(results):

    if not results:
        return None

    # Overall recommendation:
    # question wins are the primary criterion,
    # followed by evidence accuracy,
    # top-3 accuracy,
    # similarity,
    # and speed.

    return max(
        results,
        key=lambda r: (
            r.get(
                "win_count",
                0,
            ),
            r.get(
                "accuracy",
                0.0,
            ),
            r.get(
                "top3_accuracy",
                0.0,
            ),
            r.get(
                "avg_similarity",
                0.0,
            ),
            r.get(
                "top3_similarity",
                0.0,
            ),
            -r.get(
                "avg_time",
                float("inf"),
            ),
        ),
    )
