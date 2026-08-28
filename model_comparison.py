import re
import time
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODELS = {
    "BGE-M3": "BAAI/bge-m3",
    "Multilingual-E5-large": "intfloat/multilingual-e5-large",
    "BGE-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
}


@lru_cache(maxsize=4)
def load_model(model_name):
    return SentenceTransformer(MODELS[model_name])


def _text(chunk):
    if isinstance(chunk, dict):
        return str(chunk.get("text", "")).strip()
    return str(chunk).strip()


def _page(chunk):
    return chunk.get("page", 1) if isinstance(chunk, dict) else 1


def _encode(model, texts, model_name, is_query=False):
    """
    E5 is trained with query:/passage: prefixes. Other models do not need them.
    This fixes the common E5 retrieval-quality problem.
    """
    if model_name == "Multilingual-E5-large":
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + str(x) for x in texts]

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


@lru_cache(maxsize=32)
def _cached_document_embeddings(model_name, chunks_key, texts_tuple):
    model = load_model(model_name)
    return np.asarray(
        _encode(model, list(texts_tuple), model_name, is_query=False),
        dtype="float32",
    )


def retrieve_with_model(model_name, question, chunks, top_k=3):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    valid = [
        (i, _text(c))
        for i, c in enumerate(chunks)
        if _text(c)
    ]

    if not valid:
        return None

    original_ids = [x[0] for x in valid]
    texts = [x[1] for x in valid]

    model = load_model(model_name)

    # Include document identity in cache key so different PDFs never share vectors.
    chunks_key = hash(tuple(texts))

    t0 = time.perf_counter()
    doc_embeddings = _cached_document_embeddings(
        model_name,
        chunks_key,
        tuple(texts),
    )
    embedding_time = time.perf_counter() - t0

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    t1 = time.perf_counter()
    q_embedding = np.asarray(
        _encode(model, [question], model_name, is_query=True),
        dtype="float32",
    )
    k = min(top_k, len(texts))
    scores, ids = index.search(q_embedding, k)
    retrieval_time = time.perf_counter() - t1

    retrieved = []
    for rank, local_id in enumerate(ids[0], start=1):
        local_id = int(local_id)
        if local_id < 0:
            continue

        original_id = original_ids[local_id]
        chunk = chunks[original_id]

        retrieved.append({
            "rank": rank,
            "score": float(scores[0][rank - 1]),
            "page": _page(chunk),
            "text": _text(chunk),
        })

    if not retrieved:
        return None

    return {
        "model": model_name,
        "dimension": int(doc_embeddings.shape[1]),
        "similarity": retrieved[0]["score"],
        "top3_score": float(np.mean([x["score"] for x in retrieved])),
        "embedding_time": embedding_time,
        "retrieval_time": retrieval_time,
        "total_time": embedding_time + retrieval_time,
        "page": retrieved[0]["page"],
        "source": retrieved[0]["text"],
        "retrieved": retrieved,
    }


# ---------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------

def normalize_answer(text):
    text = str(text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"https?://", " ", text)
    text = re.sub(r"[^a-z0-9+#.%/&-]+", " ", text)
    return " ".join(text.split())


def answer_match(source_text, expected_answer):
    """
    Model-independent evidence check.
    It does NOT call an LLM and does NOT use the model being evaluated.
    """
    expected = normalize_answer(expected_answer)
    source = normalize_answer(source_text)

    if not expected or not source:
        return False

    if expected in source:
        return True

    # Compare meaningful tokens while ignoring very common words.
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to",
        "in", "on", "for", "and", "or", "with", "what", "which",
        "who", "where", "when", "this", "that"
    }

    exp_tokens = [
        x for x in expected.split()
        if len(x) > 1 and x not in stop
    ]
    src_tokens = set(
        x for x in source.split()
        if len(x) > 1 and x not in stop
    )

    if not exp_tokens:
        return False

    overlap = sum(x in src_tokens for x in exp_tokens) / len(exp_tokens)

    # High token coverage counts as supporting evidence.
    return overlap >= 0.70


def evaluate_model(model_name, chunks, questions):
    valid_questions = [
        q for q in questions
        if str(q.get("question", "")).strip()
        and str(q.get("answer", "")).strip()
    ]

    if not valid_questions:
        return None

    similarities = []
    top3_similarities = []
    times = []
    details = []

    for item in valid_questions:
        result = retrieve_with_model(
            model_name, item["question"], chunks, top_k=3
        )

        if result is None:
            details.append({
                "question": item["question"],
                "expected": item["answer"],
                "similarity": 0.0,
                "top3_similarity": 0.0,
                "page": None,
                "time": 0.0,
                "source": "No evidence retrieved.",
                "evidence_match": False,
            })
            continue

        top1 = float(result["similarity"])
        top3 = float(result["top3_score"])
        total_time = float(result["total_time"])

        similarities.append(top1)
        top3_similarities.append(top3)
        times.append(total_time)

        details.append({
            "question": item["question"],
            "expected": item["answer"],
            "similarity": top1,
            "top3_similarity": top3,
            "page": result["page"],
            "time": total_time,
            "source": result["source"],
            "evidence_match": answer_match(
                result["source"], item["answer"]
            ),
        })

    return {
        "model": model_name,
        "avg_similarity": round(
            float(np.mean(similarities)) if similarities else 0.0, 4
        ),
        "top3_similarity": round(
            float(np.mean(top3_similarities)) if top3_similarities else 0.0, 4
        ),
        "avg_time": round(
            float(np.mean(times)) if times else 0.0, 4
        ),
        "details": details,
    }


def evaluate_all_models(chunks, questions):
    results = []

    for model_name in MODELS:
        result = evaluate_model(
            model_name,
            chunks,
            questions,
        )
        if result is not None:
            results.append(result)

    return results


def recommend_model(results):
    """Choose a winner dynamically for the current PDF/question."""
    if not results:
        return None

    return max(
        results,
        key=lambda r: (
            r.get("avg_similarity", 0.0),
            r.get("top3_similarity", 0.0),
            -r.get("avg_time", float("inf")),
        ),
    )
