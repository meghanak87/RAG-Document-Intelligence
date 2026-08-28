import os
import re
import time
import hashlib
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from chunker import create_chunks
from model_comparison import (
    MODELS,
    retrieve_with_model,
    evaluate_all_models,
    recommend_model,
)


# =========================================================
# CONFIG
# =========================================================

load_dotenv(".env", override=True)

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="🤖",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "documents": {},
    "selected_document": None,
    "history": {},
    "comparison_question": {},
    "comparison_answer": {},
    "comparison_source": {},
    "eval_results": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# PDF CACHE
# =========================================================

def file_key(uploaded_file):
    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


def cache_path(key):
    return STORAGE_DIR / f"{key}.pkl"


def load_cached_documents():

    for path in STORAGE_DIR.glob("*.pkl"):

        try:

            with open(path, "rb") as f:
                item = pickle.load(f)

            if (
                "name" in item
                and "chunks" in item
            ):
                st.session_state.documents[
                    item["name"]
                ] = item["chunks"]

        except Exception:
            continue


if not st.session_state.documents:
    load_cached_documents()


def process_pdf(uploaded_file):

    key = file_key(uploaded_file)
    path = cache_path(key)

    if path.exists():

        with open(path, "rb") as f:
            item = pickle.load(f)

        return (
            item["name"],
            item["chunks"],
            False,
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as f:

        f.write(
            uploaded_file.getvalue()
        )

        pdf_path = f.name

    try:

        chunks = create_chunks(
            pdf_path
        )

    finally:

        try:
            os.remove(pdf_path)
        except OSError:
            pass

    if not chunks:
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    with open(path, "wb") as f:

        pickle.dump(
            {
                "name": uploaded_file.name,
                "chunks": chunks,
            },
            f,
        )

    return (
        uploaded_file.name,
        chunks,
        True,
    )


def remove_pdf(name):

    for path in STORAGE_DIR.glob("*.pkl"):

        try:

            with open(path, "rb") as f:
                item = pickle.load(f)

            if item.get("name") == name:
                path.unlink(
                    missing_ok=True
                )

        except Exception:
            continue

    st.session_state.documents.pop(
        name,
        None,
    )

    st.session_state.history.pop(
        name,
        None,
    )

    st.session_state.comparison_question.pop(
        name,
        None,
    )

    st.session_state.comparison_answer.pop(
        name,
        None,
    )

    st.session_state.comparison_source.pop(
        name,
        None,
    )

    st.session_state.eval_results.pop(
        name,
        None,
    )

    names = list(
        st.session_state.documents
    )

    st.session_state.selected_document = (
        names[0] if names else None
    )


# =========================================================
# TEXT HELPERS
# =========================================================

def clean_text(text):

    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()


def chunk_text(chunk):

    if isinstance(chunk, dict):
        return clean_text(
            chunk.get("text", "")
        )

    return clean_text(chunk)


def chunk_page(chunk):

    if isinstance(chunk, dict):
        return chunk.get(
            "page",
            1,
        )

    return 1


# =========================================================
# RAG ANSWER GENERATION
# =========================================================

def generate_concise_answer(
    question,
    retrieved_chunks,
):

    if not retrieved_chunks:

        return (
            "The answer is not available "
            "in the PDF."
        )

    context_parts = []

    for item in retrieved_chunks:

        text = clean_text(
            item.get("text", "")
        )

        page = item.get(
            "page",
            1,
        )

        if text:

            context_parts.append(
                f"Page {page}:\n{text}"
            )

    context = "\n\n".join(
        context_parts
    )

    if not context:

        return (
            "The answer is not available "
            "in the PDF."
        )

    try:

        client = genai.Client()

        prompt = f"""
You are a precise PDF question-answering system.

Answer the user's question using ONLY the
information contained in the PDF evidence below.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not summarize the entire document.
4. Answer exactly what the question asks.
5. Give a concise answer.
6. If the question asks for a list, provide the list.
7. If the question asks for a name, give the name.
8. If the question asks for a number, give the number.
9. If several items are requested, include all supported items.
10. If the evidence does not contain the answer,
    say exactly:
    "The answer is not available in the PDF."
11. Do not mention the retrieval process.
12. Do not write a professional summary unless the
    question explicitly asks for one.
13. Do not add information that is not present in
    the evidence.

PDF EVIDENCE:

{context}

USER QUESTION:

{question}

CONCISE ANSWER:
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        answer = clean_text(
            response.text
        )

        if answer:
            return answer

    except Exception as e:

        # Safe deterministic fallback.
        # This prevents the application from crashing
        # if the answer-generation service is unavailable.
        pass

    return deterministic_fallback(
        question,
        retrieved_chunks,
    )


# =========================================================
# DETERMINISTIC FALLBACK
# =========================================================

def deterministic_fallback(
    question,
    retrieved_chunks,
):

    question_words = set(
        re.findall(
            r"[a-z0-9+#.]+",
            question.lower(),
        )
    )

    stop_words = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "how",
        "why",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "for",
        "in",
        "on",
        "to",
        "from",
        "and",
        "or",
        "with",
        "does",
        "do",
        "did",
        "can",
        "could",
        "tell",
        "me",
        "give",
        "about",
    }

    question_words -= stop_words

    candidates = []

    for item in retrieved_chunks:

        text = clean_text(
            item.get("text", "")
        )

        if not text:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        for sentence in sentences:

            sentence = clean_text(
                sentence
            )

            if not sentence:
                continue

            words = set(
                re.findall(
                    r"[a-z0-9+#.]+",
                    sentence.lower(),
                )
            )

            score = len(
                question_words & words
            )

            if score > 0:

                candidates.append(
                    (
                        score,
                        -len(sentence),
                        sentence,
                    )
                )

    if candidates:

        candidates.sort(
            reverse=True
        )

        return candidates[0][2][:500]

    return (
        "The answer is not available "
        "in the PDF."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "🤖 RAG Document Intelligence"
    )

    page = st.radio(
        "Pages",
        [
            "📄 Ask Questions",
            "📊 Model Comparison",
            "🏆 Model Recommendation",
        ],
    )

    st.divider()

    if st.session_state.documents:

        names = list(
            st.session_state.documents.keys()
        )

        if (
            st.session_state.selected_document
            not in names
        ):

            st.session_state.selected_document = (
                names[0]
            )

        selected = st.selectbox(
            "Current PDF",
            names,
            index=names.index(
                st.session_state.selected_document
            ),
        )

        st.session_state.selected_document = (
            selected
        )

        if st.button(
            "🗑️ Remove selected PDF",
            width="stretch",
        ):

            remove_pdf(selected)
            st.rerun()


# =========================================================
# UPLOAD
# =========================================================

st.title(
    "🤖 RAG Document Intelligence"
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    try:

        (
            name,
            chunks,
            newly_processed,
        ) = process_pdf(
            uploaded_file
        )

        st.session_state.documents[
            name
        ] = chunks

        st.session_state.selected_document = (
            name
        )

        if newly_processed:

            st.success(
                f"Processed {name} — "
                f"{len(chunks)} chunks created."
            )

        else:

            st.success(
                f"{name} is already stored. "
                "No re-processing needed."
            )

    except Exception as e:

        st.error(
            f"PDF processing failed: {e}"
        )

        st.stop()


if not st.session_state.documents:

    st.info(
        "Upload a PDF to begin."
    )

    st.stop()


document_name = (
    st.session_state.selected_document
)

chunks = st.session_state.documents[
    document_name
]


# =========================================================
# PAGE 1 — ASK QUESTIONS
# =========================================================

if page == "📄 Ask Questions":

    st.header(
        "💬 Ask a Question"
    )

    model_name = st.selectbox(
        "Embedding model",
        list(MODELS.keys()),
        key=f"qa_model_{document_name}",
    )

    question = st.text_input(
        "Question",
        placeholder=(
            "Ask anything that is present "
            "in the uploaded PDF..."
        ),
    )

    if st.button(
        "🔍 Get Answer",
        type="primary",
        width="stretch",
    ):

        if not question.strip():

            st.warning(
                "Enter a question."
            )

            st.stop()

        with st.spinner(
            f"Retrieving relevant information "
            f"with {model_name}..."
        ):

            result = retrieve_with_model(
                model_name,
                question,
                chunks,
                top_k=5,
            )

        if not result:

            st.error(
                "No relevant information was "
                "retrieved from this PDF."
            )

        else:

            with st.spinner(
                "Preparing concise answer..."
            ):

                answer = generate_concise_answer(
                    question,
                    result["retrieved"],
                )

            st.subheader(
                "🤖 Answer"
            )

            st.success(
                answer
            )

            st.caption(
                f"Model: {model_name} | "
                f"Top source page: "
                f"{result['page']} | "
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )

            # -------------------------------------------------
            # SAVE QUESTION FOR PAGE 2
            # -------------------------------------------------

            st.session_state.comparison_question[
                document_name
            ] = question.strip()

            st.session_state.comparison_answer[
                document_name
            ] = answer

            st.session_state.comparison_source[
                document_name
            ] = result["retrieved"]

            # New question means previous comparison
            # is no longer valid.
            st.session_state.eval_results.pop(
                document_name,
                None,
            )

            # -------------------------------------------------
            # HISTORY
            # -------------------------------------------------

            st.session_state.history.setdefault(
                document_name,
                [],
            ).append(
                {
                    "question":
                        question.strip(),

                    "answer":
                        answer,

                    "model":
                        model_name,

                    "page":
                        result["page"],
                }
            )

            with st.expander(
                "📑 Review retrieved PDF evidence"
            ):

                for item in result["retrieved"]:

                    st.markdown(
                        f"**Rank {item['rank']} | "
                        f"Page {item['page']} | "
                        f"Similarity "
                        f"{item['score']:.4f}**"
                    )

                    st.write(
                        item["text"]
                    )

    history = st.session_state.history.get(
        document_name,
        [],
    )

    if history:

        st.divider()

        st.subheader(
            "📚 Previous Questions"
        )

        for item in reversed(history):

            with st.expander(
                item["question"]
            ):

                st.write(
                    f"**Answer:** "
                    f"{item['answer']}"
                )

                st.caption(
                    f"{item['model']} • "
                    f"Page {item['page']}"
                )


# =========================================================
# PAGE 2 — MODEL COMPARISON
# =========================================================

elif page == "📊 Model Comparison":

    st.header(
        "📊 Model Comparison"
    )

    st.write(
        f"Comparison for **{document_name}**"
    )

    saved_question = (
        st.session_state.comparison_question.get(
            document_name,
            "",
        )
    )

    saved_answer = (
        st.session_state.comparison_answer.get(
            document_name,
            "",
        )
    )

    if not saved_question:

        st.warning(
            "First ask a question on Page 1. "
            "This page automatically uses that "
            "same question."
        )

        st.stop()

    st.subheader(
        "Question from Page 1"
    )

    st.info(
        saved_question
    )

    st.subheader(
        "Reference answer from Page 1"
    )

    st.success(
        saved_answer
    )

    st.caption(
        "The question is automatically carried "
        "from Page 1. You do not need to enter it again."
    )

    st.divider()

    if st.button(
        "🚀 Compare All Models",
        type="primary",
        width="stretch",
    ):

        evaluation_questions = [
            {
                "question":
                    saved_question,

                "answer":
                    saved_answer,
            }
        ]

        with st.spinner(
            "Testing all embedding models "
            "on the same PDF and same question..."
        ):

            results = evaluate_all_models(
                chunks,
                evaluation_questions,
            )

        st.session_state.eval_results[
            document_name
        ] = results

        st.success(
            "Comparison completed for this PDF "
            "and this question."
        )

    results = st.session_state.eval_results.get(
        document_name,
        [],
    )

    if results:

        st.divider()

        st.subheader(
            "📈 Comparison Results"
        )

        rows = []

        for result in results:

            rows.append(
                {
                    "Model":
                        result["model"],

                    "Top-1 Similarity":
                        result["avg_similarity"],

                    "Top-3 Similarity":
                        result["top3_similarity"],

                    "Average Time (s)":
                        result["avg_time"],
                }
            )

        df = pd.DataFrame(
            rows
        )

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "All models are tested on the same PDF and the same question. "
            "Higher retrieval similarity indicates stronger retrieval for this question."
        )

        st.divider()

        st.subheader(
            "🔎 Review Each Model"
        )

        selected_model = st.selectbox(
            "Select model to review",
            [
                result["model"]
                for result in results
            ],
            key=f"review_{document_name}",
        )

        selected_result = next(
            result
            for result in results
            if result["model"]
            == selected_model
        )

        details = (
            selected_result["details"]
        )

        for detail in details:
            st.markdown("### Question")
            st.write(detail.get("question", saved_question))

            st.markdown("**Reference answer:**")
            st.write(detail.get("expected", saved_answer))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Top-1 Similarity",
                    f"{detail.get('similarity', 0.0):.4f}",
                )

            with col2:
                st.metric(
                    "Top-3 Similarity",
                    f"{detail.get('top3_similarity', 0.0):.4f}",
                )

            with col3:
                st.metric(
                    "Retrieval Time",
                    f"{detail.get('time', 0.0):.4f}s",
                )

            st.caption(
                f"Retrieved page: {detail.get('page', 'N/A')}"
            )

            st.markdown("**Retrieved Evidence:**")
            st.write(
                detail.get(
                    "source",
                    "No evidence retrieved.",
                )
            )

# =========================================================
# PAGE 3 — MODEL RECOMMENDATION
# =========================================================

else:

    st.header(
        "🏆 Model Recommendation"
    )

    saved_question = (
        st.session_state.comparison_question.get(
            document_name,
            "",
        )
    )

    results = (
        st.session_state.eval_results.get(
            document_name,
            [],
        )
    )

    if not saved_question:

        st.warning(
            "Ask a question on Page 1 first."
        )

        st.stop()

    st.subheader(
        "Question evaluated"
    )

    st.info(
        saved_question
    )

    if not results:

        st.warning(
            "Run 'Compare All Models' on Page 2 "
            "before viewing the recommendation."
        )

        st.stop()

    best = recommend_model(
        results
    )

    if best is None:

        st.error(
            "No recommendation is available."
        )

        st.stop()

    st.success(
        f"🏆 Recommended model for "
        f"**{document_name}**: "
        f"**{best['model']}**"
    )

    st.subheader(
        "Why this model?"
    )

    st.write(
        f"""
The recommendation is calculated from the
actual comparison results for **this PDF**
and the same question asked on Page 1.

- **Top-1 Similarity:** {best["avg_similarity"]:.4f}
- **Top-3 Similarity:** {best["top3_similarity"]:.4f}
- **Average Retrieval Time:** {best['avg_time']:.4f} seconds
"""
    )

    st.info(
        "There is no permanently fixed best model. "
        "The model with the strongest measured "
        "evaluation result for the current PDF/question "
        "is recommended."
    )

    st.subheader(
        "📊 Complete Comparison"
    )

    rows = []

    for result in results:

        rows.append(
            {
                "Model":
                    result["model"],

                "Top-1 Similarity":
                    result.get("avg_similarity", 0.0),

                "Top-3 Similarity":
                    result.get("top3_similarity", 0.0),

                "Average Similarity":
                    result["avg_similarity"],

                "Average Time (s)":
                    result["avg_time"],
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "RAG Embedding Comparison • "
    "PDF-specific evaluation • "
    "Same question across all models • "
    "No hardcoded winner"
)