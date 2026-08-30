import hashlib
import os
import pickle
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from chunker import create_chunks
from model_comparison import (
    MODELS,
    retrieve_with_model,
    evaluate_all_models,
    recommend_model,
)
st.success("App imports loaded successfully")


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG Embedding Comparison",
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
    "eval_questions": {},
    "eval_results": {},
    "comparison_history": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# PERSISTENT PDF CACHE
# =========================================================

def file_key(uploaded_file):
    data = uploaded_file.getvalue()
    return hashlib.sha256(data).hexdigest()


def cache_path(key):
    return STORAGE_DIR / f"{key}.pkl"


def load_cached_documents():
    for path in STORAGE_DIR.glob("*.pkl"):
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
            name = item["name"]
            st.session_state.documents[name] = item["chunks"]
        except Exception:
            pass


if not st.session_state.documents:
    load_cached_documents()


def process_pdf(uploaded_file):
    key = file_key(uploaded_file)
    path = cache_path(key)

    if path.exists():
        with open(path, "rb") as f:
            item = pickle.load(f)
        return item["name"], item["chunks"], key, False

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as f:
        f.write(uploaded_file.getvalue())
        pdf_path = f.name

    try:
        chunks = create_chunks(pdf_path)
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    with open(path, "wb") as f:
        pickle.dump(
            {
                "name": uploaded_file.name,
                "chunks": chunks,
            },
            f,
        )

    return uploaded_file.name, chunks, key, True


def remove_pdf(name):
    # Remove matching persistent cache by checking stored names.
    for path in STORAGE_DIR.glob("*.pkl"):
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
            if item.get("name") == name:
                path.unlink(missing_ok=True)
        except Exception:
            pass

    st.session_state.documents.pop(name, None)
    st.session_state.history.pop(name, None)
    st.session_state.eval_questions.pop(name, None)
    st.session_state.eval_results.pop(name, None)

    names = list(st.session_state.documents)
    st.session_state.selected_document = names[0] if names else None


# =========================================================
# CONCISE EXTRACTIVE ANSWER
# =========================================================

def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_answer(question, context):
    q = clean_text(question).lower()
    context = clean_text(context)

    if not context:
        return "Answer not found in the document."

    # Exact-value fields
    if "cgpa" in q:
        m = re.search(r"cgpa\s*[:\-]?\s*(\d+(?:\.\d+)?)", context, re.I)
        if m:
            return m.group(1)

    if "linkedin" in q:
        urls = re.findall(r"https?://[^\s)]+", context, re.I)
        for url in urls:
            if "linkedin.com" in url.lower():
                return url.rstrip(".,;")
        return "LinkedIn link not found in the retrieved source."

    if "email" in q or "mail id" in q:
        m = re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            context,
        )
        if m:
            return m.group(0)

    if "percentage" in q or "percent" in q:
        matches = re.findall(r"\b\d+(?:\.\d+)?\s*%", context)
        if matches:
            return matches[-1]

    if "phone" in q or "mobile" in q or "contact number" in q:
        m = re.search(r"(?:\+?\d[\d\s().-]{8,}\d)", context)
        if m:
            return clean_text(m.group(0))

    # Sentence-level extraction for normal questions
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", context)
        if s.strip()
    ]

    q_words = set(re.findall(r"[a-z0-9+#]+", q))
    scored = []

    for sentence in sentences:
        words = set(re.findall(r"[a-z0-9+#]+", sentence.lower()))
        overlap = len(q_words & words)
        if overlap:
            scored.append((overlap, -len(sentence), sentence))

    if scored:
        scored.sort(reverse=True)
        answer = scored[0][2]
        if len(answer) <= 350:
            return answer

    # Never dump a whole long chunk.
    words = context.split()
    return " ".join(words[:45]) + ("..." if len(words) > 45 else "")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🤖 RAG Document Intelligence")

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
        names = list(st.session_state.documents)

        if st.session_state.selected_document not in names:
            st.session_state.selected_document = names[0]

        selected = st.selectbox(
            "Current PDF",
            names,
            index=names.index(
                st.session_state.selected_document
            ),
        )
        st.session_state.selected_document = selected

        if st.button(
            "🗑️ Remove selected PDF",
            use_container_width=True,
        ):
            remove_pdf(selected)
            st.rerun()

    else:
        st.caption("No PDF loaded.")


# =========================================================
# UPLOAD AREA
# =========================================================

st.title("🤖 RAG Document Intelligence")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    name, chunks, key, newly_processed = process_pdf(uploaded_file)

    if name not in st.session_state.documents:
        st.session_state.documents[name] = chunks

    st.session_state.selected_document = name

    if newly_processed:
        st.success(
            f"Processed {name}: {len(chunks)} chunks created."
        )
    else:
        st.success(
            f"{name} is already stored. No re-processing needed."
        )


if not st.session_state.documents:
    st.info("Upload a PDF to begin.")
    st.stop()


document_name = st.session_state.selected_document
chunks = st.session_state.documents[document_name]


# =========================================================
# PAGE 1: ASK QUESTIONS
# =========================================================

if page == "📄 Ask Questions":

    st.header("💬 Ask a Question")

    model_name = st.selectbox(
        "Embedding model",
        list(MODELS),
        key=f"qa_model_{document_name}",
    )

    question = st.text_input(
        "Question",
        placeholder="Example: What is the CGPA?",
    )

    if st.button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True,
    ):
        if not question.strip():
            st.warning("Enter a question.")
            st.stop()

        with st.spinner(f"Searching with {model_name}..."):
            result = retrieve_with_model(
                model_name,
                question,
                chunks,
                top_k=3,
            )

        if not result:

            st.error("No relevant information found.")

        else:

            answer = extract_answer(
                question,
                result["source"],
            )

            st.subheader("🤖 Answer")
            st.success(answer)

            st.caption(
                f"Model: {model_name} | "
                f"Page: {result['page']} | "
                f"Similarity: {result['similarity']:.4f}"
            )

            st.session_state.history.setdefault(
                document_name,
                [],
            ).append({
                "question": question,
                "answer": answer,
                "model": model_name,
                "page": result["page"],
            })

            with st.expander("📑 Review source"):
                st.write(result["source"])

            st.subheader("🤖 Answer")
            st.success(answer)

            st.caption(
                f"Model: {model_name} | "
                f"Page: {result['page']} | "
                f"Similarity: {result['similarity']:.4f}"
            )

            st.session_state.history.setdefault(
                document_name,
                [],
            ).append({
                "question": question,
                "answer": answer,
                "model": model_name,
                "page": result["page"],
            })

            with st.expander("📑 Review source"):
                st.write(result["source"])

    history = st.session_state.history.get(
        document_name,
        [],
    )

    if history:
        st.divider()
        st.subheader("📚 Previous Questions")

        for item in reversed(history):
            with st.expander(item["question"]):
                st.write(f"**Answer:** {item['answer']}")
                st.caption(
                    f"{item['model']} • Page {item['page']}"
                )


# =========================================================
# PAGE 2: MODEL COMPARISON

elif page == "📊 Model Comparison":

    st.header("📊 Model Comparison")
    st.write(
        f"Comparison for **{document_name}**"
    )

    # Use questions already asked on Page 1.
    # This keeps the original simple workflow: ask a question -> compare models.
    history = st.session_state.history.get(
        document_name,
        [],
    )

    if not history:
        st.info(
            "Ask at least one question on the 📄 Ask Questions page first. "
            "The question and answer will automatically appear here."
        )
        st.stop()

    # Build evaluation set from previously asked questions.
    # The answer generated from the selected model is used as the reference
    # evidence for the prototype's comparison logic.
    evaluation_questions = []
    seen = set()

    for item in history:
        q_text = str(item.get("question", "")).strip()
        a_text = str(item.get("answer", "")).strip()

        if q_text and a_text and q_text not in seen:
            evaluation_questions.append({
                "question": q_text,
                "answer": a_text,
            })
            seen.add(q_text)

    st.subheader("📋 Questions tested")
    st.caption(
        "Questions are automatically collected from the questions asked on Page 1. "
        "You do not need to enter them again."
    )

    question_rows = []
    for i, item in enumerate(evaluation_questions, start=1):
        question_rows.append({
            "#": i,
            "Question": item["question"],
        })

    st.dataframe(
        pd.DataFrame(question_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    if st.button(
        "🚀 Compare All Models",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Testing all embedding models on the same PDF and questions..."
        ):
            results = evaluate_all_models(
                chunks,
                evaluation_questions,
            )

        st.session_state.eval_results[document_name] = results
        st.success(
            f"Comparison completed for {len(evaluation_questions)} question(s)."
        )

    results = st.session_state.eval_results.get(
        document_name,
        [],
    )

    if results:
        st.divider()
        st.subheader("📈 Comparison Results")

        rows = []
        for r in results:
            rows.append({
                "Model": r["model"],
                "Accuracy (%)": round(r.get("accuracy", 0.0), 2),
                "Top-3 Accuracy (%)": round(r.get("top3_accuracy", 0.0), 2),
                "Top-1 Similarity": round(r.get("avg_similarity", 0.0), 4),
                "Top-3 Similarity": round(r.get("top3_similarity", 0.0), 4),
                "Average Time (s)": round(r.get("avg_time", 0.0), 4),
                "Questions Won": r.get("win_count", 0),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Each model is evaluated independently on the same PDF and the same questions. "
            "The recommendation is calculated from the measured comparison results."
        )

        # ---------------------------------------------------------
        # QUESTION-WISE WINNERS
        # ---------------------------------------------------------
        question_winners = results[0].get(
            "question_winners",
            [],
        )

        if question_winners:
            st.divider()
            st.subheader("🏆 Best Model for Each Question")

            winner_rows = []
            for item in question_winners:
                winner_rows.append({
                    "Question": item.get("question", ""),
                    "Best Model": item.get("best_model", ""),
                    "Top-1 Similarity": round(
                        item.get("similarity", 0.0), 4
                    ),
                })

            st.dataframe(
                pd.DataFrame(winner_rows),
                use_container_width=True,
                hide_index=True,
            )

        # ---------------------------------------------------------
        # OVERALL WIN COUNT
        # ---------------------------------------------------------
        win_counts = results[0].get(
            "win_counts",
            {},
        )

        if win_counts:
            st.divider()
            st.subheader("📊 Overall Model Performance")

            total_questions = len(question_winners)
            performance_rows = []

            for model_name in MODELS:
                performance_rows.append({
                    "Model": model_name,
                    "Questions Won": win_counts.get(
                        model_name,
                        0,
                    ),
                })

            performance_df = (
                pd.DataFrame(performance_rows)
                .sort_values("Questions Won", ascending=False)
                .reset_index(drop=True)
            )

            st.dataframe(
                performance_df,
                use_container_width=True,
                hide_index=True,
            )

            overall_best = results[0].get(
                "overall_best_model"
            )

            if overall_best:
                st.success(
                    f"🏆 Overall Best Model: **{overall_best}**"
                )
                st.caption(
                    f"Based on {total_questions} evaluated question(s) for this PDF. "
                    "The winner is recalculated whenever the comparison is run."
                )


# PAGE 3: MODEL RECOMMENDATION
# =========================================================

else:

    st.header("🏆 Model Recommendation")

    results = st.session_state.eval_results.get(
        document_name,
        [],
    )

    if not results:
        st.warning(
            "Run Model Comparison first for this PDF."
        )
        st.stop()

    recommendation = recommend_model(results)

    if recommendation is None:
        st.error("No evaluation result available.")
        st.stop()

    st.success(
        f"🏆 Recommended model for **{document_name}**: "
        f"**{recommendation['model']}**"
    )

    st.write(
        f"""
**Why this model was selected**

- Questions Won: **{recommendation.get('win_count', 0)}**
- Accuracy: **{recommendation['accuracy']:.2f}%**
- Top-3 Accuracy: **{recommendation['top3_accuracy']:.2f}%**
- Average Similarity: **{recommendation['avg_similarity']:.4f}**
- Average Time: **{recommendation['avg_time']:.4f} seconds**

Question wins are the primary criterion for the overall recommendation.
Accuracy, Top-3 accuracy, similarity and speed are used as tie-breakers.
"""
    )

    st.info(
        "This recommendation is calculated from the evaluation "
        "results of the currently selected PDF. Another PDF can "
        "produce a different winner."
    )

    st.subheader("🔎 Review individual model")

    selected_review = st.selectbox(
        "Model",
        [r["model"] for r in results],
    )

    selected_result = next(
        r for r in results
        if r["model"] == selected_review
    )

    review_rows = []

    for d in selected_result["details"]:
        review_rows.append({
    "Question": d.get("question", ""),
    "Expected": d.get("expected", ""),
    "Top-1": "Correct" if d.get("evidence_match", False) else "Wrong",
    "Top-3": "Correct" if d.get("top3_evidence_match", False) else "Wrong",
    "Similarity": round(d.get("similarity", 0.0), 4),
    "Page": d.get("page", ""),
    })

    st.dataframe(
        pd.DataFrame(review_rows),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "PDF-specific RAG evaluation • No hardcoded model winner • "
    "Concise extractive answers"
)
