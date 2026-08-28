import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)
import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer

from chunker import process_pdf


class RAGEngine:

    def __init__(self, model_name="BAAI/bge-m3"):

        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.embedding_model = SentenceTransformer(
            model_name
        )

        self.chunks = []
        self.index = None

        print("Embedding model loaded successfully!")

    def process_document(self, pdf_path):

        print("\nProcessing PDF...")

        self.chunks = process_pdf(pdf_path)

        if not self.chunks:
            raise ValueError(
                "No text could be extracted from the PDF."
            )

        print(
            f"Number of chunks: {len(self.chunks)}"
        )

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        print("Creating embeddings...")

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        print("FAISS index created!")
        print(
            f"Vectors stored: {self.index.ntotal}"
        )

    def retrieve(self, question, top_k=3):

        if self.index is None:
            raise ValueError(
                "Please process a PDF first."
            )

        question_embedding = (
            self.embedding_model.encode(
                [question],
                normalize_embeddings=True
            )
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            question_embedding,
            min(top_k, len(self.chunks))
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            results.append({
                "score": float(score),
                "page": self.chunks[index]["page"],
                "text": self.chunks[index]["text"]
            })

        return results

    def generate_answer(
        self,
        question,
        retrieved_chunks
    ):

        client = genai.Client()

        context = "\n\n".join(
            [
                f"Page {item['page']}:\n"
                f"{item['text']}"
                for item in retrieved_chunks
            ]
        )

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
information provided in the document context.

If the answer is not present in the context,
say that the information is not available
in the document.

Give a specific and concise answer.

Mention the page number containing the answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text

    def ask(self, question, top_k=3):

        retrieved = self.retrieve(
            question,
            top_k
        )

        answer = self.generate_answer(
            question,
            retrieved
        )

        return {
            "answer": answer,
            "sources": retrieved
        }


if __name__ == "__main__":

    pdf_path = "data/documents/sample.pdf"

    print("\n===================================")
    print("RAG ENGINE TEST")
    print("===================================")

    engine = RAGEngine(
        model_name="BAAI/bge-m3"
    )

    engine.process_document(
        pdf_path
    )

    question = input(
        "\nAsk a question about the PDF: "
    )

    result = engine.ask(
        question,
        top_k=3
    )

    print("\n===================================")
    print("ANSWER")
    print("===================================")

    print(result["answer"])

    print("\n===================================")
    print("SOURCES")
    print("===================================")

    for source in result["sources"]:

        print("\nPage:", source["page"])
        print(
            "Similarity:",
            source["score"]
        )
        print(
            "Text:",
            source["text"]
        )