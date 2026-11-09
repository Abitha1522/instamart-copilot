from pathlib import Path
import sys
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mock_llm


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT_DIR / "data" / "policies"
EVAL_FILE = ROOT_DIR / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. LOAD AND CHUNK POLICY DOCUMENTS
# ---------------------------------------------------------

def load_chunks(overlap=10):
    chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        words = text.split()

        start = 0

        while start < len(words):
            end = start + 40
            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunks.append({
                "source": file_path.name,
                "text": " ".join(chunk_words)
            })

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# ---------------------------------------------------------
# 2. TF-IDF VECTORISATION
# ---------------------------------------------------------

chunks = load_chunks(overlap=10)

chunk_texts = [chunk["text"] for chunk in chunks]

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)


# ---------------------------------------------------------
# 3. RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------------

def retrieve(query, k=3, active_chunks=None, active_vectorizer=None,
             active_vectors=None):

    if active_chunks is None:
        active_chunks = chunks
        active_vectorizer = vectorizer
        active_vectors = chunk_vectors

    query_vector = active_vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        active_vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:
        results.append({
            "source": active_chunks[index]["source"],
            "score": float(similarities[index]),
            "text": active_chunks[index]["text"]
        })

    return results


# ---------------------------------------------------------
# 4. RAG ANSWER FUNCTION
# ---------------------------------------------------------

def answer(question):
    retrieved = retrieve(question)

    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n".join(context_parts)

    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Instamart POLICY ASSISTANT. "
                "Answer only using the supplied policy context. "
                "Do not use outside knowledge or invent rules. "
                "If the answer is not present in the context, "
                "begin your response with: I could not find"
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = mock_llm.chat(messages)

    return {
        "answer": response.choices[0].message.content,
        "sources": list(dict.fromkeys(
            item["source"] for item in retrieved
        ))
    }


# ---------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------

def evaluate(active_chunks, active_vectorizer, active_vectors):

    results = {
        "answerable": 0,
        "retrieval_correct": 0,
        "fact_correct": 0,
        "unanswerable": 0,
        "fallback_correct": 0
    }

    broken_questions = []

    with EVAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            question = row["question"]
            expected_doc = row["expected_doc"].strip()
            expected_fact = row["expected_fact"].strip()

            retrieved = retrieve(
                question,
                k=3,
                active_chunks=active_chunks,
                active_vectorizer=active_vectorizer,
                active_vectors=active_vectors
            )

            sources = [item["source"] for item in retrieved]

            # Use the same retrieved context to generate the answer.
            context = "\n".join(
                f"[{item['source']}]\n{item['text']}"
                for item in retrieved
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an Instamart POLICY ASSISTANT. "
                        "Answer only using the supplied context. "
                        "If the information is absent, begin with "
                        "I could not find"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ]

            response = mock_llm.chat(messages)
            generated_answer = response.choices[0].message.content

            if expected_doc.lower() in ("", "none", "nan"):
                results["unanswerable"] += 1

                if generated_answer.startswith("I could not find"):
                    results["fallback_correct"] += 1

            else:
                results["answerable"] += 1

                if expected_doc in sources:
                    results["retrieval_correct"] += 1
                else:
                    broken_questions.append(question)

                if expected_fact in generated_answer:
                    results["fact_correct"] += 1
                else:
                    broken_questions.append(question)

    return results, broken_questions


# ---------------------------------------------------------
# 6. ZERO-OVERLAP VECTOR STORE
# ---------------------------------------------------------

def build_vector_store(overlap):
    active_chunks = load_chunks(overlap=overlap)

    active_vectorizer = TfidfVectorizer()

    active_vectors = active_vectorizer.fit_transform(
        [item["text"] for item in active_chunks]
    )

    return active_chunks, active_vectorizer, active_vectors


# ---------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("PART D1 - POLICY RAG ASSISTANT")
    print("=" * 55)

    print("\nCHUNK SUMMARY")
    print("-" * 55)

    print("Total chunks:", len(chunks))

    for source in sorted(set(item["source"] for item in chunks)):
        count = sum(
            item["source"] == source for item in chunks
        )
        print(source, ":", count)

    print("\nRETRIEVAL TEST")
    print("-" * 55)

    test_results = retrieve("refund policy late delivery")

    for item in test_results:
        print(
            f"Source: {item['source']} | "
            f"Score: {item['score']:.4f}"
        )

    print("\nANSWER TEST")
    print("-" * 55)

    result = answer(
        "What happens if a completed order is delivered after 45 minutes?"
    )

    print("Answer:", result["answer"])
    print("Sources:", result["sources"])

    print("\nEVALUATION WITH 10-WORD OVERLAP")
    print("-" * 55)

    overlap_results, overlap_broken = evaluate(
        chunks, vectorizer, chunk_vectors
    )

    print(overlap_results)

    print("\nEVALUATION WITH ZERO OVERLAP")
    print("-" * 55)

    zero_chunks, zero_vectorizer, zero_vectors = build_vector_store(0)

    zero_results, zero_broken = evaluate(
        zero_chunks, zero_vectorizer, zero_vectors
    )

    print(zero_results)

    print("\nZERO-OVERLAP QUESTIONS TO INSPECT")
    for question in list(dict.fromkeys(zero_broken)):
        print("-", question)

    # Save evaluation results
    with (OUTPUT_DIR / "d1_rag_evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "configuration",
            "answerable",
            "retrieval_correct",
            "fact_correct",
            "unanswerable",
            "fallback_correct"
        ])

        writer.writerow([
            "overlap_10",
            *overlap_results.values()
        ])

        writer.writerow([
            "overlap_0",
            *zero_results.values()
        ])

    print("\nSaved: outputs/d1_rag_evaluation.csv")
