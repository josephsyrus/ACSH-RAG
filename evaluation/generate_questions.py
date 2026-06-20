import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve_api import retrieve_chunks


# These words cast a wide net across your corpus to surface varied chunks.
# Replace or add words specific to your actual document domain.
SAMPLE_QUERIES = [
    "what", "how", "when", "who", "why",
    "define", "explain", "requirement", "process",
    "policy", "rule", "condition", "procedure",
]


def browse_chunks():
    print("\n" + "=" * 65)
    print("  ACSH-RAG — Golden Dataset Question Builder")
    print("=" * 65)
    print(
        "\nThis tool shows real chunks from your documents.\n"
        "For each chunk:\n"
        "  1. Pick one clear fact from it\n"
        "  2. Write a question someone would ask to get that fact\n"
        "  3. Write the answer using ONLY what the chunk says\n"
        "  4. Add both to evaluation/golden_dataset.csv\n"
        "\nOpen golden_dataset.csv in a second window before starting.\n"
    )

    all_chunks = []
    seen_ids   = set()

    for query in SAMPLE_QUERIES:
        try:
            results = retrieve_chunks(query, top_k=8)
            for r in results:
                cid = r.get("chunk_id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(r)
        except Exception as e:
            print(f"  Warning: query '{query}' failed — {e}")

    if not all_chunks:
        print(
            "\nERROR: No chunks returned.\n"
            "Make sure ingestion has been run:\n"
            "  python ingest_documents.py\n"
        )
        return

    print(f"  Found {len(all_chunks)} unique chunks to browse.\n")
    print("  Press ENTER for the next chunk. Type 'q' to quit.\n")
    print("-" * 65)

    for i, chunk in enumerate(all_chunks):
        meta     = chunk.get("metadata", {})
        filename = chunk.get("filename") or meta.get("filename", "unknown")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text     = chunk.get("text", "").strip().replace("\n", " ")

        # Retrieval source breakdown
        found_in     = chunk.get("found_in", [])
        graph_score  = chunk.get("graph_score", 0.0)
        vector_score = chunk.get("vector_score", 0.0)
        bm25_score   = chunk.get("bm25_score", 0.0)

        print(f"\n  Chunk {i+1}/{len(all_chunks)}")
        print(f"  File    : {filename}")
        print(f"  ID      : {chunk_id}")
        print(f"  Found in: {found_in}  |  vec={vector_score:.3f}  bm25={bm25_score:.3f}  graph={graph_score:.3f}")
        print(f"  Content :")
        print()

        # Word-wrap at 62 chars
        words = text.split()
        line  = []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > 62:
                print("    " + " ".join(line[:-1]))
                line = [word]
        if line:
            print("    " + " ".join(line))

        print()
        print("  " + "-" * 60)
        print("  Write a Q&A pair in golden_dataset.csv for this chunk.")
        print("  " + "-" * 60)

        try:
            user_input = input("\n  [ENTER = next | 'q' = quit] > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("q", "quit", "exit"):
            break

    print(
        "\n  Done browsing.\n"
        "  Fill in evaluation/golden_dataset.csv with your QA pairs.\n"
    )


if __name__ == "__main__":
    browse_chunks()
