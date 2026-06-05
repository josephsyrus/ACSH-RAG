import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hybrid_retriever import HybridRetriever

CHROMA_DIR = "./chroma_db"
BM25_DIR   = "./bm25_index"
GRAPH_DIR  = "./graph_db"


def print_results(results: list, query: str):
    print(f"\n{'─'*60}")
    print(f" Query : {query}")
    print(f" Found : {len(results)} chunk(s)")
    print(f"{'─'*60}")

    for i, chunk in enumerate(results, start=1):
        meta        = chunk.get("metadata", {})
        filename    = meta.get("filename",    chunk.get("filename",    "unknown"))
        chunk_idx   = meta.get("chunk_index", chunk.get("chunk_index", "?"))
        total       = meta.get("total_chunks",chunk.get("total_chunks","?"))
        found_in    = ", ".join(chunk.get("found_in", []))
        rrf         = chunk.get("rrf_score",    0.0)
        vec         = chunk.get("vector_score", 0.0)
        bm25        = chunk.get("bm25_score",   0.0)
        graph       = chunk.get("graph_score",  0.0)
        v_rank      = chunk.get("vector_rank",  "—")
        b_rank      = chunk.get("bm25_rank",    "—")
        g_rank      = chunk.get("graph_rank",   "—")

        print(f"\n  ┌─ Result #{i}")
        print(f"  │  File     : {filename}  (chunk {chunk_idx}/{total})")
        print(f"  │  RRF      : {rrf:.5f}   (Vector rank: {v_rank}, BM25 rank: {b_rank}, Graph rank: {g_rank})")
        print(f"  │  Scores   : vector={vec:.4f}  bm25={bm25:.4f}  graph={graph:.4f}")
        print(f"  │  Found in : {found_in}")
        print(f"  │  Preview  :")

        text = chunk.get("text", "").replace("\n", " ")
        words = text.split()
        line, lines = [], []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > 70:
                lines.append("  │    " + " ".join(line[:-1]))
                line = [word]
        if line:
            lines.append("  │    " + " ".join(line))
        print("\n".join(lines))
        print(f"  └{'─'*55}")


def main():
    parser = argparse.ArgumentParser(description="Query the retrieval layer.")
    parser.add_argument("query",   nargs="?", default=None, help="Query string")
    parser.add_argument("--top_k", type=int,  default=5,    help="Number of results (default: 5)")
    args = parser.parse_args()

    retriever = HybridRetriever(chroma_persist_dir=CHROMA_DIR, bm25_index_dir=BM25_DIR, graph_db_dir=GRAPH_DIR)

    if args.query:
        results = retriever.retrieve(args.query, top_k=args.top_k)
        print_results(results, args.query)
    else:
        print("\nInteractive mode — type 'exit' to quit, 'k=10' to change result count\n")
        top_k = args.top_k

        while True:
            try:
                query = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            if query.startswith("k="):
                try:
                    top_k = int(query[2:])
                    print(f"top_k set to {top_k}")
                    continue
                except ValueError:
                    pass

            results = retriever.retrieve(query, top_k=top_k)
            print_results(results, query)


if __name__ == "__main__":
    main()
