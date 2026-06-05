import os
import json
import math
from typing import List, Dict, Optional
from collections import defaultdict


class GraphStore:
    """
    Graph database over document chunks using NetworkX.

    Graph schema:
      - Chunk nodes   (type="chunk")   : one per document chunk
      - Entity nodes  (type="entity")  : one per unique named entity (NER)
      - Concept nodes (type="concept") : one per unique multi-word noun phrase
      - Edges chunk→entity/concept (rel="CONTAINS") : chunk mentions this signal
      - Edges chunk→chunk          (rel="NEXT")     : sequential chunks in same document

    Retrieval: given a query, extract named entities AND noun phrases, find chunks
    that share any of those signals, score by overlap count, return ranked list.
    """

    def __init__(self, graph_db_dir: str = "./graph_db"):
        self.graph_db_dir = graph_db_dir
        self.graph_path   = os.path.join(graph_db_dir, "graph.json")
        self.graph        = None   # networkx.DiGraph, populated by build() or load()
        self._nlp         = None   # spaCy pipeline, loaded lazily
        self._chunk_data: Dict[str, Dict] = {}   # chunk_id → chunk dict (fast lookup)

    # ── spaCy ──────────────────────────────────────────────────────────────────

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise OSError(
                    "spaCy model 'en_core_web_sm' not found.\n"
                    "Run: python -m spacy download en_core_web_sm"
                )
        return self._nlp

    def _extract_entities(self, text: str) -> List[tuple]:
        """Return list of (normalised_text, label) for named entities in text."""
        nlp = self._get_nlp()
        doc = nlp(text)
        seen = set()
        entities = []
        for ent in doc.ents:
            # Collapse all whitespace (tabs, newlines from PDF extraction) into single spaces
            key = " ".join(ent.text.split()).lower()
            if key and key not in seen:
                seen.add(key)
                entities.append((key, ent.label_))
        return entities

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Dict]) -> None:
        """Build the knowledge graph from a list of chunk dicts."""
        import networkx as nx

        self.graph = nx.DiGraph()
        self._chunk_data = {}

        nlp = self._get_nlp()
        doc_chunks: Dict[str, List[Dict]] = defaultdict(list)

        print(f"  Building graph: processing {len(chunks)} chunks...")

        # Process in batches for speed (spaCy pipe is faster than per-text calls).
        # "lemmatizer" is disabled for speed; parser must stay ON for noun_chunks.
        texts = [c["text"] for c in chunks]
        batch_size = 64
        # Each entry: {"entities": [(text, label), ...], "noun_phrases": [text, ...]}
        all_signals: List[Dict] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for doc in nlp.pipe(batch, disable=["lemmatizer"]):
                # Named entities — normalise whitespace
                seen_ents: set = set()
                ents: List[tuple] = []
                for ent in doc.ents:
                    key = " ".join(ent.text.split()).lower()
                    if key and key not in seen_ents:
                        seen_ents.add(key)
                        ents.append((key, ent.label_))

                # Noun phrases — multi-word only to avoid single generic nouns
                seen_nps: set = set()
                nps: List[str] = []
                for np in doc.noun_chunks:
                    key = " ".join(np.text.split()).lower()
                    if len(key.split()) >= 2 and key not in seen_nps:
                        seen_nps.add(key)
                        nps.append(key)

                all_signals.append({"entities": ents, "noun_phrases": nps})

        for chunk, signals in zip(chunks, all_signals):
            cid = chunk["chunk_id"]

            # Add chunk node — store only serialisable scalar metadata
            self.graph.add_node(
                cid,
                type         = "chunk",
                chunk_id     = cid,
                text         = chunk["text"],
                source       = chunk.get("source", ""),
                filename     = chunk.get("filename", ""),
                doc_type     = chunk.get("doc_type", ""),
                chunk_index  = chunk.get("chunk_index", 0),
                total_chunks = chunk.get("total_chunks", 0),
            )
            self._chunk_data[cid] = chunk
            doc_chunks[chunk.get("source", "")].append(chunk)

            # Add entity nodes and chunk→entity edges
            for ent_text, ent_label in signals["entities"]:
                eid = f"ent_{ent_text}"
                if not self.graph.has_node(eid):
                    self.graph.add_node(eid, type="entity", text=ent_text, label=ent_label)
                if self.graph.has_edge(cid, eid):
                    self.graph[cid][eid]["count"] += 1
                else:
                    self.graph.add_edge(cid, eid, rel="CONTAINS", count=1)

            # Add concept (noun phrase) nodes and chunk→concept edges
            for np_text in signals["noun_phrases"]:
                nid = f"np_{np_text}"
                if not self.graph.has_node(nid):
                    self.graph.add_node(nid, type="concept", text=np_text)
                if self.graph.has_edge(cid, nid):
                    self.graph[cid][nid]["count"] += 1
                else:
                    self.graph.add_edge(cid, nid, rel="CONTAINS", count=1)

        # Add sequential NEXT edges within each document
        for source, doc_chunk_list in doc_chunks.items():
            ordered = sorted(doc_chunk_list, key=lambda c: c.get("chunk_index", 0))
            for i in range(len(ordered) - 1):
                src_id = ordered[i]["chunk_id"]
                dst_id = ordered[i + 1]["chunk_id"]
                self.graph.add_edge(src_id, dst_id, rel="NEXT")

        n_chunks   = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "chunk")
        n_entities = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "entity")
        n_concepts = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "concept")
        print(f"  Graph built: {n_chunks} chunk nodes, {n_entities} entity nodes, "
              f"{n_concepts} concept nodes, {self.graph.number_of_edges()} edges")

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the graph to graph_db/graph.json."""
        if self.graph is None:
            raise RuntimeError("No graph to save. Call build() first.")
        from networkx.readwrite import json_graph
        os.makedirs(self.graph_db_dir, exist_ok=True)
        data = json_graph.node_link_data(self.graph)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  Graph saved → {self.graph_path}")

    def load(self) -> bool:
        """
        Load the graph from disk.
        Returns True on success, False if the file doesn't exist (no crash).
        """
        if not os.path.exists(self.graph_path):
            return False
        try:
            import networkx as nx
            from networkx.readwrite import json_graph
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = json_graph.node_link_graph(data)
            # Rebuild fast chunk lookup
            self._chunk_data = {
                nid: dict(attrs)
                for nid, attrs in self.graph.nodes(data=True)
                if attrs.get("type") == "chunk"
            }
            return True
        except Exception as e:
            print(f"  WARNING: Could not load graph ({e}). Graph retrieval disabled.")
            self.graph = None
            return False

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Find chunks relevant to query via named entity + noun phrase matching.

        Steps:
          1. Run spaCy on the query once; extract NER entities AND noun phrases.
          2. Build a combined set of graph node IDs to look up (ent_* and np_*).
          3. For each signal node, find predecessor chunk nodes via CONTAINS edges.
          4. Score each chunk by how many query signals it shares (normalised 0-1).
          5. Return top_k chunks sorted by graph_score descending.

        Returns [] if graph not loaded or no signals found in query.
        """
        if self.graph is None:
            return []

        nlp = self._get_nlp()
        doc = nlp(query)

        # Named entities
        signal_ids: set = set()
        for ent in doc.ents:
            key = " ".join(ent.text.split()).lower()
            if key:
                signal_ids.add(f"ent_{key}")

        # Noun phrases (multi-word only — same filter as build)
        for np in doc.noun_chunks:
            key = " ".join(np.text.split()).lower()
            if len(key.split()) >= 2:
                signal_ids.add(f"np_{key}")

        if not signal_ids:
            return []

        # IDF-weighted scoring: rare signals (few chunks contain them) count more
        # than common ones (e.g. "baseball bat" > "the author").
        # IDF = 1 / log(1 + degree), where degree = number of chunks containing this signal.
        # graph_score = sum(idf for matched signals) / sum(idf for all query signals in graph)
        chunk_scores: Dict[str, float] = defaultdict(float)
        total_idf = 0.0

        for sid in signal_ids:
            if not self.graph.has_node(sid):
                continue
            degree = self.graph.in_degree(sid)          # chunks containing this signal
            idf = 1.0 / math.log(1.0 + degree)         # rare → high weight; common → low
            total_idf += idf
            # Predecessors of a signal node are chunk nodes (chunk→signal edges)
            for chunk_id in self.graph.predecessors(sid):
                if self.graph.nodes[chunk_id].get("type") == "chunk":
                    chunk_scores[chunk_id] += idf

        if not chunk_scores or total_idf == 0:
            return []

        # Normalise: graph_score = fraction of total possible IDF weight matched
        for chunk_id in chunk_scores:
            chunk_scores[chunk_id] = round(chunk_scores[chunk_id] / total_idf, 6)

        # Build result list sorted by graph_score descending
        results = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
            if chunk_id not in self._chunk_data:
                continue
            chunk = dict(self._chunk_data[chunk_id])
            chunk["graph_score"] = score
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results
