import os
import json
import math
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ── Tuning constants ──────────────────────────────────────────────────────────

FUZZY_SIM_THRESHOLD  = 0.80   # min cosine for a query signal to resolve to an entity
FUZZY_TOP_N          = 3      # how many fuzzy entity matches a single signal may resolve to
PPR_ALPHA            = 0.75   # PageRank damping — lower = mass stays nearer the seeds
PPR_HOPS             = 2      # how far the traversal subgraph expands from the seeds
MAX_SUBGRAPH_NODES   = 8000   # cap on the traversal subgraph (bounds per-query cost)
MAX_SCORING_ENTITIES = 300    # only the heaviest entities contribute chunk evidence
PPR_MASS_EPSILON     = 1e-6   # ignore entities below this PageRank mass
CO_OCCUR_WEIGHT      = 0.3    # co-occurrence edges carry less signal than parsed relations
REVERSE_EDGE_FACTOR  = 0.5    # relation edges also conduct backwards, at reduced weight
OCR_EVIDENCE_FACTOR  = 0.8    # OCR'd chunks are noisier — discount their evidence slightly
MIN_CONFIDENCE       = 0.15   # below this, the caller should ignore graph results entirely

SCHEMA_VERSION = 2            # bumped when the on-disk format changes


class GraphStore:
    """
    Knowledge graph over document chunks using NetworkX.

    Graph schema:
      - Entity nodes (type="entity") : one per canonical concept, keyed by a
        lemma-normalised string. Carries df (chunk frequency) and idf.
      - Chunk nodes  (type="chunk")  : evidence that an entity was mentioned.
      - entity→chunk  (rel="MENTIONS", count=tf)
      - entity→entity (rel="RELATED", labels=[...], weight) : subject-verb-object
        triples from the dependency parse, plus sentence-level co-occurrence.
      - chunk→chunk   (rel="NEXT")   : sequential chunks in the same document.

    Retrieval resolves query text to seed entities (exact, then embedding-fuzzy),
    runs personalised PageRank over the entity graph so relevance flows across
    relation edges, then collects the chunks the reachable entities are evidenced
    by. Because mass travels 2 hops, this surfaces chunks containing none of the
    query's own terms — which is the whole point of having a graph arm alongside
    vector and BM25 search.
    """

    def __init__(self, graph_db_dir: str = "./graph_db"):
        self.graph_db_dir = graph_db_dir
        self.graph_path    = os.path.join(graph_db_dir, "graph.json")
        self.vocab_path    = os.path.join(graph_db_dir, "entities.json")
        self.emb_path      = os.path.join(graph_db_dir, "entity_emb.npy")

        self.graph         = None   # networkx.DiGraph, populated by build() or load()
        self._nlp          = None   # spaCy pipeline, loaded lazily
        self._embedder     = None   # SentenceTransformer, loaded lazily and shared
        self._chunk_data: Dict[str, Dict] = {}   # chunk_id → chunk dict (fast lookup)

        self._entity_graph = None   # entity-only view used for PageRank traversal
        self._vocab: List[str] = [] # canonical entity keys, index-aligned with _emb
        self._emb          = None   # np.ndarray (len(vocab), dim), L2-normalised
        self._n_chunks     = 0

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

    def _get_embedder(self):
        """Reuse the same MiniLM instance the vector store already keeps in memory."""
        if self._embedder is None:
            from .vector_store import _get_shared_model, EMBEDDING_MODEL
            self._embedder = _get_shared_model(EMBEDDING_MODEL)
        return self._embedder

    def _clean_text_for_nlp(self, text: str) -> str:
        """
        Remove PDF section-number noise where digits are glued to words.
        e.g. '13Combination Cooking' → 'Combination Cooking'
             '12Grill' → 'Grill'
        Only strips digits that are immediately followed by a letter (no space).
        Leaves legitimate numbers like 'Chapter 13' or '100%' untouched.
        """
        return re.sub(r'\b\d+([A-Za-z])', r'\1', text)

    # ── Normalisation ──────────────────────────────────────────────────────────
    #
    # Every entity key — at build time and at query time — passes through here.
    # This is what makes 'the Ovens', 'ovens' and 'oven' a single node, which the
    # previous exact-string schema could not do.

    @staticmethod
    def _norm_tokens(tokens, keep_stopwords: bool = False) -> str:
        """Lemma-lowercase a token sequence, dropping determiners and punctuation."""
        parts = []
        for t in tokens:
            if t.is_punct or t.is_space or t.like_num:
                continue
            if t.pos_ == "DET" or t.pos_ == "PRON":
                continue
            if not keep_stopwords and t.is_stop:
                continue
            lemma = t.lemma_.lower().strip()
            lemma = re.sub(r"[^a-z0-9\-/&' ]", "", lemma).strip()
            if lemma:
                parts.append(lemma)
        return " ".join(parts)

    def _norm_entity_span(self, span) -> str:
        """Normalise a named-entity span. Internal stopwords are kept ('bank of england')."""
        return self._norm_tokens(span, keep_stopwords=True)

    def _norm_noun_phrase(self, span) -> str:
        """Normalise a noun chunk down to its content words ('the big oven' → 'big oven')."""
        content = [t for t in span if t.pos_ in ("NOUN", "PROPN", "ADJ")]
        return self._norm_tokens(content, keep_stopwords=False)

    @staticmethod
    def _is_usable_key(key: str) -> bool:
        return bool(key) and len(key) >= 3 and len(key.split()) <= 6

    # ── Signal extraction ──────────────────────────────────────────────────────

    def _extract_signals(self, doc) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Pull entity keys and relation triples out of one parsed spaCy doc.

        Returns:
            keys    — list of canonical entity keys mentioned (with repeats, so the
                      caller can compute term frequency)
            triples — list of (subject_key, relation_label, object_key)
        """
        keys: List[str] = []

        # Map every token index to the noun chunk containing it, so a dependency
        # token like 'oven' resolves to the full concept 'convection oven'.
        tok2phrase: Dict[int, str] = {}

        for np in doc.noun_chunks:
            phrase = self._norm_noun_phrase(np)
            if self._is_usable_key(phrase):
                keys.append(phrase)
                for t in np:
                    tok2phrase[t.i] = phrase
            # The bare head noun is indexed too. The old schema required >=2 words,
            # so single-word concepts ('oven', 'voltage') had no node at all.
            head = np.root
            if head.pos_ in ("NOUN", "PROPN"):
                head_key = self._norm_tokens([head], keep_stopwords=True)
                if self._is_usable_key(head_key):
                    keys.append(head_key)
                    tok2phrase.setdefault(head.i, head_key)

        for ent in doc.ents:
            ent_key = self._norm_entity_span(ent)
            if self._is_usable_key(ent_key):
                keys.append(ent_key)
                for t in ent:
                    tok2phrase[t.i] = ent_key

        def resolve(token) -> Optional[str]:
            if token.i in tok2phrase:
                return tok2phrase[token.i]
            if token.pos_ in ("NOUN", "PROPN"):
                k = self._norm_tokens([token], keep_stopwords=True)
                return k if self._is_usable_key(k) else None
            return None

        # ── Dependency-parse triples (offline substitute for LLM extraction) ──
        triples: List[Tuple[str, str, str]] = []

        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            subjects, objects = [], []
            passive_subjects, agents = [], []

            for child in token.children:
                dep = child.dep_
                if dep == "nsubj":
                    subjects.append(child)
                elif dep == "nsubjpass":
                    passive_subjects.append(child)
                elif dep in ("dobj", "dative", "attr", "oprd"):
                    objects.append(child)
                elif dep == "agent":
                    # 'X was built by Y' — the real subject hangs off 'by'
                    agents.extend(c for c in child.children if c.dep_ == "pobj")
                elif dep == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            objects.append(pobj)

            # Copular 'is a' relations carry taxonomy, which is worth labelling.
            label = "is_a" if token.lemma_ == "be" else token.lemma_.lower()

            for subj in subjects:
                s = resolve(subj)
                if not s:
                    continue
                for obj in objects:
                    o = resolve(obj)
                    if o and o != s:
                        triples.append((s, label, o))

            # Passive: 'the door was opened by the technician' → technician opened door
            for subj in passive_subjects:
                o = resolve(subj)
                if not o:
                    continue
                for ag in agents:
                    s = resolve(ag)
                    if s and s != o:
                        triples.append((s, token.lemma_.lower(), o))

        # ── Sentence-level co-occurrence ──
        # Restricted to a single sentence rather than the whole chunk: it keeps the
        # edge count linear-ish and the association is far more meaningful.
        for sent in doc.sents:
            sent_keys = []
            seen = set()
            for t in sent:
                k = tok2phrase.get(t.i)
                if k and k not in seen:
                    seen.add(k)
                    sent_keys.append(k)
            for i in range(len(sent_keys)):
                for j in range(i + 1, len(sent_keys)):
                    triples.append((sent_keys[i], "__cooccur__", sent_keys[j]))

        return keys, triples

    def _extract_query_signals(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Query-side counterpart of _extract_signals.

        Returns (primary, fallback):
          primary  — keys from noun chunks / heads / named entities. Only these
                     count towards confidence.
          fallback — raw n-grams over content tokens, kept from the original
                     implementation because spaCy sometimes fails to produce a
                     useful noun chunk for short imperative queries
                     ('tell me about combination cooking' → only 'me'). These add
                     recall but are deliberately excluded from the confidence
                     denominator, which is what used to drag graph_score down.
        """
        nlp = self._get_nlp()
        doc = nlp(self._clean_text_for_nlp(query))

        primary, seen = [], set()
        keys, _ = self._extract_signals(doc)
        for k in keys:
            if k not in seen:
                seen.add(k)
                primary.append(k)

        content = [t for t in doc if t.pos_ in ("NOUN", "PROPN", "ADJ") and not t.is_stop]
        norm_tokens = [self._norm_tokens([t], keep_stopwords=True) for t in content]
        norm_tokens = [t for t in norm_tokens if t]

        fallback = []
        for n in (1, 2, 3):
            for i in range(len(norm_tokens) - n + 1):
                gram = " ".join(norm_tokens[i:i + n])
                if self._is_usable_key(gram) and gram not in seen:
                    seen.add(gram)
                    fallback.append(gram)

        return primary, fallback

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Dict]) -> None:
        """Build the knowledge graph from a list of chunk dicts."""
        import networkx as nx

        self.graph = nx.DiGraph()
        self._chunk_data = {}
        self._n_chunks = len(chunks)

        nlp = self._get_nlp()
        doc_chunks: Dict[str, List[Dict]] = defaultdict(list)

        print(f"  Building graph: parsing {len(chunks)} chunks...")

        texts = [self._clean_text_for_nlp(c["text"]) for c in chunks]
        parsed = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            for doc in nlp.pipe(texts[start : start + batch_size]):
                parsed.append(self._extract_signals(doc))
            if (start // batch_size) % 10 == 0 and start:
                print(f"    parsed {min(start + batch_size, len(texts))}/{len(texts)}")

        # entity_key → {chunk_id: term frequency}
        mentions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # (subject, object) → {"labels": set, "count": int, "cooccur": int}
        relations: Dict[Tuple[str, str], Dict] = {}

        for chunk, (keys, triples) in zip(chunks, parsed):
            cid = chunk["chunk_id"]

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
                page         = chunk.get("page"),
                ocr          = bool(chunk.get("ocr", False)),
            )
            self._chunk_data[cid] = chunk
            doc_chunks[chunk.get("source", "")].append(chunk)

            for k in keys:
                mentions[k][cid] += 1

            for s, label, o in triples:
                if s == o:
                    continue
                pair = (s, o)
                rec = relations.get(pair)
                if rec is None:
                    rec = {"labels": set(), "count": 0, "cooccur": 0}
                    relations[pair] = rec
                if label == "__cooccur__":
                    rec["cooccur"] += 1
                else:
                    rec["labels"].add(label)
                    rec["count"] += 1

        # ── Entity nodes with IDF ──
        n = max(1, self._n_chunks)
        for key, chunk_tfs in mentions.items():
            df = len(chunk_tfs)
            self.graph.add_node(
                f"e:{key}",
                type = "entity",
                text = key,
                df   = df,
                # Textbook IDF. The old 1/log(1+degree) only separated a
                # 1-chunk term from a 100-chunk term by ~6x, which was far too
                # flat to let distinctive entities outrank generic phrases.
                idf  = round(math.log(1.0 + n / df), 6),
            )
            for cid, tf in chunk_tfs.items():
                self.graph.add_edge(f"e:{key}", cid, rel="MENTIONS", count=tf)

        # ── Entity→entity edges ──
        n_rel = 0
        for (s, o), rec in relations.items():
            s_id, o_id = f"e:{s}", f"e:{o}"
            if not (self.graph.has_node(s_id) and self.graph.has_node(o_id)):
                continue
            weight = rec["count"] + CO_OCCUR_WEIGHT * rec["cooccur"]
            if weight <= 0:
                continue
            self.graph.add_edge(
                s_id, o_id,
                rel    = "RELATED",
                labels = sorted(rec["labels"]),
                weight = round(float(weight), 4),
            )
            n_rel += 1

        # Sequential NEXT edges within each document
        for _, doc_chunk_list in doc_chunks.items():
            ordered = sorted(doc_chunk_list, key=lambda c: c.get("chunk_index", 0))
            for i in range(len(ordered) - 1):
                self.graph.add_edge(
                    ordered[i]["chunk_id"], ordered[i + 1]["chunk_id"], rel="NEXT"
                )

        # ── Entity embeddings, for fuzzy query resolution ──
        self._vocab = sorted(mentions.keys())
        if self._vocab:
            print(f"  Embedding {len(self._vocab)} entity keys for fuzzy matching...")
            model = self._get_embedder()
            self._emb = model.encode(
                self._vocab,
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")

        self._build_entity_graph()

        n_entities = len(self._vocab)
        typed = sum(1 for _, _, d in self.graph.edges(data=True)
                    if d.get("rel") == "RELATED" and d.get("labels"))
        print(f"  Graph built: {self._n_chunks} chunk nodes, {n_entities} entity nodes, "
              f"{n_rel} entity-entity edges ({typed} carrying parsed relations), "
              f"{self.graph.number_of_edges()} edges total")

    def _build_entity_graph(self) -> None:
        """
        Project the entity-only subgraph used for PageRank traversal.

        Relation edges also conduct backwards at reduced weight, so relevance can
        reach a subject from its object without the traversal being direction-blind.
        """
        import networkx as nx

        eg = nx.DiGraph()
        eg.add_nodes_from(nid for nid, d in self.graph.nodes(data=True)
                          if d.get("type") == "entity")

        for u, v, d in self.graph.edges(data=True):
            if d.get("rel") != "RELATED":
                continue
            w = float(d.get("weight", 1.0))
            eg.add_edge(u, v, weight=w)
            if not eg.has_edge(v, u):
                eg.add_edge(v, u, weight=w * REVERSE_EDGE_FACTOR)

        self._entity_graph = eg

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the graph, entity vocabulary and entity embeddings."""
        if self.graph is None:
            raise RuntimeError("No graph to save. Call build() first.")
        import numpy as np

        os.makedirs(self.graph_db_dir, exist_ok=True)

        # Hand-rolled node/edge serialisation rather than networkx's node_link_data,
        # whose default key names have shifted between networkx versions.
        data = {
            "schema_version": SCHEMA_VERSION,
            "n_chunks":       self._n_chunks,
            "nodes":          [{"id": nid, **attrs} for nid, attrs in self.graph.nodes(data=True)],
            "edges":          [{"u": u, "v": v, **attrs} for u, v, attrs in self.graph.edges(data=True)],
        }
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self._vocab}, f)

        if self._emb is not None:
            np.save(self.emb_path, self._emb.astype("float16"))

        print(f"  Graph saved → {self.graph_path}")

    def load(self) -> bool:
        """
        Load the graph from disk.
        Returns True on success, False if missing or stale (no crash — the caller
        degrades to vector+BM25 retrieval).
        """
        if not os.path.exists(self.graph_path):
            return False
        try:
            import networkx as nx
            import numpy as np

            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("schema_version") != SCHEMA_VERSION:
                print(
                    f"  WARNING: graph_db uses an old schema (v{data.get('schema_version')}, "
                    f"expected v{SCHEMA_VERSION}). Graph retrieval disabled.\n"
                    "  Re-run: python ingest_documents.py --force"
                )
                self.graph = None
                return False

            g = nx.DiGraph()
            for node in data["nodes"]:
                attrs = dict(node)
                g.add_node(attrs.pop("id"), **attrs)
            for edge in data["edges"]:
                attrs = dict(edge)
                u, v = attrs.pop("u"), attrs.pop("v")
                g.add_edge(u, v, **attrs)

            self.graph      = g
            self._n_chunks  = data.get("n_chunks", 0)
            self._chunk_data = {
                nid: dict(attrs)
                for nid, attrs in g.nodes(data=True)
                if attrs.get("type") == "chunk"
            }

            if os.path.exists(self.vocab_path):
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    self._vocab = json.load(f).get("vocab", [])
            if os.path.exists(self.emb_path):
                self._emb = np.load(self.emb_path).astype("float32")

            self._build_entity_graph()
            return True
        except Exception as e:
            print(f"  WARNING: Could not load graph ({e}). Graph retrieval disabled.")
            self.graph = None
            return False

    # ── Query resolution ───────────────────────────────────────────────────────

    def _resolve(self, keys: List[str], allow_fuzzy: bool) -> Tuple[Dict[str, float], int, float]:
        """
        Map query keys onto seed entities.

        Exact node hits resolve at similarity 1.0. Anything else falls back to
        nearest-neighbour lookup over the entity embeddings — the old schema did
        exact string comparison only, so a single character of drift scored zero.

        Returns (seed_weights, n_resolved, similarity_sum).
        """
        seeds: Dict[str, float] = defaultdict(float)
        n_resolved = 0
        sim_sum    = 0.0

        fuzzy_queue = []
        for k in keys:
            nid = f"e:{k}"
            if self.graph.has_node(nid):
                seeds[nid] += self.graph.nodes[nid].get("idf", 1.0)
                n_resolved += 1
                sim_sum    += 1.0
            elif allow_fuzzy:
                fuzzy_queue.append(k)

        if fuzzy_queue and self._emb is not None and len(self._vocab):
            import numpy as np
            model = self._get_embedder()
            qv = model.encode(fuzzy_queue, normalize_embeddings=True).astype("float32")
            sims = qv @ self._emb.T                       # (len(queue), len(vocab))

            top_n = min(FUZZY_TOP_N, sims.shape[1])
            for row in sims:
                idx = np.argpartition(-row, top_n - 1)[:top_n]
                hit_best = 0.0
                for i in idx:
                    sim = float(row[i])
                    if sim < FUZZY_SIM_THRESHOLD:
                        continue
                    nid = f"e:{self._vocab[i]}"
                    if self.graph.has_node(nid):
                        # Squared so a marginal match contributes markedly less.
                        seeds[nid] += self.graph.nodes[nid].get("idf", 1.0) * sim * sim
                        hit_best = max(hit_best, sim)
                if hit_best > 0:
                    n_resolved += 1
                    sim_sum    += hit_best

        return dict(seeds), n_resolved, sim_sum

    def _traverse(self, seeds: Dict[str, float]) -> Dict[str, float]:
        """
        Personalised PageRank from the seed entities over the relation graph.

        This is what makes the arm non-redundant with BM25: mass flows across
        relation edges, so chunks that never contain a query term but are
        relationally connected to it still surface.
        """
        import networkx as nx

        eg = self._entity_graph
        if eg is None or eg.number_of_nodes() == 0:
            return dict(seeds)

        present = [s for s in seeds if eg.has_node(s)]
        if not present:
            return dict(seeds)

        # Expand a bounded neighbourhood instead of running PageRank over the whole
        # entity graph — with alpha this low, mass beyond PPR_HOPS is negligible.
        frontier = set(present)
        nodes    = set(present)
        for _ in range(PPR_HOPS):
            nxt = set()
            for u in frontier:
                nxt.update(eg.successors(u))
                nxt.update(eg.predecessors(u))
            nxt -= nodes
            if not nxt:
                break
            room = MAX_SUBGRAPH_NODES - len(nodes)
            if room <= 0:
                break
            if len(nxt) > room:
                nxt = set(list(nxt)[:room])
            nodes.update(nxt)
            frontier = nxt

        sub = eg.subgraph(nodes)
        pers = {node: seeds[node] for node in present}

        try:
            ranks = nx.pagerank(
                sub, alpha=PPR_ALPHA, personalization=pers,
                weight="weight", max_iter=50, tol=1e-6,
            )
        except Exception:
            # Power iteration can fail to converge on pathological graphs; the
            # seeds alone still give usable (1-hop) results.
            return dict(seeds)

        # Keep the seeds' own weight in play — traversal should extend the seed
        # matches, not dilute them.
        for node, w in seeds.items():
            ranks[node] = ranks.get(node, 0.0) + w
        return ranks

    # ── Search ─────────────────────────────────────────────────────────────────

    def search_with_confidence(self, query: str, top_k: int = 20) -> Tuple[List[Dict], float]:
        """
        Graph retrieval. Returns (results, confidence).

        confidence is the fraction of the query's *primary* concepts that resolved
        to graph entities, weighted by how well they resolved. Callers should scale
        the graph's fusion weight by it so entity-free queries stop contributing
        noise at full weight.
        """
        if self.graph is None:
            return [], 0.0

        primary, fallback = self._extract_query_signals(query)
        if not primary and not fallback:
            return [], 0.0

        seeds, n_res, sim_sum = self._resolve(primary, allow_fuzzy=True)

        # Confidence is measured over primary signals only. Folding the n-gram
        # fallback into the denominator is what made the old graph_score
        # structurally tiny regardless of match quality.
        confidence = 0.0
        if primary:
            coverage = n_res / len(primary)
            mean_sim = (sim_sum / n_res) if n_res else 0.0
            confidence = round(coverage * mean_sim, 4)

        # Fallback n-grams add recall without touching confidence.
        fb_seeds, _, _ = self._resolve(fallback, allow_fuzzy=False)
        for nid, w in fb_seeds.items():
            seeds[nid] = seeds.get(nid, 0.0) + w * 0.5

        if not seeds:
            return [], 0.0

        ranks = self._traverse(seeds)

        # ── Collect chunk evidence ──
        ranked_entities = sorted(
            ((nid, m) for nid, m in ranks.items() if m > PPR_MASS_EPSILON),
            key=lambda x: x[1], reverse=True,
        )[:MAX_SCORING_ENTITIES]

        chunk_scores: Dict[str, float] = defaultdict(float)
        for nid, mass in ranked_entities:
            if not self.graph.has_node(nid):
                continue
            idf = self.graph.nodes[nid].get("idf", 1.0)
            for cid in self.graph.successors(nid):
                edge = self.graph[nid][cid]
                if edge.get("rel") != "MENTIONS":
                    continue
                tf = edge.get("count", 1)
                chunk_scores[cid] += mass * idf * (1.0 + math.log(tf))

        if not chunk_scores:
            return [], confidence

        # Length-normalise so verbose chunks stop winning purely on volume, and
        # discount OCR'd text, whose entities are noisier.
        for cid in list(chunk_scores):
            n_ents = sum(
                1 for p in self.graph.predecessors(cid)
                if self.graph[p][cid].get("rel") == "MENTIONS"
            )
            chunk_scores[cid] /= math.sqrt(max(1, n_ents))
            if self.graph.nodes[cid].get("ocr"):
                chunk_scores[cid] *= OCR_EVIDENCE_FACTOR

        top = max(chunk_scores.values()) or 1.0

        results = []
        for cid, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
            if cid not in self._chunk_data:
                continue
            chunk = dict(self._chunk_data[cid])
            chunk["graph_score"] = round(score / top, 6)
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results, confidence

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Backwards-compatible wrapper — see search_with_confidence()."""
        results, _ = self.search_with_confidence(query, top_k=top_k)
        return results
