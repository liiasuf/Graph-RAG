from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag_system.graph.entities import extract_entities, normalize_entity

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None


_FULLTEXT_INDEX_NAME = "rag_system_entity_display"


@dataclass
class Neo4jGraphIndex:
    """
    Entity co-occurrence graph stored in Neo4j (stage 3 MVP).

    Schema (all nodes/relationships tagged with `experiment` so multiple
    experiments can share one Neo4j instance without clashing):

        (:Entity {key, display, experiment})
        (:Chunk  {chunk_id, doc_id, title, experiment})

        (:Entity)-[:MENTIONS]->(:Chunk)
        (:Entity)-[:CO_OCCURS {weight}]->(:Entity)   -- stored in both directions

    This mirrors `rag_system.graph.graph_index.GraphIndex` (same entity
    extraction via `extract_entities`/`normalize_entity`, same
    co-occurrence semantics): the only difference is that the graph lives in
    Neo4j and is traversed via Cypher instead of plain Python dicts. This
    keeps the "graph" and "graph_neo4j" retrieval modes directly comparable
    (see H5 in docs/hypotheses.md).
    """

    driver: object
    database: str
    experiment: str

    @classmethod
    def connect(cls, uri, user, password, database, experiment):
        """
        Open a driver connection to Neo4j. Raises `ImportError` with a
        helpful message if the `neo4j` package is not installed
        (`pip install -e .[neo4j]`).
        """

        if GraphDatabase is None:
            raise ImportError(
                "The 'neo4j' package is required for retrieval.mode == 'graph_neo4j'. "
                "Install it with: pip install -e .[neo4j]"
            )

        driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls(driver=driver, database=database, experiment=experiment)

    def close(self) -> None:
        self.driver.close()


    def ensure_indexes(self) -> None:
        """
        Create the indexes/full-text index used by `build`/`match_entities`
        if they don't already exist. Safe to call on every run.
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE INDEX rag_system_entity_key IF NOT EXISTS "
                "FOR (e:Entity) ON (e.key, e.experiment)"
            )
            session.run(
                "CREATE INDEX rag_system_chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.chunk_id, c.experiment)"
            )
            # The composite indexes above only help lookups that constrain
            # `key`/`chunk_id` first - clear() and any other query that
            # filters ONLY on `experiment` (across all :Entity/:Chunk nodes
            # of one experiment, regardless of key/chunk_id) can't use them
            # and falls back to a full label/store scan. With multiple
            # experiments accumulating in the same database (millions of
            # nodes total), that scan alone can exceed
            # dbms.memory.transaction.total.max even when the experiment
            # being cleared is small. A plain single-property index per
            # label makes those experiment-only lookups a real index seek.
            session.run(
                "CREATE INDEX rag_system_entity_experiment IF NOT EXISTS "
                "FOR (e:Entity) ON (e.experiment)"
            )
            session.run(
                "CREATE INDEX rag_system_chunk_experiment IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.experiment)"
            )
            session.run(
                f"CREATE FULLTEXT INDEX {_FULLTEXT_INDEX_NAME} IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.display]"
            )

    def clear(self, *, batch_size: int = 1000) -> None:
        """
        Delete all :Entity/:Chunk nodes (and their relationships) tagged
        with this index's `experiment`. Used when
        `graph.neo4j.clear_before_build` is true, so re-running an
        experiment doesn't accumulate duplicate graphs.

        Three batched passes (Neo4j 5's `IN TRANSACTIONS` subquery, no APOC
        needed) rather than one giant DETACH DELETE transaction:
          1. delete CO_OCCURS/MENTIONS relationships (anchored at :Entity,
             which is always at least one endpoint of both relationship
             types in this schema) in batches
          2. delete the now-relationship-free :Entity nodes in batches
          3. delete the now-relationship-free :Chunk nodes in batches
        A single `MATCH (n) DETACH DELETE n` (even batched by node count)
        still has to delete every relationship attached to each node
        within that same batch's transaction - on a large uncapped-corpus
        graph, hub entities can have thousands of CO_OCCURS edges each, so
        a handful of high-degree nodes in one batch can still blow past
        `dbms.memory.transaction.total.max`. Splitting into a
        relationships-only pass and a nodes-only pass keeps each batch's
        transaction small regardless of degree distribution.

        Every pattern below matches `(:Label {experiment: $experiment})`
        (not a label-less `MATCH (n) WHERE n.experiment = ...`) specifically
        so the planner can use the single-property `experiment` indexes
        from `ensure_indexes()` - with multiple experiments' data
        accumulated in the same database, a label-less/unindexed match has
        to scan every node regardless of label, which alone can exceed the
        transaction memory limit even when the experiment being cleared is
        small.
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (e:Entity {experiment: $experiment})-[r]-()
                CALL (r) { DELETE r } IN TRANSACTIONS OF $batch_size ROWS
                """,
                experiment=self.experiment,
                batch_size=batch_size,
            )
            session.run(
                """
                MATCH (e:Entity {experiment: $experiment})
                CALL (e) { DELETE e } IN TRANSACTIONS OF $batch_size ROWS
                """,
                experiment=self.experiment,
                batch_size=batch_size,
            )
            session.run(
                """
                MATCH (c:Chunk {experiment: $experiment})
                CALL (c) { DELETE c } IN TRANSACTIONS OF $batch_size ROWS
                """,
                experiment=self.experiment,
                batch_size=batch_size,
            )


    def build(
        self,
        chunks,
        *,
        title_as_entity: bool = True,
        max_entity_words: int = 5,
        min_entity_chars: int = 3,
        batch_size: int = 1000,
    ) -> dict:
        """
        Extract entities from `chunks` (same heuristic as the in-memory
        `GraphIndex.build`) and push the resulting graph into Neo4j as
        :Entity/:Chunk nodes plus :MENTIONS/:CO_OCCURS relationships.

        Returns the same shape as `GraphIndex.stats()`:
        `{"n_entities", "n_chunks", "n_edges"}`.
        """

        self.ensure_indexes()

        entity_chunks: dict[str, set[str]] = defaultdict(set)
        entity_neighbors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        entity_display: dict[str, str] = {}
        chunk_rows: list[dict] = []

        for c in chunks:
            mentions = extract_entities(
                c.text, max_words=max_entity_words, min_chars=min_entity_chars
            )

            if title_as_entity and c.title:
                mentions.add(c.title)

            keys: set[str] = set()
            for mention in mentions:
                key = normalize_entity(mention)
                if not key:
                    continue
                keys.add(key)
                entity_display.setdefault(key, mention)
                entity_chunks[key].add(c.chunk_id)

            keys_list = sorted(keys)
            for i, a in enumerate(keys_list):
                for b in keys_list[i + 1 :]:
                    entity_neighbors[a][b] += 1
                    entity_neighbors[b][a] += 1

            chunk_rows.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title})

        entity_rows = [{"key": k, "display": v} for k, v in entity_display.items()]

        mentions_rows = [
            {"key": key, "chunk_id": chunk_id}
            for key, chunk_ids in entity_chunks.items()
            for chunk_id in chunk_ids
        ]

        edge_rows = [
            {"a": a, "b": b, "weight": w}
            for a, neigh in entity_neighbors.items()
            for b, w in neigh.items()
        ]

        with self.driver.session(database=self.database) as session:
            for batch in _chunked(chunk_rows, batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (c:Chunk {chunk_id: row.chunk_id, experiment: $experiment})
                    SET c.doc_id = row.doc_id, c.title = row.title
                    """,
                    rows=batch,
                    experiment=self.experiment,
                )

            for batch in _chunked(entity_rows, batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Entity {key: row.key, experiment: $experiment})
                    SET e.display = row.display
                    """,
                    rows=batch,
                    experiment=self.experiment,
                )

            for batch in _chunked(mentions_rows, batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (e:Entity {key: row.key, experiment: $experiment})
                    MATCH (c:Chunk {chunk_id: row.chunk_id, experiment: $experiment})
                    MERGE (e)-[:MENTIONS]->(c)
                    """,
                    rows=batch,
                    experiment=self.experiment,
                )

            for batch in _chunked(edge_rows, batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Entity {key: row.a, experiment: $experiment})
                    MATCH (b:Entity {key: row.b, experiment: $experiment})
                    MERGE (a)-[r:CO_OCCURS]->(b)
                    SET r.weight = row.weight
                    """,
                    rows=batch,
                    experiment=self.experiment,
                )

        return self.stats()


    def stats(self) -> dict:
        with self.driver.session(database=self.database) as session:
            n_entities = session.run(
                "MATCH (e:Entity {experiment: $experiment}) RETURN count(e) AS n",
                experiment=self.experiment,
            ).single()["n"]

            n_chunks = session.run(
                "MATCH (c:Chunk {experiment: $experiment}) RETURN count(c) AS n",
                experiment=self.experiment,
            ).single()["n"]

            n_edges = session.run(
                "MATCH (:Entity {experiment: $experiment})"
                "-[r:CO_OCCURS]->(:Entity {experiment: $experiment}) "
                "RETURN count(r) AS n",
                experiment=self.experiment,
            ).single()["n"]

        return {
            "n_entities": n_entities,
            "n_chunks": n_chunks,
            "n_edges": n_edges // 2,
        }

    def match_entities(
        self, mention: str, *, min_overlap: float = 0.5, max_candidates: int = 5
    ) -> list[str]:
        """
        Resolve a free-text mention to graph entity keys: exact normalised
        match first, then fuzzy match via the full-text index on
        `Entity.display`, scored by token-overlap (same `min_overlap`
        semantics as `GraphIndex.match_entities`).
        """

        key = normalize_entity(mention)

        with self.driver.session(database=self.database) as session:
            exact = session.run(
                "MATCH (e:Entity {key: $key, experiment: $experiment}) RETURN e.key AS key",
                key=key,
                experiment=self.experiment,
            ).single()

            if exact is not None:
                return [exact["key"]]

            tokens = [t for t in key.split() if len(t) > 2]
            if not tokens:
                return []

            lucene_query = " OR ".join(tokens)
            records = session.run(
                f"""
                CALL db.index.fulltext.queryNodes('{_FULLTEXT_INDEX_NAME}', $search_query)
                YIELD node, score
                WHERE node.experiment = $experiment
                RETURN node.key AS key
                LIMIT 50
                """,
                search_query=lucene_query,
                experiment=self.experiment,
            )

            scored = []
            for rec in records:
                cand = rec["key"]
                cand_tokens = cand.split()
                overlap = len(set(tokens) & set(cand_tokens))
                ratio = overlap / max(len(tokens), len(cand_tokens))
                if ratio >= min_overlap:
                    scored.append((ratio, cand))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:max_candidates]]

    def neighbors(self, key: str, *, max_neighbors: int = 20) -> list[str]:
        """
        Return up to `max_neighbors` entity keys most strongly co-occurring
        with `key`, ranked by `CO_OCCURS.weight` (descending).
        """

        with self.driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (e:Entity {key: $key, experiment: $experiment})
                      -[r:CO_OCCURS]->(n:Entity {experiment: $experiment})
                RETURN n.key AS key
                ORDER BY r.weight DESC
                LIMIT $max_neighbors
                """,
                key=key,
                experiment=self.experiment,
                max_neighbors=max_neighbors,
            )
            return [rec["key"] for rec in records]

    def hop_distances(
        self, seed_keys: list[str], chunk_ids: list[str], *, max_hops: int = 3
    ) -> dict[str, int | None]:
        """
        For each chunk in `chunk_ids`, return the minimum number of
        CO_OCCURS hops from any entity in `seed_keys` to any entity that
        MENTIONS that chunk (0 if a seed entity itself mentions the chunk;
        None if no path of length <= max_hops exists). Used by
        `Neo4jGraphReranker` to turn graph proximity into a re-ranking
        signal without adding new candidate documents.

        Single batched round-trip (UNWIND over chunk_ids) rather than one
        query per chunk - `rerank()` is called once per question, so this
        keeps latency to one Cypher call regardless of pool size.
        """

        if not seed_keys or not chunk_ids:
            return dict.fromkeys(chunk_ids)

        # Variable-length relationship bounds must be literals in Cypher, not
        # parameters - max_hops is an internal int (GraphRerankConfig), never
        # user input, so formatting it in is safe.
        query = f"""
            UNWIND $chunk_ids AS cid
            OPTIONAL MATCH (c:Chunk {{chunk_id: cid, experiment: $experiment}})
                            <-[:MENTIONS]-(ce:Entity {{experiment: $experiment}})
            WITH cid, collect(DISTINCT ce) AS chunk_entities
            UNWIND $seed_keys AS seed_key
            OPTIONAL MATCH (seed:Entity {{key: seed_key, experiment: $experiment}})
            WITH cid, chunk_entities, collect(DISTINCT seed) AS seeds
            UNWIND chunk_entities AS ce
            UNWIND seeds AS seed
            OPTIONAL MATCH p = shortestPath((seed)-[:CO_OCCURS*0..{int(max_hops)}]-(ce))
            WITH cid, min(length(p)) AS hops
            RETURN cid, hops
        """

        with self.driver.session(database=self.database) as session:
            records = session.run(
                query, chunk_ids=chunk_ids, seed_keys=seed_keys, experiment=self.experiment
            )
            hops_by_chunk: dict[str, int | None] = dict.fromkeys(chunk_ids)
            for rec in records:
                hops_by_chunk[rec["cid"]] = rec["hops"]
        return hops_by_chunk

    def chunks_for(self, key: str) -> list[str]:
        with self.driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (e:Entity {key: $key, experiment: $experiment})-[:MENTIONS]->(c:Chunk)
                RETURN c.chunk_id AS chunk_id
                """,
                key=key,
                experiment=self.experiment,
            )
            return [rec["chunk_id"] for rec in records]


def _chunked(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]
