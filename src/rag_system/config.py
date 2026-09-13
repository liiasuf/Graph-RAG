from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """
    A class representing the dataset configuration for the end-to-end runner.
    It currently supports only the "hotpotqa" dataset, but there is an opportunity to easily extend it to support more datasets in the future.

    The 'name' field is a literal type that specifies the name of the dataset to be used.
    To add a new dataset, one would need to create a new adapter in the `rag_system/datasets/` directory and register it in the `rag_system/datasets/factory.py` file.
    """

    name: Literal["hotpotqa", "natural_questions", "musique", "wiki2multihop", "triviaqa"] = "hotpotqa"


class HotpotQAConfig(BaseModel):
    subset: Literal["distractor", "fullwiki"] = "distractor"
    train_split: str = "train"
    eval_split: str = "validation"
    max_train_examples: int | None = 20000
    max_corpus_docs: int | None = 50000
    max_corpus_chunks: int | None = 200000
    max_eval_examples: int | None = 200
    seed: int = 42


class NaturalQuestionsConfig(BaseModel):
    eval_split: str = "validation"
    max_corpus_docs: int | None = 50000
    max_eval_examples: int | None = 200


class MusiqueConfig(BaseModel):
    train_split: str = "train"
    eval_split: str = "validation"
    max_corpus_docs: int | None = 50000
    max_eval_examples: int | None = 200


class Wiki2MultiHopConfig(BaseModel):
    train_split: str = "train"
    eval_split: str = "validation"
    max_corpus_docs: int | None = 50000
    max_eval_examples: int | None = 200


class TriviaQAConfig(BaseModel):
    eval_split: str = "validation"
    max_corpus_docs: int | None = 50000
    max_eval_examples: int | None = 200


class ChunkingConfig(BaseModel):
    max_words: int = 220
    overlap_words: int = 40


class EmbeddingsConfig(BaseModel):
    # These defaults point at a local LM Studio server - a reasonable
    # zero-config fallback for a from-scratch clone, but every run in this
    # project's actual history overrides them via EMBED_BASE_URL/EMBED_MODEL/
    # EMBED_API_KEY in .env (see load_config() below and .env.example).
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "LLM_API_KEY"
    model: str = "text-embedding-nomic-embed-text-v1.5"
    batch_size: int = 64
    use_cache: bool = True
    cache_path: str | None = "cache/embeddings.json"


class IndexConfig(BaseModel):
    backend: Literal["numpy_cosine"] = "numpy_cosine"


class RetrievalConfig(BaseModel):
    top_k: int = 8
    # "vector"              - plain cosine similarity search (baseline RAG)
    # "graph_neo4j"         - entity-graph traversal (Neo4j) + vector re-ranking
    #                         (GraphRAG "overlay"); vector top-k is always included
    #                         (see GraphConfig.neo4j). Requires `docker-compose up -d`.
    # "graph_neo4j_only"    - Neo4j graph traversal WITHOUT the vector-floor guarantee:
    #                         candidates come only from the graph; if the graph yields
    #                         no candidates at all, falls back to pure vector search
    #                         (logged as `fallback_used: true` in per_example.jsonl).
    #                         Requires Neo4j.
    # "graph_neo4j_rerank"  - graph as a RE-RANKER, not a candidate source: vector
    #                         top-`graph_rerank.vector_pool_k` candidates are re-scored
    #                         by graph-hop distance from the query's seed entities (see
    #                         GraphRerankConfig). Requires Neo4j.
    # "adaptive"            - router (skip the graph entirely for low-entity questions)
    #                         + subquestion decomposition over the Neo4j graph (see
    #                         AdaptiveMethodConfig). Requires Neo4j.
    mode: Literal[
        "vector", "graph_neo4j", "graph_neo4j_only", "graph_neo4j_rerank", "adaptive",
    ] = "vector"


class Neo4jConfig(BaseModel):
    """
    Connection settings for the Neo4j-backed entity graph (MVP, stage 3).

    Matches the default credentials in `docker-compose.yml`. Override via
    the YAML config or environment variables (NEO4J_URI / NEO4J_USER /
    NEO4J_PASSWORD) - see `e2e_graphrag_neo4j.py`.
    """

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j_password_change_me"
    database: str = "neo4j"
    # wipe all :Entity/:Chunk nodes belonging to this experiment before
    # (re)building the graph, so repeated runs don't accumulate duplicates
    clear_before_build: bool = True
    # batch size for UNWIND-based bulk inserts
    batch_size: int = 1000


class GraphConfig(BaseModel):
    """
    Configuration for the GraphRAG entity graph built over corpus chunks.
    """

    # include the document title itself as a graph entity
    title_as_entity: bool = True
    # max number of consecutive capitalised words considered one entity mention
    max_entity_words: int = 5
    # minimum character length of an entity mention
    min_entity_chars: int = 3
    # number of graph hops to expand from the query's seed entities
    max_hops: int = 1
    # max neighbours kept per entity per hop (by co-occurrence weight)
    max_neighbors_per_hop: int = 20
    # cap on candidate chunks gathered from the graph before vector re-ranking
    max_candidate_chunks: int = 300

    # connection settings for the Neo4j-backed graph (used by every
    # retrieval.mode except "vector")
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)


class GraphRerankConfig(BaseModel):
    """
    Configuration for retrieval.mode == "graph_neo4j_rerank": the graph never
    adds new candidate documents, it only re-scores a vector-search pool.
    """

    # size of the vector-search candidate pool the graph re-ranks (independent
    # of retrieval.top_k, which is the number returned after re-ranking)
    vector_pool_k: int = 30
    # combined = beta * graph_signal_norm + (1 - beta) * cosine_score
    # beta=0 -> pure vector rerank (no-op); beta=1 -> pure graph-hop ranking
    beta: float = 0.5
    # graph_signal = hop_decay ** hop_distance (0 hops = seed entity directly
    # mentions the chunk -> signal 1.0; farther hops decay toward 0)
    hop_decay: float = 0.5
    # cap on hop distance considered when searching for the shortest path
    # between seed entities and a candidate chunk's entities
    max_hops: int = 3


class RouterConfig(BaseModel):
    """
    Decides, per-question, whether to touch the Neo4j graph at all
    (retrieval.mode == "adaptive"). Disabled by default so the other two
    adaptive components (subquestions) can be ablated independently.
    """

    enabled: bool = False
    method: Literal["heuristic", "llm"] = "heuristic"
    # "heuristic": if extract_entities(question) yields fewer than this many
    # mentions, skip the graph entirely (pure vector search, zero Neo4j calls).
    min_entities_for_graph: int = 1


class SubQuestionConfig(BaseModel):
    """
    Iterative subquestion decomposition for retrieval.mode == "adaptive":
    original question -> subquestion1 -> entity from graph -> subquestion2
    -> sufficiency check -> (repeat or answer).
    """

    enabled: bool = False
    max_subquestions: int = 2
    # local single-slot LLM servers make every extra call expensive - keep
    # small by default; each iteration issues 2+ extra LLM calls.
    max_iterations: int = 1
    sufficiency_check: Literal["llm_judge", "heuristic"] = "heuristic"


class AdaptiveMethodConfig(BaseModel):
    """
    retrieval.mode == "adaptive": three independently toggleable components
    for ablations (run_ablations_full.py B1_router / B2_graph_rerank /
    B3_subquestions / B4_combined):
      1. router.enabled           - skip the graph entirely for low-signal questions
      2. graph_retrieval_mode     - which of steps 1-2's Neo4j strategies is reused
                                     whenever a (sub)question IS sent to the graph;
                                     "graph_neo4j_rerank" is the graph-as-reranker
                                     signal from GraphRerankConfig, on by default
      3. subquestions.enabled     - iterative subquestion decomposition
    """

    router: RouterConfig = Field(default_factory=RouterConfig)
    subquestions: SubQuestionConfig = Field(default_factory=SubQuestionConfig)
    # reuses the retrieval strategies from steps 1-2 instead of a separate
    # graph-traversal branch (see module docstring above)
    graph_retrieval_mode: Literal["graph_neo4j", "graph_neo4j_only", "graph_neo4j_rerank"] = (
        "graph_neo4j_rerank"
    )


class LocalFilesConfig(BaseModel):
    corpus_dir: str = "data/documents"
    eval_jsonl: str | None = None
    max_corpus_docs: int | None = None


class RerankerConfig(BaseModel):
    enabled: bool = False
    alpha: float = 0.5          # weight of LLM score (0=cosine only, 1=LLM only)
    max_passage_chars: int = 500


class QueryExpansionConfig(BaseModel):
    enabled: bool = False


class SummarizationConfig(BaseModel):
    enabled: bool = False
    max_input_chars: int = 1000


class LLMConfig(BaseModel):
    # Same caveat as EmbeddingsConfig above: local LM Studio fallback,
    # overridden in practice via LLM_BASE_URL/LLM_MODEL/LLM_API_KEY in .env.
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "LLM_API_KEY"
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_s: float = 120.0
    max_context_chars: int = 12000  # truncate prompt context to avoid 400 (context length)


class RunConfig(BaseModel):
    experiment_name: str = "hotpotqa_rag_baseline"
    output_dir: str = "experiments"

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    hotpotqa: HotpotQAConfig = Field(default_factory=HotpotQAConfig)
    natural_questions: NaturalQuestionsConfig = Field(default_factory=NaturalQuestionsConfig)
    musique: MusiqueConfig = Field(default_factory=MusiqueConfig)
    wiki2multihop: Wiki2MultiHopConfig = Field(default_factory=Wiki2MultiHopConfig)
    triviaqa: TriviaQAConfig = Field(default_factory=TriviaQAConfig)
    local_files: LocalFilesConfig = Field(default_factory=LocalFilesConfig)

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    query_expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    graph_rerank: GraphRerankConfig = Field(default_factory=GraphRerankConfig)
    adaptive: AdaptiveMethodConfig = Field(default_factory=AdaptiveMethodConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(path: str | Path) -> RunConfig:
    """
    Load a YAML config file into a validated `RunConfig`.

    Environment variable overrides (take precedence over YAML values):
      LLM_BASE_URL    - base_url for LLM (and embeddings if EMBED_BASE_URL not set)
      LLM_API_KEY     - api_key for LLM (and embeddings if EMBED_API_KEY not set)
      LLM_MODEL       - chat model name
      LLM_MAX_TOKENS  - max_tokens for LLM generation
      EMBED_BASE_URL  - base_url for embeddings (overrides LLM_BASE_URL for embeds)
      EMBED_API_KEY   - api_key for embeddings (overrides LLM_API_KEY for embeds)
      EMBED_MODEL     - embedding model name
    """
    import os

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    llm_base_url  = os.environ.get("LLM_BASE_URL")
    llm_api_key   = os.environ.get("LLM_API_KEY")
    llm_model     = os.environ.get("LLM_MODEL")
    max_tokens    = os.environ.get("LLM_MAX_TOKENS")

    emb_base_url  = os.environ.get("EMBED_BASE_URL") or llm_base_url
    emb_api_key   = os.environ.get("EMBED_API_KEY")  or llm_api_key
    emb_model     = os.environ.get("EMBED_MODEL")

    if llm_base_url:
        data.setdefault("llm", {})["base_url"] = llm_base_url
    if llm_api_key:
        data.setdefault("llm", {})["api_key"] = llm_api_key
    if llm_model:
        data.setdefault("llm", {})["model"] = llm_model
    if max_tokens:
        data.setdefault("llm", {})["max_tokens"] = int(max_tokens)

    if emb_base_url:
        data.setdefault("embeddings", {})["base_url"] = emb_base_url
    if emb_api_key:
        data.setdefault("embeddings", {})["api_key"] = emb_api_key
    if emb_model:
        data.setdefault("embeddings", {})["model"] = emb_model

    return RunConfig.model_validate(data)
