# Research Question Analysis

GraphRAG vs RAG - structured comparison for paper.

Generated automatically from `experiments/model_sweep/*__qwen3.8/*/metrics.json` and `per_example.jsonl` (RQ1-RQ5), RQ6 averages all 3 chat models.

---

## RQ1 - Retrieval Quality

> Does GraphRAG improve retrieval over RAG?
> Metrics: Recall@k, Precision@k, MRR, NDCG@k
> (methodology: LinkedIn GraphRAG, Barry et al. 2404.17723)

| Dataset | Type | System | Recall@k | Precision@k | MRR | NDCG@k |
|---------|------|--------|----------|-------------|-----|--------|
| HotpotQA | multi-hop | RAG | 0.8080 | 0.2025 | 0.8869 | 0.7692 |
| HotpotQA | multi-hop | GraphRAG | 0.8270 | 0.2070 | 0.9009 | 0.7871 |
| | | **Δ (Graph−Base)** | **+0.0190 ↑** | **+0.0045 ↑** | **+0.0140 ↑** | **+0.0178 ↑** |

| MuSiQue | multi-hop | RAG | 0.5620 | 0.1398 | 0.7482 | 0.5552 |
| MuSiQue | multi-hop | GraphRAG | 0.5750 | 0.1427 | 0.7334 | 0.5549 |
| | | **Δ (Graph−Base)** | **+0.0130 ↑** | **+0.0030 ↑** | **-0.0149 ↓** | **-0.0003 =** |

| 2WikiMultiHop | multi-hop | RAG | 0.6815 | 0.2080 | 0.9462 | 0.7160 |
| 2WikiMultiHop | multi-hop | GraphRAG | 0.6860 | 0.2135 | 0.9487 | 0.7249 |
| | | **Δ (Graph−Base)** | **+0.0045 ↑** | **+0.0055 ↑** | **+0.0026 ↑** | **+0.0090 ↑** |

| NQ | single-hop | RAG | 0.5600 | 0.0762 | 0.2818 | 0.3497 |
| NQ | single-hop | GraphRAG | 0.9720 | 0.1400 | 0.5600 | 0.6622 |
| | | **Δ (Graph−Base)** | **+0.4120 ↑** | **+0.0638 ↑** | **+0.2781 ↑** | **+0.3126 ↑** |

| TriviaQA | single-hop | RAG | 0.7128 | 0.6100 | 0.7541 | 0.6947 |
| TriviaQA | single-hop | GraphRAG | 0.7101 | 0.6235 | 0.7412 | 0.6864 |
| | | **Δ (Graph−Base)** | **-0.0027 ↓** | **+0.0135 ↑** | **-0.0129 ↓** | **-0.0083 ↓** |

---

## RQ2 - Answer Quality

> Does better retrieval translate to better answers?
> Metrics: EM, F1, ROUGE-L
> (methodology: GCR, 2410.13080, Table 1; LinkedIn GraphRAG, 2404.17723)

| Dataset | Type | System | EM | F1 | ROUGE-L |
|---------|------|--------|----|----|---------|
| HotpotQA | multi-hop | RAG | 0.4920 | 0.6168 | 0.6167 |
| HotpotQA | multi-hop | GraphRAG | 0.5240 | 0.6486 | 0.6482 |
| | | **Δ** | **+0.0320 ↑** | **+0.0317 ↑** | **+0.0314 ↑** |

| MuSiQue | multi-hop | RAG | 0.1820 | 0.2251 | 0.2247 |
| MuSiQue | multi-hop | GraphRAG | 0.1740 | 0.2252 | 0.2248 |
| | | **Δ** | **-0.0080 ↓** | **+0.0001 =** | **+0.0001 =** |

| 2WikiMultiHop | multi-hop | RAG | 0.2840 | 0.3231 | 0.3231 |
| 2WikiMultiHop | multi-hop | GraphRAG | 0.2980 | 0.3419 | 0.3419 |
| | | **Δ** | **+0.0140 ↑** | **+0.0188 ↑** | **+0.0188 ↑** |

| NQ | single-hop | RAG | 0.4100 | 0.4952 | 0.4948 |
| NQ | single-hop | GraphRAG | 0.5760 | 0.7138 | 0.7117 |
| | | **Δ** | **+0.1660 ↑** | **+0.2186 ↑** | **+0.2168 ↑** |

| TriviaQA | single-hop | RAG | 0.5360 | 0.6057 | 0.6047 |
| TriviaQA | single-hop | GraphRAG | 0.5340 | 0.6035 | 0.6025 |
| | | **Δ** | **-0.0020 ↓** | **-0.0022 ↓** | **-0.0022 ↓** |

---

## RQ3 - Hallucination Rate

> Does GraphRAG reduce hallucination?
> Metric: hallucination_rate (token-grounding proxy; 1.0 = hallucinated)
> (methodology: GCR 2410.13080 Table 5; Barry et al. 2025)

| Dataset | Type | RAG | GraphRAG | Δ |
|---------|------|------------|----------|---|
| HotpotQA | multi-hop | 0.2960 | 0.2740 | -0.0220 ↓ |
| MuSiQue | multi-hop | 0.6640 | 0.6340 | -0.0300 ↓ |
| 2WikiMultiHop | multi-hop | 0.6420 | 0.6300 | -0.0120 ↓ |
| NQ | single-hop | 0.3860 | 0.0660 | -0.3200 ↓ |
| TriviaQA | single-hop | 0.2600 | 0.2720 | +0.0120 ↑ |

**Note:** Lower is better for hallucination_rate.

---

## RQ4 - Multi-hop vs Single-hop Breakdown

> Does GraphRAG help more on multi-hop questions?
> Aggregated over dataset type (multi-hop: HotpotQA, MuSiQue, 2Wiki; single-hop: NQ, TriviaQA)

### Multi-hop datasets

| Dataset | EM Δ | F1 Δ | Recall Δ | NDCG Δ | Halluc Δ |
|---------|------|------|----------|--------|----------|
| HotpotQA | +0.0320 ↑ | +0.0317 ↑ | +0.0190 ↑ | +0.0178 ↑ | +0.0220 ↑ |
| MuSiQue | -0.0080 ↓ | +0.0001 = | +0.0130 ↑ | -0.0003 = | +0.0300 ↑ |
| 2WikiMultiHop | +0.0140 ↑ | +0.0188 ↑ | +0.0045 ↑ | +0.0090 ↑ | +0.0120 ↑ |

### Single-hop datasets

| Dataset | EM Δ | F1 Δ | Recall Δ | NDCG Δ | Halluc Δ |
|---------|------|------|----------|--------|----------|
| NQ | +0.1660 ↑ | +0.2186 ↑ | +0.4120 ↑ | +0.3126 ↑ | +0.3200 ↑ |
| TriviaQA | -0.0020 ↓ | -0.0022 ↓ | -0.0027 ↓ | -0.0083 ↓ | -0.0120 ↓ |

---

## RQ5 - Computational Overhead

> What is the latency cost of adding the knowledge graph?
> (methodology: GCR 2410.13080, Table 2)

| Dataset | System | Index (s) | Graph build (s) | Query avg (s) | Tokens/req |
|---------|--------|-----------|-----------------|---------------|------------|
| HotpotQA | RAG | 60.7637 | - | 5.9842 | 1757.5880 |
| HotpotQA | GraphRAG | 393.8812 | 286.0179 | 4.6229 | 1705.1200 |

| MuSiQue | RAG | 115.4169 | - | 7.9244 | 2172.3320 |
| MuSiQue | GraphRAG | 94.6635 | 54.3194 | 7.9538 | 2177.0800 |

| 2WikiMultiHop | RAG | 297.6397 | - | 3.8256 | 1836.5360 |
| 2WikiMultiHop | GraphRAG | 239.7110 | 254.8112 | 3.8150 | 1879.1000 |

| NQ | RAG | 2.5994 | - | 1.3950 | 2294.9520 |
| NQ | GraphRAG | 3.9248 | 1.9724 | 1.0155 | 2394.2400 |

| TriviaQA | RAG | 91.1200 | - | 3.8630 | 3075.5900 |
| TriviaQA | GraphRAG | 90.7263 | 216.2787 | 2.7805 | 3090.0280 |

**Note:** graph build (Neo4j entity/edge write, one-time offline cost) ranges 1.9s (NQ, small entity count) to ~287s (HotpotQA); it does not need to repeat per query. Per-query time is dominated by LLM generation, not graph traversal - GraphRAG's query time is comparable to or faster than plain RAG's in this table.

---

## RQ6 - Neo4j Retrieval Modes, Averaged Across All 5 Datasets

> vector vs graph-only vs overlay vs graph-as-reranker, x 3 chat models -
> which Neo4j-backed strategy wins on quality/latency? n=500/dataset, uncapped
> corpus. Per-dataset breakdown: `experiments/model_sweep/model_sweep_report.md`.

| System | Model | EM | F1 | Recall@k | NDCG@k | MRR | Halluc↓ | Query time avg (s) |
|--------|-------|----|----|----------|--------|-----|---------|--------------------|
| rag_baseline | qwen3.8 | 0.3808 | 0.4532 | 0.6649 | 0.6170 | 0.7234 | 0.4496 | 4.5984 |
| rag_baseline | gemma-4-26B-A4B-it | 0.3268 | 0.3999 | 0.6649 | 0.6170 | 0.7234 | 0.4836 | 0.2138 |
| rag_baseline | qwen3-vl-30b | 0.3188 | 0.4140 | 0.6649 | 0.6170 | 0.7234 | 0.3064 | 0.2348 |
| graph_only | qwen3.8 | 0.2176 | 0.2640 | 0.3784 | 0.3374 | 0.3977 | 0.6316 | 4.9691 |
| graph_only | gemma-4-26B-A4B-it | 0.1840 | 0.2325 | 0.4081 | 0.3623 | 0.4289 | 0.6664 | 0.2478 |
| graph_only | qwen3-vl-30b | 0.1956 | 0.2542 | 0.3620 | 0.3159 | 0.3690 | 0.5728 | 0.2594 |
| graph_overlay | qwen3.8 | 0.4176 | 0.5016 | 0.7525 | 0.6847 | 0.7850 | 0.3808 | 4.2676 |
| graph_overlay | gemma-4-26B-A4B-it | 0.3600 | 0.4509 | 0.7525 | 0.6847 | 0.7850 | 0.4124 | 0.2548 |
| graph_overlay | qwen3-vl-30b | 0.3508 | 0.4570 | 0.7525 | 0.6847 | 0.7850 | 0.2384 | 0.3022 |
| graph_rerank | qwen3.8 | 0.4212 | 0.5066 | 0.7540 | 0.6831 | 0.7768 | 0.3752 | 4.0375 |
| graph_rerank | gemma-4-26B-A4B-it | 0.3624 | 0.4540 | 0.7542 | 0.6837 | 0.7778 | 0.4016 | 0.5829 |
| graph_rerank | qwen3-vl-30b | 0.3624 | 0.4692 | 0.7542 | 0.6833 | 0.7771 | 0.2376 | 0.6516 |

**Note:** averages over HotpotQA/MuSiQue/2WikiMultiHop/NQ/TriviaQA (n=500 each, real e2e runs against Neo4j 5). `adaptive` (router + subquestion decomposition) is excluded from this sweep by design and evaluated separately (MVP-scale, `configs/e2e_hotpotqa_adaptive_smoke.yaml`).

---

## Win / Loss / Tie Analysis (EM)

Counts per dataset: how often GraphRAG is better/worse/equal vs RAG.

| Dataset | Type | Wins (Graph) | Losses (Graph) | Ties | Win% |
|---------|------|-------------|----------------|------|------|
| HotpotQA | multi-hop | 28 | 12 | 460 | 5.6% |
| MuSiQue | multi-hop | 12 | 16 | 472 | 2.4% |
| 2WikiMultiHop | multi-hop | 12 | 5 | 483 | 2.4% |
| NQ | single-hop | 96 | 13 | 391 | 19.2% |
| TriviaQA | single-hop | 8 | 9 | 483 | 1.6% |
