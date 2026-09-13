from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LatencySums:
    n: int = 0
    index_time_s: float = 0.0
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0
    total_query_time_s: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    tokens_with_usage: int = 0
    # per-stage breakdown (e.g. "router", "retrieval", "subquestion") reported
    # by retrievers via `last_trace["stage_times"]`; empty for modes that
    # don't report stages.
    stage_time_sums: dict = field(default_factory=dict)
    stage_counts: dict = field(default_factory=dict)

    def add_stage(self, name: str, seconds: float) -> None:
        self.stage_time_sums[name] = self.stage_time_sums.get(name, 0.0) + seconds
        self.stage_counts[name] = self.stage_counts.get(name, 0) + 1

    def add_request(
        self,
        *,
        retrieval_time_s,
        generation_time_s,
        total_query_time_s,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    ):
        self.n += 1
        self.retrieval_time_s += retrieval_time_s
        self.generation_time_s += generation_time_s
        self.total_query_time_s += total_query_time_s

        if prompt_tokens is not None and completion_tokens is not None and total_tokens is not None:
            self.tokens_prompt += int(prompt_tokens)
            self.tokens_completion += int(completion_tokens)
            self.tokens_total += int(total_tokens)
            self.tokens_with_usage += 1

    def mean(self):
        """
        Return a dictionary of averaged latency metrics
        """

        if self.n == 0:
            return {
                "n": 0,
                "index_time_s": self.index_time_s,
                "retrieval_time_s": 0.0,
                "generation_time_s": 0.0,
                "total_query_time_s": 0.0,
                "tokens_per_request": None,
                "tokens_prompt_per_request": None,
                "tokens_completion_per_request": None,
                "usage_coverage": 0.0,
                "stage_times": {},
            }

        tokens_per_req = None
        tokens_prompt_per_req = None
        tokens_completion_per_req = None
        usage_cov = self.tokens_with_usage / self.n

        if self.tokens_with_usage > 0:
            tokens_per_req = self.tokens_total / self.tokens_with_usage
            tokens_prompt_per_req = self.tokens_prompt / self.tokens_with_usage
            tokens_completion_per_req = self.tokens_completion / self.tokens_with_usage

        return {
            "n": self.n,
            "index_time_s": self.index_time_s,
            "retrieval_time_s": self.retrieval_time_s / self.n,
            "generation_time_s": self.generation_time_s / self.n,
            "total_query_time_s": self.total_query_time_s / self.n,
            "tokens_per_request": tokens_per_req,
            "tokens_prompt_per_request": tokens_prompt_per_req,
            "tokens_completion_per_request": tokens_completion_per_req,
            "usage_coverage": usage_cov,
            "stage_times": {
                name: round(total / self.stage_counts[name], 5)
                for name, total in self.stage_time_sums.items()
            },
        }
