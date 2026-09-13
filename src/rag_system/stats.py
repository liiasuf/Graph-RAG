from __future__ import annotations

import math
import random


def bootstrap_ci(
    values: list[float],
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Return (mean, lower, upper) bootstrap confidence interval.
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - ci) / 2
    lower = means[int(alpha * n_bootstrap)]
    upper = means[int((1 - alpha) * n_bootstrap)]
    return sum(values) / n, lower, upper


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    Paired t-test: H0 = mean(b - a) == 0.
    Returns (t_statistic, p_value_approx).
    Uses a simple two-tailed approximation via t-distribution.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 1.0

    diffs = [bv - av for av, bv in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d == 0:
        return 0.0, 1.0

    se = math.sqrt(var_d / n)
    t = mean_d / se

    # Approximate p-value using normal distribution for large n,
    # or a rough t-table lookup for small n.
    p = _approx_pvalue(t, df=n - 1)
    return t, p


def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size between two paired samples."""
    if len(a) < 2:
        return 0.0
    diffs = [bv - av for av, bv in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    return mean_d / math.sqrt(var_d) if var_d > 0 else 0.0


def _approx_pvalue(t: float, df: int) -> float:
    """
    Rough two-tailed p-value approximation.
    Uses normal approximation for df >= 30, otherwise a conservative bound.
    """
    abs_t = abs(t)
    if df >= 30:
        # Normal approximation
        z = abs_t
        p = 2 * (1 - _norm_cdf(z))
    else:
        # Conservative: compare against t-distribution critical values
        # (rough lookup table for two-tailed test)
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            10: 2.228, 15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045,
        }
        df_key = min(table.keys(), key=lambda k: abs(k - df))
        crit = table[df_key]
        if abs_t >= crit:
            p = 0.04  # significant at ~0.05
        else:
            p = 0.5   # not significant
    return max(0.0, min(1.0, p))


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if z < 0:
        return 1 - _norm_cdf(-z)
    t = 1 / (1 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_per_example_stats(
    baseline_jsonl: list[dict],
    graphrag_jsonl: list[dict],
    metric: str = "f1",
) -> dict:
    """
    Given two lists of per-example dicts (matched by 'id'), compute:
    - bootstrap CI for each system
    - paired t-test
    - Cohen's d
    - win/loss/tie counts
    """
    base_by_id = {d["id"]: d for d in baseline_jsonl}
    graph_by_id = {d["id"]: d for d in graphrag_jsonl}
    common_ids = sorted(set(base_by_id) & set(graph_by_id))

    base_vals = [base_by_id[i].get(metric, 0.0) for i in common_ids]
    graph_vals = [graph_by_id[i].get(metric, 0.0) for i in common_ids]

    _, b_lo, b_hi = bootstrap_ci(base_vals)
    _, g_lo, g_hi = bootstrap_ci(graph_vals)
    t_stat, p_val = paired_ttest(base_vals, graph_vals)
    d = cohen_d(base_vals, graph_vals)

    wins = sum(1 for b, g in zip(base_vals, graph_vals) if g > b)
    losses = sum(1 for b, g in zip(base_vals, graph_vals) if g < b)
    ties = len(common_ids) - wins - losses

    return {
        "n": len(common_ids),
        "metric": metric,
        "baseline": {"mean": sum(base_vals) / len(base_vals) if base_vals else 0, "ci95_lo": b_lo, "ci95_hi": b_hi},
        "graphrag": {"mean": sum(graph_vals) / len(graph_vals) if graph_vals else 0, "ci95_lo": g_lo, "ci95_hi": g_hi},
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "significance": significance_label(p_val),
        "cohen_d": round(d, 4),
        "win_loss_tie": {"graphrag_wins": wins, "graphrag_losses": losses, "ties": ties},
    }
