"""Behavior-matching gate for confirmatory neural claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class MatchConfig:
    choice_epsilon: float = 0.05
    rt_nll_floor: float = 2.03  # include if rt_nll <= floor (lower is better)
    choice_primary: bool = True


@dataclass
class ModelMatchRow:
    model: str
    choice_nll: float
    rt_nll: float
    choice_delta_from_best: float
    in_choice_epsilon_ball: bool
    passes_rt_floor: bool
    matched: bool
    exclude_reason: str


def select_behavior_matched(
    metrics: pd.DataFrame,
    *,
    condition: str = "history_only",
    cfg: MatchConfig | None = None,
) -> dict:
    """Apply choice-ε ball + RT floor; choice is primary.

    `metrics` needs columns: model, condition, choice_nll, rt_nll
    (held-out or val — caller chooses; document which).
    """
    cfg = cfg or MatchConfig()
    d = metrics.loc[metrics["condition"] == condition].copy()
    if d.empty:
        raise ValueError(f"No rows for condition={condition}")
    # One row per model (if duplicates, keep best choice_nll)
    d = d.sort_values("choice_nll").groupby("model", as_index=False).first()
    best = float(d["choice_nll"].min())
    best_model = str(d.loc[d["choice_nll"].idxmin(), "model"])

    rows: list[ModelMatchRow] = []
    for _, r in d.iterrows():
        choice = float(r["choice_nll"])
        rt = float(r["rt_nll"])
        delta = choice - best
        in_eps = delta <= cfg.choice_epsilon + 1e-12
        rt_ok = rt <= cfg.rt_nll_floor + 1e-12
        reasons = []
        if not in_eps:
            reasons.append(f"choice_nll_delta={delta:.4f}>{cfg.choice_epsilon}")
        if not rt_ok:
            reasons.append(f"rt_nll={rt:.4f}>{cfg.rt_nll_floor}")
        matched = in_eps and rt_ok
        # Enforce choice-primary: cannot match on RT alone
        assert not (matched and not in_eps)
        rows.append(
            ModelMatchRow(
                model=str(r["model"]),
                choice_nll=choice,
                rt_nll=rt,
                choice_delta_from_best=delta,
                in_choice_epsilon_ball=in_eps,
                passes_rt_floor=rt_ok,
                matched=matched,
                exclude_reason="" if matched else ";".join(reasons),
            )
        )

    matched_models = [r.model for r in rows if r.matched]
    excluded = [r.model for r in rows if not r.matched]
    return {
        "condition": condition,
        "choice_primary": cfg.choice_primary,
        "choice_epsilon": cfg.choice_epsilon,
        "rt_nll_floor": cfg.rt_nll_floor,
        "best_model": best_model,
        "best_choice_nll": best,
        "matched_models": matched_models,
        "excluded_models": excluded,
        "rows": [asdict(r) for r in rows],
        "assert_choice_primary": True,
    }


def filter_ve_to_matched(ve_df: pd.DataFrame, matched_models: list[str]) -> pd.DataFrame:
    """Confirmatory table: only matched models."""
    out = ve_df.loc[ve_df["model"].isin(matched_models)].copy()
    out["confirmatory"] = True
    return out
