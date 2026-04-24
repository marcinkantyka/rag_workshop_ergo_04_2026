"""
Experiment log for RAG improvement tracking.

Persists experiments as JSON so results survive kernel restarts
and can be reviewed at end of Day 2.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_PATH = Path(__file__).parent.parent / "data" / "experiment_log.json"


class ExperimentLog:
    """
    Tracks RAG experiments across the workshop.

    Usage:
        log = ExperimentLog()
        log.add(
            name="Baseline",
            config={"top_k": 5, "chunk_size": 800, "model": "haiku"},
            scores={"faithfulness": 0.72, "context_recall": 0.65},
            notes="Day 1 baseline — recursive chunking, vector retrieval",
        )
        log.summary()
        df = log.to_dataframe()
    """

    METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def __init__(self, path: str | Path = DEFAULT_LOG_PATH):
        self.path = Path(path)
        self._entries: list[dict] = self._load()

    def add(
        self,
        name: str,
        config: dict,
        scores: dict,
        notes: str = "",
    ) -> None:
        """
        Record an experiment result.

        Args:
            name: short label, e.g. "Baseline" or "Round1-Recursive-400"
            config: dict of pipeline parameters (chunk_size, top_k, retrieval_method, ...)
            scores: dict of RAGAS metric scores
            notes: free-text observation
        """
        # TODO: Build an entry dict with keys: "name", "timestamp" (ISO format,
        # minute precision), "config", "scores" (values rounded to 4 decimals),
        # and "notes".
        # Remove any existing entry with the same name, append the new one,
        # call self._save(), and print a confirmation message.
        raise NotImplementedError

    def summary(self) -> None:
        """Print a formatted table of all experiments."""
        if not self._entries:
            print("No experiments logged yet.")
            return

        header = f"{'Name':<30} {'Faithf':>8} {'AnswRel':>8} {'CtxPrec':>8} {'CtxRec':>8}  Notes"
        print(header)
        print("─" * len(header))

        for e in self._entries:
            s = e["scores"]
            row = (
                f"{e['name']:<30}"
                f" {s.get('faithfulness', 0):>8.3f}"
                f" {s.get('answer_relevancy', 0):>8.3f}"
                f" {s.get('context_precision', 0):>8.3f}"
                f" {s.get('context_recall', 0):>8.3f}"
                f"  {e.get('notes', '')[:50]}"
            )
            print(row)

    def to_dataframe(self):
        """Convert log to a pandas DataFrame for plotting."""
        import pandas as pd
        rows = []
        for e in self._entries:
            row = {"name": e["name"], "timestamp": e["timestamp"]}
            row.update(e["scores"])
            row.update({f"cfg_{k}": v for k, v in e["config"].items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def plot(self, metrics: list[str] | None = None) -> None:
        """Bar chart comparing all logged experiments across selected metrics."""
        import matplotlib.pyplot as plt
        import numpy as np

        if not self._entries:
            print("No experiments to plot.")
            return

        if metrics is None:
            metrics = self.METRICS

        df = self.to_dataframe()
        names = df["name"].tolist()
        n, m = len(names), len(metrics)
        x = np.arange(n)
        width = 0.8 / m

        fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 5))
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax.bar(x + i * width, df[metric].fillna(0), width, label=metric, alpha=0.85)

        ax.set_xticks(x + width * (m - 1) / 2)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.7, color="orange", linestyle="--", alpha=0.4, label="Target (0.7)")
        ax.set_ylabel("Score")
        ax.set_title("Experiment Comparison — RAGAS Metrics")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def delta_table(self, baseline_name: str) -> None:
        """Show improvement/regression vs. a named baseline experiment."""
        df = self.to_dataframe()
        if baseline_name not in df["name"].values:
            print(f"Baseline '{baseline_name}' not found in log.")
            return

        baseline = df[df["name"] == baseline_name].iloc[0]
        metrics = [m for m in self.METRICS if m in df.columns]

        print(f"Delta vs baseline: '{baseline_name}'")
        print(f"{'Name':<30}", end="")
        for m in metrics:
            print(f" {m[:8]:>8}", end="")
        print()
        print("─" * (30 + 9 * len(metrics)))

        for _, row in df.iterrows():
            if row["name"] == baseline_name:
                continue
            print(f"{row['name']:<30}", end="")
            for m in metrics:
                delta = row.get(m, 0) - baseline.get(m, 0)
                sign = "+" if delta > 0 else ""
                print(f" {sign}{delta:>7.3f}", end="")
            print()

    def clear(self) -> None:
        """Remove all entries (use with caution)."""
        self._entries = []
        self._save()
        print("[ExperimentLog] Cleared.")

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ExperimentLog(entries={len(self._entries)}, path={self.path})"

    def _load(self) -> list[dict]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._entries, f, indent=2, ensure_ascii=False)
