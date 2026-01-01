"""Shared plotting utilities and constants."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use("ggplot")

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

# Nord Theme Colors
COLORS = {
    "ideal": "#4C566A",
    "naive": "#B48EAD",
    "sdpa": "#A3BE8C",
    "kernels": [
        "#88C0D0",
        "#BF616A",
        "#D08770",
        "#EBCB8B",
        "#8FBCBB",
        "#81A1C1",
        "#5E81AC",
    ],
}


def load_csv(path: Path) -> dict[str, dict[int, float]]:
    """Load benchmark CSV into {version: {threads: time_s}}."""
    data: dict[str, dict[int, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ver, thr, time = row["version"], int(row["threads"]), float(row["time_s"])
            data.setdefault(ver, {})[thr] = time
    return data


def split_versions(data: dict) -> tuple[dict, dict]:
    """Split into (pytorch_versions, kernel_versions)."""
    pytorch = {k: v for k, v in data.items() if k.startswith("pytorch")}
    kernels = {k: v for k, v in data.items() if not k.startswith("pytorch")}
    return pytorch, kernels


def save_and_show(fig, path: Path | None, show: bool) -> None:
    """Save figure to path and/or display it."""
    plt.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200)
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)
