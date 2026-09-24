"""Central paths + run config. Works identically locally and on Kaggle."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _detect_root() -> Path:
    """Repo root locally; Kaggle working dir when running in a notebook."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path(__file__).resolve().parents[1]


def _detect_data() -> Path:
    """Competition data dir. Kaggle mounts read-only input; locally use data/raw."""
    env = os.environ.get("AML_DATA_DIR")
    if env:
        return Path(env)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        subs = sorted(p for p in kaggle_input.iterdir() if p.is_dir())
        if subs:
            return subs[0]
    return _detect_root() / "data" / "raw"


ROOT = _detect_root()
DATA_RAW = _detect_data()
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"

ON_KAGGLE = Path("/kaggle/input").exists()

for _p in (DATA_INTERIM, DATA_PROC, MODELS, SUBMISSIONS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    """One experiment. Keep every run reproducible by bumping `name`."""

    name: str = "baseline"
    seed: int = 42
    n_folds: int = 5
    target: str = "target"
    id_col: str = "id"
    group_col: str | None = None      # set for GroupKFold (e.g. product_id)
    task: str = "regression"           # regression | binary | multiclass
    metric: str = "rmse"
    params: dict = field(default_factory=dict)

    @property
    def oof_path(self) -> Path:
        return DATA_PROC / f"oof_{self.name}.npy"

    @property
    def pred_path(self) -> Path:
        return DATA_PROC / f"pred_{self.name}.npy"

    @property
    def sub_path(self) -> Path:
        return SUBMISSIONS / f"sub_{self.name}.csv"


def describe_env() -> str:
    import sys
    lines = [
        f"python  : {sys.version.split()[0]}",
        f"root    : {ROOT}",
        f"data_raw: {DATA_RAW}  (exists={DATA_RAW.exists()})",
        f"kaggle  : {ON_KAGGLE}",
    ]
    try:
        import torch
        lines.append(f"torch   : {torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu     : {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
    except Exception:
        lines.append("torch   : not installed")
    return "\n".join(lines)
