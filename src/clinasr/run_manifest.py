"""Run isolation: run IDs, per-run result directories, and manifests.

Policy: every run gets its own directory under the results root, named from
a timestamp plus the short git commit hash of the code that produced it (or
"nogit" if no repository/commit is available, e.g. a Kaggle sandbox without
.git). A run never overwrites another run's directory; collisions raise
FileExistsError, since this project's integrity rules require every result
to be traceable to exactly one run's manifest.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_commit(repo_dir: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def make_run_id(now: Optional[datetime] = None, git_commit: Optional[str] = None) -> str:
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit if git_commit else "nogit"
    return f"{stamp}_{commit}"


@dataclass
class RunHandle:
    run_id: str
    run_dir: Path
    manifest_path: Path
    log_path: Path

    def write_manifest(self, extra: Dict[str, Any]) -> None:
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            **extra,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")

    def get_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"clinasr.run.{self.run_id}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger


def new_run(results_root: Path, git_commit: Optional[str] = None) -> RunHandle:
    """Create a fresh, never-overwritten run directory under results_root."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    commit = git_commit if git_commit is not None else get_git_commit()
    run_id = make_run_id(git_commit=commit)
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # raises FileExistsError on any collision
    return RunHandle(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        log_path=run_dir / "run.log",
    )
