"""Local Kaggle CLI wrapper for ClinASR. No heavy local ML dependencies.

    python scripts/kaggle_workflow.py init --username YOUR_USERNAME
    python scripts/kaggle_workflow.py push --confirm --budget-minutes 30
    python scripts/kaggle_workflow.py status
    python scripts/kaggle_workflow.py fetch

`push` is explicitly opt-in: it refuses to run without --confirm and a
--budget-minutes estimate, and it logs every submission (requested budget,
kernel id, timestamp, git commit) to results/gpu_budget_log.json so GPU
usage across runs stays visible. This script does not know this account's
real Kaggle GPU quota -- it is a local paper trail, not quota enforcement.

`fetch` downloads into a fresh, timestamped subdirectory under results/
rather than overwriting the previous download, so earlier run outputs are
never silently lost.

This installed kaggle CLI version (verified: 1.7.4.5) has no `logs`
subcommand (valid kernels subcommands: list, files, init, push, pull,
output, status) -- the execution console log is only viewable on the
Kaggle website at the kernel's URL, printed by `push` on submission.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle"
META = KAGGLE_DIR / "kernel-metadata.json"
SCRIPT = KAGGLE_DIR / "run_clinasr.py"
RESULTS = ROOT / "results"
BUDGET_LOG = RESULTS / "gpu_budget_log.json"
KAGGLE_CREDENTIALS_FILE = Path.home() / ".kaggle" / "kaggle.json"


def _kaggle_subprocess_env() -> dict:
    """Return an env for the kaggle CLI subprocess with KAGGLE_API_TOKEN set.

    The installed kaggle CLI (verified: 1.7.4.5) sends classic HTTP Basic
    auth (username+key from kaggle.json) for every endpoint EXCEPT when the
    KAGGLE_API_TOKEN env var is set, in which case it switches to Bearer
    auth (kagglesdk/kaggle_http_client.py's `_try_fill_auth`). As of
    2026-10-29, Kaggle's Kernels API (push/list/status/logs/output) requires
    Bearer auth for newer "KGAT_"-prefixed tokens and 401s on Basic auth,
    while the Datasets API still accepts Basic auth either way -- so this
    bug is invisible unless you specifically exercise Kernels. Deriving the
    token from kaggle.json here means the user never has to remember to
    `export KAGGLE_API_TOKEN=...` themselves. See docs/ISSUES.md.
    """
    env = dict(os.environ)
    # The kaggle CLI itself prints non-ASCII characters (e.g. a checkmark on
    # success) that crash with 'charmap' codec errors on Windows' default
    # console codepage; force UTF-8 stdout/stderr for the subprocess so a
    # cosmetic print doesn't make a successful command look like a failure.
    env["PYTHONIOENCODING"] = "utf-8"
    if env.get("KAGGLE_API_TOKEN"):
        return env
    if not KAGGLE_CREDENTIALS_FILE.exists():
        return env
    try:
        creds = json.loads(KAGGLE_CREDENTIALS_FILE.read_text(encoding="utf-8"))
        key = creds.get("key")
        if key:
            env["KAGGLE_API_TOKEN"] = key
    except (json.JSONDecodeError, OSError):
        pass
    return env


def load_meta():
    if not META.exists():
        raise RuntimeError(
            "Kaggle metadata is not initialized. Run: "
            "python scripts/kaggle_workflow.py init --username YOUR_KAGGLE_USERNAME"
        )
    data = json.loads(META.read_text(encoding="utf-8"))
    kernel_id = data.get("id", "")
    if not kernel_id or "/" not in kernel_id:
        raise RuntimeError("Invalid Kaggle metadata id. Expected username/kernel-slug.")
    return data


def kaggle(args):
    executable = shutil.which("kaggle")
    if not executable:
        raise RuntimeError(
            "The 'kaggle' CLI was not found on PATH. Install via "
            "python -m pip install --upgrade kaggle; reopen VS Code's terminal "
            "and authenticate with 'kaggle auth login'."
        )
    cmd = [executable, "kernels", *args]
    print("Executing:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=_kaggle_subprocess_env())


def _get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _append_budget_log(kernel_id: str, budget_minutes: float) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    entries = []
    if BUDGET_LOG.exists():
        entries = json.loads(BUDGET_LOG.read_text(encoding="utf-8"))
    entries.append(
        {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "kernel_id": kernel_id,
            "requested_budget_minutes": budget_minutes,
            "git_commit": _get_git_commit(),
        }
    )
    BUDGET_LOG.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    total = sum(e["requested_budget_minutes"] for e in entries)
    print(f"Logged submission to {BUDGET_LOG}. Cumulative requested budget so far: {total:.1f} min.")


def main():
    parser = argparse.ArgumentParser(description="ClinASR Kaggle execution helper")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create private Kaggle kernel metadata")
    init.add_argument("--username", required=True, help="Your Kaggle username, not your email")
    init.add_argument("--slug", default="clinasr-medical-asr")
    init.add_argument("--force", action="store_true", help="Overwrite existing metadata")

    push = sub.add_parser("push", help="Upload and start the GPU job (opt-in)")
    push.add_argument(
        "--confirm",
        action="store_true",
        help="Required. Explicit acknowledgment that this submits a real GPU job.",
    )
    push.add_argument(
        "--budget-minutes",
        type=float,
        default=None,
        help="Required. Your own estimate of this run's GPU minutes, logged locally for tracking "
        "(this script cannot see your actual Kaggle quota).",
    )

    sub.add_parser("status", help="Check latest GPU job status")
    sub.add_parser("fetch", help="Download results into a new timestamped results/ subdirectory")
    args = parser.parse_args()

    if args.command == "init":
        username, slug = args.username.strip(), args.slug.strip()
        if not username or not slug or "/" in username or "/" in slug or " " in slug:
            parser.error("Invalid username or slug.")
        if META.exists() and not args.force:
            parser.error("Metadata already exists. Add --force only if you want to overwrite it.")
        metadata = {
            "id": f"{username}/{slug}",
            "title": "ClinASR Medical ASR",
            "code_file": "run_clinasr.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Created {META}\nKernel ID: {metadata['id']}\nPrivate GPU job: yes")
        print("Next: ask Claude Code to generate kaggle/run_clinasr.py using prompts/01-build.md")
        return 0

    meta = load_meta()
    kernel_id = meta["id"]
    if args.command == "push":
        if not SCRIPT.exists():
            raise RuntimeError(
                f"{SCRIPT} does not exist yet. Open prompts/01-build.md and "
                "ask Claude Code to implement the first Kaggle smoke test before pushing."
            )
        if not args.confirm:
            raise RuntimeError(
                "Refusing to push without --confirm. This submits a real GPU job to Kaggle; "
                "pass --confirm --budget-minutes N once you intend that."
            )
        if args.budget_minutes is None or args.budget_minutes <= 0:
            raise RuntimeError("Pass --budget-minutes N (your own estimate) along with --confirm.")
        _append_budget_log(kernel_id, args.budget_minutes)
        # This kaggle CLI version has no --accelerator flag; GPU/machine shape
        # come from kernel-metadata.json's enable_gpu/machine_shape (set by
        # `init`). -t/--timeout is in seconds and actually enforces the
        # requested budget server-side, rather than only logging it locally.
        timeout_seconds = int(args.budget_minutes * 60)
        return kaggle(["push", "-p", str(KAGGLE_DIR), "-t", str(timeout_seconds)])
    if args.command == "status":
        return kaggle(["status", kernel_id])
    if args.command == "fetch":
        RESULTS.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = RESULTS / f"fetched_{timestamp}"
        suffix = 0
        while True:
            try:
                dest.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                dest = RESULTS / f"fetched_{timestamp}_{suffix}"
        return kaggle(["output", kernel_id, "-p", str(dest)])
    raise AssertionError("Unknown command")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(2)
