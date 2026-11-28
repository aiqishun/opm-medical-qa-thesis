"""Write-once run artifacts and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_once_json(path: Path, payload: Any) -> None:
    """Create ``path``; refuse to overwrite an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


# Directories whose untracked files change what a run executes. Output
# directories such as experiments/runs/ are deliberately not listed.
CODE_DIRS = ("src", "scripts", "prompts", "schemas", "configs")


def git_info(root: Path) -> dict[str, Any]:
    """Commit, exact tag (if any) and cleanliness of the working tree.

    The tree is dirty if a tracked file has uncommitted changes, or if an
    untracked (and not ignored) file exists under one of ``CODE_DIRS``.
    """
    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)

    try:
        tracked = run("status", "--porcelain", "--untracked-files=no").stdout.split("\n")
        untracked = run("ls-files", "--others", "--exclude-standard", "--", *CODE_DIRS).stdout.split("\n")
        tag = run("describe", "--exact-match", "--tags", "HEAD", check=False)
        tracked, untracked = [x for x in tracked if x.strip()], [x for x in untracked if x.strip()]
        return {
            "commit": run("rev-parse", "HEAD").stdout.strip(),
            "short": run("rev-parse", "--short", "HEAD").stdout.strip(),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
            "tag": tag.stdout.strip() if tag.returncode == 0 else None,
            "dirty": bool(tracked or untracked),
            "modified_tracked": tracked[:50],
            "untracked_code_files": untracked[:50],
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "short": "nogit", "branch": None, "tag": None, "dirty": None,
                "modified_tracked": [], "untracked_code_files": []}


def safe_component(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
