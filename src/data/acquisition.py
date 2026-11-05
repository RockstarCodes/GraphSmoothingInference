"""Download the Pokec-z files from the official FairGNN repository.

Safe to rerun: an existing file is never silently replaced. If it exists and
its SHA-256 matches the recorded checksum it is kept; if it differs the script
stops and asks the user to inspect it (use --force to replace). Downloads go to
``<name>.part`` and resume with an HTTP Range request after interruption.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 4, timeout: int = 60) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "afgml-capstone-acquire/1.0"})
        if start:
            req.add_header("Range", f"bytes={start}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                mode = "ab" if (start and resp.status == 206) else "wb"
                with open(part, mode) as fh:
                    shutil.copyfileobj(resp, fh)
            part.replace(dest)
            return
        except Exception as exc:  # network errors: retry, keeping the partial file
            log.warning("download attempt %d/%d for %s failed: %s", attempt, retries, url, exc)
    raise RuntimeError(f"could not download {url} after {retries} attempts; partial file kept at {part}")


def acquire(files: Dict[str, Dict[str, str]], base_url: str, raw_dir: Path,
            provenance_path: Path, force: bool = False) -> Dict[str, dict]:
    """files: {filename: {"sha256": expected or ''}}. Returns provenance records."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, meta in files.items():
        dest = raw_dir / name
        url = f"{base_url.rstrip('/')}/{name}"
        expected = (meta or {}).get("sha256", "")
        status = "existing"
        if dest.exists():
            got = sha256sum(dest)
            if expected and got != expected and not force:
                raise RuntimeError(f"{dest} exists but its checksum {got} differs from the recorded {expected}. "
                                   "Inspect it, then rerun with --force to replace it.")
            if force and expected and got != expected:
                dest.unlink()
        if not dest.exists():
            log.info("downloading %s", url)
            _download(url, dest)
            status = "downloaded"
        got = sha256sum(dest)
        if expected and got != expected:
            raise RuntimeError(f"checksum mismatch for {dest}: got {got}, expected {expected}")
        records[name] = {"url": url, "sha256": got, "bytes": dest.stat().st_size, "status": status,
                         "checksum_verified": bool(expected),
                         "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    old.update(records)
    provenance_path.write_text(json.dumps(old, indent=2))
    return records
