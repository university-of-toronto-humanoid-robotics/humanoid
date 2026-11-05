"""Download the competition testbed CAD from the organizers' GitHub repo.

The file is ~100 MB (Git LFS upstream, over GitHub's 100 MB file limit), so we
fetch it instead of committing it. It lands in sim/competition/cad/, which is
git-ignored.

Usage:
  python tools/cad/fetch_testbed.py
  python tools/cad/fetch_testbed.py --any-version   # accept an updated CAD
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
URL = (
  "https://media.githubusercontent.com/media/peterso/"
  "ieee-humanoids-2026-loco-manipulation-challenge/main/CAD-Files/"
  "LocoManipulationTestbed.dae"
)
# Version measured on 2026-10-28 (sim/competition/README.md).
SHA256 = "ce0d4344d02fb183144470c5ebe7ed06b2bbf88417778f215875814ea5ab427c"
DEFAULT_OUT = REPO_ROOT / "sim" / "competition" / "cad" / "LocoManipulationTestbed.dae"


def sha256_of(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def download(url: str, dest: Path) -> None:
  part = dest.with_suffix(dest.suffix + ".part")
  with urllib.request.urlopen(url) as resp, part.open("wb") as out:
    total = int(resp.headers.get("Content-Length", 0))
    done = 0
    while chunk := resp.read(1 << 20):
      out.write(chunk)
      done += len(chunk)
      if total:
        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
  print()
  part.replace(dest)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument(
    "--any-version",
    action="store_true",
    help="Keep the file even if it differs from the version we measured.",
  )
  args = parser.parse_args()
  out: Path = args.out
  out.parent.mkdir(parents=True, exist_ok=True)

  if out.exists() and sha256_of(out) == SHA256:
    print(f"Already up to date: {out}")
    return 0

  print(f"Downloading {URL}")
  download(URL, out)
  digest = sha256_of(out)
  if digest == SHA256:
    print(f"OK: {out}")
    return 0
  if args.any_version:
    print(f"Saved a NEW CAD version (sha256 {digest}). Re-check the course dimensions.")
    return 0
  print(
    f"The CAD changed upstream (sha256 {digest}, expected {SHA256}).\n"
    "Rerun with --any-version to use it, then rebuild and re-check the course.",
    file=sys.stderr,
  )
  return 1


if __name__ == "__main__":
  sys.exit(main())
