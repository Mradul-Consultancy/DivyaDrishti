"""
Sync the test-count claim in PROJECT_DOCUMENTATION.md with the real suite.

The verification table quotes "N/N tests". Rather than hand-editing that
number (which drifts the moment a test is added), this script rewrites it
from what pytest actually collects.

Usage:
    python scripts/sync_doc_counts.py          # update the file
    python scripts/sync_doc_counts.py --check  # exit 1 if it is out of date
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "PROJECT_DOCUMENTATION.md"
PATTERN = re.compile(r"\*\*(\d+)/(\d+) tests\*\*")


def collected_tests() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--collect-only"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    return len([l for l in out.stdout.splitlines() if "::" in l])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify instead of rewriting")
    args = ap.parse_args()

    n = collected_tests()
    txt = DOC.read_text(encoding="utf-8")
    m = PATTERN.search(txt)
    if not m:
        sys.exit("no test-count claim found in PROJECT_DOCUMENTATION.md")

    if args.check:
        if int(m.group(1)) != n:
            sys.exit(f"PROJECT_DOCUMENTATION.md claims {m.group(1)} tests, "
                     f"pytest collects {n}. Run scripts/sync_doc_counts.py")
        print(f"test count claim is accurate ({n})")
        return

    updated = PATTERN.sub(f"**{n}/{n} tests**", txt, count=1)
    if updated != txt:
        DOC.write_text(updated, encoding="utf-8")
        print(f"updated test count to {n}/{n}")
    else:
        print(f"test count already {n}/{n}")


if __name__ == "__main__":
    main()
