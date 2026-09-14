#!/usr/bin/env python3
"""Run collection, processing, and report generation in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name)]
    subprocess.run(command, check=True)


def main() -> int:
    run("collect_news.py")
    run("process_news.py")
    run("generate_report.py")
    run("validate_outputs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
