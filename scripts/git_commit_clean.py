#!/usr/bin/env python3
"""Create a git commit without Co-authored-by trailers."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time


def run(*args: str, input_text: str | None = None) -> str:
    proc = subprocess.run(
        list(args),
        input=input_text.encode() if input_text is not None else None,
        capture_output=True,
        check=True,
    )
    return proc.stdout.decode().strip()


def parse_author(raw: str) -> tuple[str, int, str]:
    match = re.search(r"^author (.+>) (\d+) ([+-]\d+)$", raw, re.MULTILINE)
    if not match:
        raise SystemExit("Could not parse author from HEAD")
    return match.group(1), int(match.group(2)), match.group(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", action="append", required=True)
    args = parser.parse_args()

    tree = run("git", "write-tree")
    try:
        parent = run("git", "rev-parse", "HEAD")
        parents = f"parent {parent}\n"
    except subprocess.CalledProcessError:
        parents = ""

    head_raw = run("git", "cat-file", "-p", "HEAD") if parents else ""
    if parents:
        author, _, tz = parse_author(head_raw)
    else:
        author = run("git", "var", "GIT_AUTHOR_IDENT")
        author = author.rsplit(" ", 2)[0] if author.count(">") else author
        tz = "+0900"

    now = int(time.time())
    body = "\n\n".join(args.message).strip() + "\n"
    commit_text = (
        f"tree {tree}\n"
        f"{parents}"
        f"author {author} {now} {tz}\n"
        f"committer {author} {now} {tz}\n"
        f"\n"
        f"{body}"
    )

    new_hash = run(
        "git", "hash-object", "-w", "-t", "commit", "--stdin", input_text=commit_text
    )
    run("git", "update-ref", "refs/heads/main", new_hash)
    print(new_hash)


if __name__ == "__main__":
    main()
