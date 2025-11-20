"""
Small CLI utility to download, clean, and save Project Gutenberg corpora for
the trigram language model.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

from utils import normalize_whitespace, strip_gutenberg_header_footer


GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    "https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    "https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
]


def download_book(book_id: int) -> str:
    """
    Downloads the raw text for a Project Gutenberg book.

    Args:
        book_id: Numeric Gutenberg identifier (e.g., 11 for
            "Alice's Adventures in Wonderland").

    Returns:
        The raw text of the book.

    Raises:
        RuntimeError: If all URL variants fail to download.
    """
    errors = []
    headers = {"User-Agent": "trigram-assignment/1.0"}

    for template in GUTENBERG_URLS:
        url = template.format(book_id=book_id)
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                raw_bytes = response.read()
                return raw_bytes.decode("utf-8", errors="ignore")
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Unable to download Project Gutenberg text:\n" + "\n".join(errors)
    )


def process_text(raw_text: str) -> str:
    """
    Applies header/footer stripping and whitespace normalization.
    """
    cleaned = strip_gutenberg_header_footer(raw_text)
    return normalize_whitespace(cleaned)


def save_text(text: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and clean a Project Gutenberg book."
    )
    parser.add_argument(
        "--book-id",
        type=int,
        required=True,
        help="Project Gutenberg numeric ID (e.g., 11 for Alice in Wonderland)",
    )
    parser.add_argument(
        "--output",
        default="data/corpus.txt",
        help="Where to store the cleaned text (default: data/corpus.txt)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    raw = download_book(args.book_id)
    cleaned = process_text(raw)
    save_text(cleaned, args.output)
    print(f"Saved cleaned corpus to {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])

