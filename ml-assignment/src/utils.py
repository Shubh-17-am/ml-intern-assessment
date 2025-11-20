"""
Utility helpers for data preparation and corpus management.
"""

import re
from typing import Tuple


START_FLAG = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_FLAG = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_gutenberg_header_footer(raw_text: str) -> str:
    """
    Removes the standard Project Gutenberg boilerplate so only the book
    content remains.

    Args:
        raw_text: Full text as downloaded from Project Gutenberg.

    Returns:
        Cleaned text without header/footer markers if they were present.
    """
    if not raw_text:
        return ""

    lower = raw_text.lower()
    start_idx = lower.find(START_FLAG.lower())
    end_idx = lower.find(END_FLAG.lower())

    if start_idx != -1:
        start_idx = lower.find("\n", start_idx)
    else:
        start_idx = 0

    if end_idx == -1:
        end_idx = len(raw_text)

    return raw_text[start_idx:end_idx].strip()


def normalize_whitespace(text: str) -> str:
    """
    Collapses multiple spaces/newlines to single spaces for easier downstream
    processing.
    """
    return re.sub(r"\s+", " ", text).strip()
