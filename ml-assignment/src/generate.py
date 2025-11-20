"""
Command-line entry point for training the trigram model and generating text.
"""

from __future__ import annotations

import argparse
import random

from ngram_model import TrigramModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with the trigram model.")
    parser.add_argument(
        "--corpus",
        default="data/example_corpus.txt",
        help="Path to the training corpus (default: data/example_corpus.txt)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum occurrences required to keep a token out of <unk> (default: 2)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many independent generations to produce (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: random)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    model = TrigramModel(min_count=args.min_count)
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()
    model.fit(text)
    for idx in range(1, args.num_samples + 1):
        generated_text = model.generate(max_length=args.max_length)
        print(f"Generated Text #{idx}:")
        print(generated_text)
        print("-" * 60)


if __name__ == "__main__":
    main()
