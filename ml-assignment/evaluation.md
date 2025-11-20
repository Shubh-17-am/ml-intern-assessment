# Evaluation

## Problem Framing
I treated the assignment as building a reusable trigram language model that can
train on any Project Gutenberg text (tested locally with the provided sample
corpus). The objective was to keep the code dependency-light, easy to test, and
able to generalize beyond the small unit tests.

## Data Extraction & Cleaning
`src/data_pipeline.py` automates pulling corpora straight from Project
Gutenberg. Given a book ID, it tries the common CDN URLs, strips the boilerplate
license text using `strip_gutenberg_header_footer`, normalizes whitespace, and
stores the cleaned text under `data/`. This keeps the repo self-contained—no
manual copy/paste needed—and satisfies the “write code extracting and cleaning
data” requirement.

Once the text is ready, `fit` performs the modeling pipeline:

1. Convert to lowercase for case-insensitive statistics.
2. Split sentences on `[.!?]+` to preserve sentence boundaries.
3. Tokenize with `re.findall(r"\b\w+\b, ...)`, which drops punctuation and
   keeps alphanumeric tokens (covers contractions and Gutenberg metadata).
4. Filter out empty sentences; the remaining list-of-lists feeds directly into
   the n‑gram counter.

This keeps extraction logic inside the model so no extra preprocessing script is
required. For large novels, reading the `.txt` file and passing the string to
`fit` is sufficient.

## N-Gram Storage Strategy
I use a pair of nested dictionaries from `collections`:

- `defaultdict(lambda: defaultdict(int))` maps `(w_{i-2}, w_{i-1})` contexts to a
  histogram of candidate next words.
- `defaultdict(int)` stores the total count per context to avoid recomputing
  sums during generation.

This structure is lightweight, serializable, and keeps count updates `O(1)`.
It also generalizes to any `n >= 2` (configurable via the constructor) without
changing the rest of the code.

## Padding & Unknown Handling
Each sentence receives `n-1` `<s>` tokens up front and a trailing `</s>` so the
model can learn both how sentences start and when they end. Vocabulary is
derived from token frequency with a configurable `min_count` (default 2). Rare
tokens are mapped to `<unk>` via `_normalize_sentence`, ensuring that both
training and generation can cope with words not seen often (or at all). The
vocabulary set always contains `<s>`, `</s>`, and `<unk>` so repeated calls to
`fit` remain consistent.

## Generation & Sampling
`generate` begins with the starting context `(<s>, <s>)` and iteratively samples
words until either `</s>` is drawn or `max_length` is hit. The key steps are:

1. Pull the histogram for the current context; if missing, fall back to the
   default start context to avoid dead ends.
2. Convert counts to an implicit categorical distribution by drawing a random
   number `U(0, total_count)` and accumulating counts until the threshold is
   exceeded. This yields true probabilistic sampling rather than greedy search.
3. Slide the context window forward with the sampled word and continue.

This mirrors multinomial sampling in traditional n‑gram models and naturally
produces varied sentences across runs. The fallback prevents generation from
stalling when encountering contexts unseen during training (common when
`max_length` exceeds average sentence length).

The CLI exposes `--max-length`, `--min-count`, `--num-samples`, and `--seed`
flags so it is easy to demonstrate both stochastic behaviour (multiple samples
per command) and reproducibility (pin the seed when needed).

## Testing & Extensibility
- `pytest ml-assignment/tests/test_ngram.py` validates empty text, short text,
  and normal training/generation.
- The public API intentionally mirrors scikit-learn (`fit`, `generate`) so it
  can slot into notebooks or downstream evaluation easily.
- Additional utilities (e.g., perplexity computation, serialization) can be
  added inside `src/utils.py` without touching the core model.

Overall, this design keeps the implementation transparent while meeting all the
assignment requirements: cleaned input, padded trigrams, unknown handling,
probabilistic generation, documentation, and runnable instructions.

## Qualitative Samples
Using the automated data pipeline plus the new multi-sample generator produced
the following representative outputs (first few tokens shown):

- `Alice in Wonderland` (`--book-id 11`, `--max-length 75`):
  `Generated Text #1: the march hare was the cook and the cook and the`
- `Pride and Prejudice` (`--book-id 1342`, `--min-count 3`):
  `Generated Text #1: said he`
- `Frankenstein` (`--book-id 84`, `--max-length 70`):
  `Generated Text #1: it was less strange that i must <unk> up <unk> beside ...`
- `A Tale of Two Cities` (`--book-id 98`, `--min-count 4`):
  `Generated Text #1: the young lady just now if this doctor s daughter ...`
- `Sense and Sensibility` (`--book-id 161`, `--max-length 65`):
  `Generated Text #1: ferrars i think i can t think how much longer said mrs`

Re-running the same command without a seed yields new sentences each time,
demonstrating the required stochastic sampling.
