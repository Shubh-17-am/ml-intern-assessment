import random
import re
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Tuple


class TrigramModel:
    """
    Simple trigram language model that supports text cleaning, padding,
    handling unknown tokens and probabilistic generation.
    """

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    def __init__(self, n: int = 3, min_count: int = 2):
        """
        Initializes the TrigramModel.

        Args:
            n (int): The size of the n-gram. Defaults to 3 (trigram).
            min_count (int): Minimum frequency required for a token to be added
                to the vocabulary. Rare tokens are replaced with <unk>.
        """
        if n < 2:
            raise ValueError("n must be >= 2 for an n-gram model.")

        self.n = n
        self.min_count = min_count
        self.counts: DefaultDict[Tuple[str, ...], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.context_totals: DefaultDict[Tuple[str, ...], int] = defaultdict(int)
        self.vocab = {self.UNK_TOKEN, self.END_TOKEN, self.START_TOKEN}
        self._trained = False

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _reset_model_state(self) -> None:
        self.counts.clear()
        self.context_totals.clear()
        self.vocab = {self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN}
        self._trained = False

    def _prepare_sentences(self, text: str) -> List[List[str]]:
        """
        Cleans the raw text and returns a list of tokenized sentences.
        """
        lowered = text.lower()
        # Split sentences on punctuation marks.
        sentences = re.split(r"[.!?]+", lowered)
        tokenized: List[List[str]] = []
        for sentence in sentences:
            tokens = re.findall(r"\b\w+\b", sentence)
            if tokens:
                tokenized.append(tokens)
        return tokenized

    def _build_vocabulary(self, sentences: List[List[str]]) -> None:
        flat_tokens = [token for sentence in sentences for token in sentence]
        frequency = Counter(flat_tokens)

        vocab = {token for token, count in frequency.items() if count >= self.min_count}
        if not vocab:
            vocab = set(frequency.keys())

        vocab.update({self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN})
        self.vocab = vocab

    def _normalize_sentence(self, sentence: List[str]) -> List[str]:
        normalized = []
        for token in sentence:
            normalized.append(token if token in self.vocab else self.UNK_TOKEN)
        return normalized

    def _update_counts(self, tokens: List[str]) -> None:
        for idx in range(len(tokens) - (self.n - 1)):
            context = tuple(tokens[idx : idx + self.n - 1])
            target = tokens[idx + self.n - 1]
            self.counts[context][target] += 1
            self.context_totals[context] += 1

    def _sample_next_word(self, context: Tuple[str, ...]) -> str:
        if context not in self.counts:
            # Fallback to the most generic context (sentence start)
            context = tuple([self.START_TOKEN] * (self.n - 1))
            if context not in self.counts:
                return self.END_TOKEN

        context_counts = self.counts[context]
        total = self.context_totals.get(context, sum(context_counts.values()))
        if total == 0:
            return self.END_TOKEN

        threshold = random.uniform(0, total)
        cumulative = 0.0
        for word, count in context_counts.items():
            cumulative += count
            if cumulative >= threshold:
                return word
        # Numerical edge case: return end token.
        return self.END_TOKEN

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def fit(self, text: str) -> None:
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        stripped = text.strip()
        self._reset_model_state()

        if not stripped:
            return

        sentences = self._prepare_sentences(stripped)
        if not sentences:
            return

        self._build_vocabulary(sentences)

        for sentence in sentences:
            normalized_sentence = self._normalize_sentence(sentence)
            padded = (
                [self.START_TOKEN] * (self.n - 1)
                + normalized_sentence
                + [self.END_TOKEN]
            )
            self._update_counts(padded)

        self._trained = True

    def generate(self, max_length: int = 50) -> str:
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        if not self._trained or not self.counts:
            return ""

        context = tuple([self.START_TOKEN] * (self.n - 1))
        generated: List[str] = []

        for _ in range(max_length):
            next_word = self._sample_next_word(context)
            if next_word == self.END_TOKEN:
                break
            generated.append(next_word)
            context = tuple(list(context[1:]) + [next_word])

        return " ".join(generated)
