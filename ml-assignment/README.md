# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model.

## How to Run 

1. (Optional) Create and activate a fresh virtual environment.
2. Install dependencies from the repo root:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Download and clean a Project Gutenberg corpus. Example for
   `Alice's Adventures in Wonderland` (ID 11):
   ```
   cd ml-assignment
   python src/data_pipeline.py --book-id 11 --output data/alice.txt
   cd ..
   ```
   You can swap `--book-id` for any of the recommended titles in the
   assignment brief.
4. From the repository root, execute the tests to validate the model:
   ```
   pytest ml-assignment/tests/test_ngram.py
   ```
5. To train on the sample corpus (or the file you downloaded in step 3) and
   generate text (multiple samples shown for variety):
   ```
   cd ml-assignment
   python src/generate.py --corpus data/alice.txt --max-length 75 --num-samples 3
   ```
   Adjust `--corpus`, `--max-length`, `--min-count`, `--num-samples`, or add
   `--seed` for reproducibility. Replace `data/example_corpus.txt` with any
   cleaned Project Gutenberg text to build a richer model. Re-run the command
   (without a seed) to observe stochastic variation across samples.

## Design Choices

Please document your design choices in the `evaluation.md` file. This should be a 1-page summary of the decisions you made and why you made them.
