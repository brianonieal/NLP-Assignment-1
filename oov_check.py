"""oov_check.py -- vocabulary coverage of each embedding over the IMDB dev set.

Supporting analysis for the 5.2.8 written answer. glove-twitter-100 outperforms
glove-wiki-gigaword-100 at matched dimension, and there are two competing
explanations: vocabulary size (fewer tokens dropped from the average) versus
register match (better vectors for the words that are present).

Measuring coverage separates them. Run after main.py so the embeddings are
already cached locally.

Usage:  python oov_check.py
"""

import json

import gensim.downloader as api

import model as M
from main import EMBEDDING_CONFIGS


def main():
    """Measure token and type coverage for every configured embedding."""
    _, dev, _ = M.load_data()
    docs = [M.tokenize(text) for _, text in dev]
    total_tokens = sum(len(doc) for doc in docs)
    types = {word for doc in docs for word in doc}

    print(
        f"dev set: {len(dev)} reviews, {total_tokens:,} tokens, "
        f"{len(types):,} unique types\n",
        flush=True,
    )

    coverage = {}
    for name in EMBEDDING_CONFIGS:
        vectors = api.load(name)
        token_hits = sum(1 for doc in docs for word in doc if word in vectors)
        type_hits = sum(1 for word in types if word in vectors)

        coverage[name] = {
            "vocab": len(vectors.index_to_key),
            "dim": vectors.vector_size,
            "token_coverage": token_hits / total_tokens,
            "type_coverage": type_hits / len(types),
        }
        print(
            f"{name:<26} vocab {len(vectors.index_to_key):>9,}  "
            f"dim {vectors.vector_size:>3}  "
            f"token cov {token_hits / total_tokens:6.2%}  "
            f"type cov {type_hits / len(types):6.2%}",
            flush=True,
        )
        del vectors

    with open("results_oov.json", "w", encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2)
    print("\nwrote results_oov.json")


if __name__ == "__main__":
    main()
