"""main.py -- entry point for Homework 1.

Usage:
    python main.py basics              # 5.1    PyTorch basics
    python main.py single_run          # 5.2.7  train/dev loss for one embedding
    python main.py explore_embeddings  # 5.2.8  compare four embedding choices
    python main.py all
"""

import gc
import json
import sys
import time

import gensim.downloader as api
import torch

import basics
import model as M

# The four pre-trained embedding configurations under comparison.
#
# The first three hold the training corpus fixed (Wikipedia + Gigaword) and
# vary only the dimension, which isolates the effect of embedding size. The
# fourth holds the dimension fixed at 100 against glove-wiki-gigaword-100 and
# swaps the corpus to Twitter, which isolates the effect of training domain.
#
# word2vec-google-news-300 was considered and dropped: 3M words at 300
# dimensions needs roughly 3.6 GB of RAM to hold as float32.
EMBEDDING_CONFIGS = [
    "glove-wiki-gigaword-50",
    "glove-wiki-gigaword-100",
    "glove-wiki-gigaword-300",
    "glove-twitter-100",
]

DEFAULT_EMBEDDING = "glove-wiki-gigaword-300"
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42


def run_basics():
    """5.1 -- run every PyTorch basics demonstration."""
    basics.tensor_creation()
    basics.tensor_operations()
    basics.math_operations()
    basics.torch_numpy()


def _prepare(splits, embedding_name):
    """Load one embedding, featurize all three splits, then free the embedding."""
    print(f"  loading embeddings: {embedding_name}", flush=True)
    start = time.time()
    embeddings = api.load(embedding_name)
    print(
        f"    vocab {len(embeddings.index_to_key):,}  "
        f"dim {embeddings.vector_size}  ({time.time() - start:.0f}s)",
        flush=True,
    )

    start = time.time()
    train_ds, dev_ds, test_ds = M.build_datasets(splits, embeddings, embedding_name)
    print(f"    featurized in {time.time() - start:.0f}s", flush=True)

    dim = embeddings.vector_size
    del embeddings  # 1.2M x 100 floats is a large share of available RAM
    gc.collect()
    return train_ds, dev_ds, test_ds, dim


def single_run(embedding_name=DEFAULT_EMBEDDING):
    """5.2.7 -- train one classifier and plot train loss against dev loss."""
    print(f"\n=== single_run [{embedding_name}] ===", flush=True)
    torch.manual_seed(SEED)

    splits = M.load_data()
    train_ds, dev_ds, _, dim = _prepare(splits, embedding_name)

    train_loader = M.create_dataloader(train_ds, BATCH_SIZE, shuffle=True)
    dev_loader = M.create_dataloader(dev_ds, BATCH_SIZE, shuffle=False)

    clf = M.SentimentClassifier(embed_size=dim)
    history = M.train(
        clf, train_loader, dev_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE
    )

    M.visualize_epochs(
        history, "single_run.png", title=f"Train vs. dev loss ({embedding_name})"
    )

    with open("results_single_run.json", "w", encoding="utf-8") as handle:
        json.dump({"embedding": embedding_name, "history": history}, handle, indent=2)
    return history


def explore_embeddings():
    """5.2.8 -- train the same classifier on each embedding and compare."""
    print("\n=== explore_embeddings ===", flush=True)
    splits = M.load_data()
    results = {}

    for name in EMBEDDING_CONFIGS:
        print(f"\n[{name}]", flush=True)
        # Re-seed before every config so weight initialization is identical and
        # the only variable across runs is the representation.
        torch.manual_seed(SEED)
        train_ds, dev_ds, _, dim = _prepare(splits, name)
        train_loader = M.create_dataloader(train_ds, BATCH_SIZE, shuffle=True)
        dev_loader = M.create_dataloader(dev_ds, BATCH_SIZE, shuffle=False)

        clf = M.SentimentClassifier(embed_size=dim)
        history = M.train(
            clf,
            train_loader,
            dev_loader,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            verbose=False,
        )
        history["dim"] = dim
        results[name] = history

        print(
            f"  best dev loss {min(history['dev_loss']):.4f}  "
            f"best dev acc {max(history['dev_acc']):.4f}",
            flush=True,
        )

    M.visualize_configs(results)

    with open("results_explore_embeddings.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n  summary")
    header = (
        f"  {'embedding':<28} {'dim':>5} {'best dev loss':>14} {'best dev acc':>13}"
    )
    print(header)
    for name, hist in results.items():
        print(
            f"  {name:<28} {hist['dim']:>5} "
            f"{min(hist['dev_loss']):>14.4f} {max(hist['dev_acc']):>13.4f}"
        )
    return results


if __name__ == "__main__":
    COMMAND = sys.argv[1] if len(sys.argv) > 1 else "all"
    if COMMAND == "basics":
        run_basics()
    elif COMMAND == "single_run":
        single_run()
    elif COMMAND == "explore_embeddings":
        explore_embeddings()
    elif COMMAND == "all":
        run_basics()
        single_run()
        explore_embeddings()
    else:
        print(__doc__)
        sys.exit(1)
