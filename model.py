"""model.py -- sentiment classifier on IMDB movie reviews, built with PyTorch.

Pipeline:
    raw review string
      -> tokenize
      -> map each token to a pre-trained word vector
      -> average the vectors into one fixed-size feature
      -> single nn.Linear layer -> 2 class scores
      -> nn.CrossEntropyLoss against the 0/1 label
"""

import os
import pickle
import re

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")  # headless backend: write PNGs, never open a window
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position,ungrouped-imports

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ===========================================================================
# 5.2.1  Data loading and splits
# ===========================================================================
def load_data(train_size=20000, dev_size=1000, test_size=1000, seed=42):
    """Download IMDB via HuggingFace, shuffle, and carve out three splits.

    The dev set is held out of the *training* pool, not the test pool, so the
    test set stays untouched until the very end. Sizes are capped at 20k/1k/1k
    to keep training fast, at the cost of some accuracy.

    Returns three lists of (label, text) tuples.
    """
    from datasets import load_dataset

    # datasets >= 5 wants the namespaced repo id; older versions accept the
    # bare alias. Try the canonical path first and fall back.
    try:
        raw = load_dataset("stanfordnlp/imdb")
    except Exception:  # pylint: disable=broad-except
        raw = load_dataset("imdb")

    # IMDB ships sorted by label (all negatives, then all positives), so
    # shuffling before slicing is mandatory. Without it the dev split would be
    # 100% one class and dev accuracy would be meaningless.
    train_pool = raw["train"].shuffle(seed=seed)
    test_pool = raw["test"].shuffle(seed=seed)

    train_raw = train_pool.select(range(train_size))
    dev_raw = train_pool.select(range(train_size, train_size + dev_size))
    test_raw = test_pool.select(range(test_size))

    def to_pairs(dataset):
        return [(int(ex["label"]), ex["text"]) for ex in dataset]

    return to_pairs(train_raw), to_pairs(dev_raw), to_pairs(test_raw)


# ===========================================================================
# 5.2.3  String to feature
# ===========================================================================
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(sentence):
    """Lowercase, drop the HTML line breaks IMDB is full of, split on words."""
    sentence = sentence.replace("<br />", " ")
    return _TOKEN_RE.findall(sentence.lower())


def featurize(sentence, embeddings):
    """Convert a review string to a single averaged embedding tensor.

    Tokenize, map each token to its pre-trained vector, average over all
    tokens, and return a float tensor of shape [embed_size].
    """
    tokens = tokenize(sentence)

    vectors = []
    for token in tokens:
        if token in embeddings:  # skip out-of-vocabulary tokens
            vectors.append(embeddings[token])

    # np.mean over an empty array returns nan, which silently poisons the loss.
    if len(vectors) == 0:
        return torch.zeros(embeddings.vector_size, dtype=torch.float)

    stacked = np.stack(vectors, axis=0)  # [num_tokens, embed_size]
    averaged = np.mean(stacked, axis=0)  # [embed_size]

    # Gensim stores float32, but .float() guards against a float64 source,
    # which would otherwise mismatch nn.Linear's float32 weights.
    return torch.from_numpy(averaged).float()


# ===========================================================================
# 5.2.4  Dataset and dataloader
# ===========================================================================
def create_tensor_dataset(raw_data, embeddings):
    """Featurize every example, then stack into one TensorDataset."""
    all_features, all_labels = [], []
    for label, text in raw_data:
        all_features.append(featurize(text, embeddings))
        all_labels.append(label)

    features_tensor = torch.stack(all_features)  # [N, embed_size]
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # [N], int64
    return TensorDataset(features_tensor, labels_tensor)


def create_dataloader(dataset, batch_size=64, shuffle=False):
    """Wrap a dataset so training can iterate over mini-batches."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_datasets(splits, embeddings, embedding_name):
    """Featurize all three splits, caching to disk since this is the slow step.

    The cache key is the embedding name only. Delete .cache/ after changing
    load_data, the split sizes, or the tokenizer, or you will get stale
    features back.
    """
    safe_name = embedding_name.replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"feats_{safe_name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    datasets = tuple(create_tensor_dataset(split, embeddings) for split in splits)
    with open(cache_path, "wb") as handle:
        pickle.dump(datasets, handle)
    return datasets


# ===========================================================================
# 5.2.5  The model
# ===========================================================================
class SentimentClassifier(nn.Module):
    """Single linear layer mapping an averaged embedding to class scores."""

    def __init__(self, embed_size, num_classes=2):
        super().__init__()
        self.layer_pred = nn.Linear(embed_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inp):
        """Map [batch_size, embed_size] to logits [batch_size, num_classes].

        Returns raw logits. CrossEntropyLoss applies log_softmax internally, so
        applying softmax here would flatten the gradients.
        """
        return self.layer_pred(inp)


# ===========================================================================
# 5.2.6  Accuracy, training, evaluation
# ===========================================================================
def accuracy(logits, labels):
    """Return a tensor of 0s and 1s marking each prediction correct or not."""
    predictions = torch.argmax(logits, dim=1)  # [batch_size]
    return (predictions == labels).float()


def evaluate(model, dataloader):
    """Return (mean loss, mean accuracy) over a dataloader, without gradients."""
    model.eval()
    total_loss, correct, seen = 0.0, 0.0, 0
    with torch.no_grad():
        for features, labels in dataloader:
            logits = model(features)
            loss = model.loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += accuracy(logits, labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def train(model, train_loader, dev_loader, num_epochs=30, lr=1e-3, verbose=True):
    """Forward, loss, backward, step. Track train and dev loss per epoch."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "dev_loss": [], "train_acc": [], "dev_acc": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0.0, 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = model.loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += accuracy(logits, labels).sum().item()
            seen += labels.size(0)

        train_loss, train_acc = running_loss / seen, correct / seen
        dev_loss, dev_acc = evaluate(model, dev_loader)

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["train_acc"].append(train_acc)
        history["dev_acc"].append(dev_acc)

        if verbose:
            print(
                f"  epoch {epoch:3d} | train loss {train_loss:.4f} "
                f"acc {train_acc:.4f} | dev loss {dev_loss:.4f} acc {dev_acc:.4f}",
                flush=True,
            )

    return history


# ===========================================================================
# 5.2.7 / 5.2.8  Visualization
# ===========================================================================
def visualize_epochs(history, filename="single_run.png", title=None):
    """Plot train loss and dev loss against epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        epochs, history["train_loss"], marker="o", markersize=3, label="train loss"
    )
    plt.plot(epochs, history["dev_loss"], marker="s", markersize=3, label="dev loss")

    best = int(np.argmin(history["dev_loss"]))
    plt.axvline(best + 1, linestyle="--", linewidth=1, color="gray")
    plt.annotate(
        f"min dev loss\nepoch {best + 1}",
        xy=(best + 1, history["dev_loss"][best]),
        xytext=(best + 2.5, history["dev_loss"][best] + 0.03),
        fontsize=8,
        color="gray",
    )

    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title or "Train vs. dev loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")
    return path


def visualize_configs(results, filename_prefix="embeddings"):
    """Two bar charts across embedding configs: best dev loss and dev accuracy."""
    # pylint: disable=too-many-locals
    names = list(results.keys())
    dev_losses = [min(results[n]["dev_loss"]) for n in names]
    dev_accs = [max(results[n]["dev_acc"]) for n in names]
    # split each config name onto two lines so the x-axis stays readable
    labels = [n.replace("glove-", "glove-\n") for n in names]
    paths = []

    panels = [
        (dev_losses, "best dev loss", "Dev loss by embedding", "dev_loss"),
        (dev_accs, "best dev accuracy", "Dev accuracy by embedding", "dev_acc"),
    ]
    for values, ylabel, title, suffix in panels:
        plt.figure(figsize=(7.5, 4.5))
        bars = plt.bar(range(len(names)), values, color="#4C72B0", width=0.6)
        for rect, value in zip(bars, values):
            plt.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.xticks(range(len(names)), labels, fontsize=8)
        plt.ylabel(ylabel)
        plt.title(title)
        # zoom the y-axis so differences between configs stay visible
        low, high = min(values), max(values)
        pad = max((high - low) * 0.35, 0.01)
        plt.ylim(max(0, low - pad), high + pad)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"{filename_prefix}_{suffix}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  saved {path}")
        paths.append(path)

    return paths
