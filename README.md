# NLP Assignment 1

Homework 1 for **705.641.81 Natural Language Processing: Self-Supervised
Models**, Johns Hopkins University Engineering for Professionals.

Brian Onieal

Sentiment classification on IMDB movie reviews using averaged pre-trained word
embeddings and a single linear layer in PyTorch, plus a comparison of four
embedding configurations.

## A note on structure

The assignment PDF refers to a skeleton code base to be filled in. At the time
this was written the linked repository `JHU-EP-705-603-CVA/NLP-Assignment-1`
contained zero commits, so no skeleton was available. This code base was written
from scratch against the file layout and function names the PDF specifies.
Hyperparameters the PDF leaves open (optimizer, learning rate, batch size, epoch
count, and the four embedding configurations) were chosen here and are
documented in [`hw1.md`](hw1.md).

## Quick start

```bash
conda env create -f environment.yml
conda activate nlp705
python main.py all
```

First run downloads roughly 1.2 GB (the IMDB dataset and four Gensim embedding
files) and takes 20 to 25 minutes on CPU. Everything is cached afterwards.

Requires Python 3.13 or older. gensim publishes no wheels for CPython 3.14, so
a 3.14 interpreter cannot run this even though PyTorch supports it.

## Contents

| file | purpose |
|---|---|
| `basics.py` | Section 5.1: tensor creation, operations, math, NumPy bridge |
| `model.py` | Section 5.2: data loading, featurization, model, training, plots |
| `main.py` | entry point and all configuration |
| `oov_check.py` | supplementary vocabulary coverage analysis |
| `hw1.md` | full setup, configuration, and results documentation |
| `environment.yml` | conda environment specification |
| `plots/` | generated figures |
| `results_*.json` | per-epoch histories and coverage numbers |

## Commands

```bash
python main.py basics              # 5.1    PyTorch walkthrough
python main.py single_run          # 5.2.7  train vs. dev loss, one embedding
python main.py explore_embeddings  # 5.2.8  compare four embeddings
python main.py all                 # all three, in order
python oov_check.py                # vocabulary coverage per embedding
```

## Pipeline

```
raw review string
  -> tokenize (lowercase, strip <br />, split on word characters)
  -> map each token to its pre-trained word vector, dropping OOV tokens
  -> average the vectors into one fixed-size feature
  -> nn.Linear(embed_size, 2)
  -> nn.CrossEntropyLoss against the 0/1 label
```

Splits are capped at 20,000 train / 1,000 dev / 1,000 test. The dev set is held
out of the training pool so the test set stays untouched.

## Results

`single_run` on `glove-wiki-gigaword-300`, 50 epochs: train loss falls
monotonically from 0.6399 to 0.3767 while dev loss bottoms out at 0.4118 on
epoch 46. Best dev accuracy 0.8380. The gap widens from −0.027 to +0.037, which
is mild rather than a classic overfitting curve, because 300 averaged dimensions
into one linear layer is only 602 parameters against 20,000 training examples.

`explore_embeddings`:

| embedding | dim | best dev loss | best dev acc |
|---|---|---|---|
| `glove-wiki-gigaword-50` | 50 | 0.5157 | 0.7650 |
| `glove-wiki-gigaword-100` | 100 | 0.4715 | 0.7910 |
| `glove-wiki-gigaword-300` | 300 | 0.4118 | 0.8380 |
| `glove-twitter-100` | 100 | 0.4331 | 0.8170 |

The first three isolate embedding dimension with the corpus held fixed. The
fourth swaps the corpus to Twitter at the same dimension as
`glove-wiki-gigaword-100`, and it wins, 0.8170 to 0.7910.

That is not a vocabulary-coverage effect. `oov_check.py` measures both
vocabularies against the dev set, and the Twitter embedding covers *less* of it
(96.45% of tokens, 85.14% of types) than Wikipedia + Gigaword (97.43% and
89.79%) despite a vocabulary three times larger. The advantage comes from
register: Twitter text is informal and opinion-bearing, much closer to
movie-review language than encyclopedia and newswire prose.

One ceiling worth naming: averaging every token vector discards word order and
negation before the classifier sees anything, so "not good" and "good" produce
nearly identical features. That caps what any embedding choice can buy.

## Reproducibility

Seeds are fixed at 42 and re-set before every model construction, so weight
initialization is identical across configurations and the only variable is the
representation. `load_data` shuffles before slicing, which matters because IMDB
ships sorted by label.

All results above reproduced to four decimal places across two machines running
different numpy, scipy, and matplotlib versions.

## Code style

Formatted with `black`, linted with `pylint` at 10.00/10.

```bash
black basics.py model.py main.py oov_check.py
pylint basics.py model.py main.py oov_check.py --max-line-length=100 \
       --disable=import-outside-toplevel
```
