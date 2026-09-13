# Homework 1 — Setup and Run Instructions

Brian Onieal · 705.641.81 Natural Language Processing: Self-Supervised Models

## Provenance

The assignment PDF refers to a skeleton code base to be filled in, distributed
through Canvas. At the time this was written, the linked repository
(`JHU-EP-705-603-CVA/NLP-Assignment-1`) contained **zero commits**, so no
skeleton was available to complete.

This code base was therefore written from scratch against the structure the PDF
describes (`basics.py`, `model.py`, `main.py`, `hw1.md`) and the function names
it specifies: `tensor_creation`, `tensor_operations`, `math_operations`,
`torch_numpy`, `load_data`, `featurize`, `create_tensor_dataset`,
`create_dataloader`, `SentimentClassifier`, `accuracy`, `train`, `evaluate`,
`visualize_epochs`, `visualize_configs`, `single_run`, `explore_embeddings`.

Hyperparameters the PDF leaves unspecified (optimizer, learning rate, batch
size, epoch count, and the four embedding configurations) were chosen here and
are listed under **Configuration** below.

If the skeleton is published later, the five implementation points the
assignment marks as TODOs are `featurize`, `create_tensor_dataset`,
`SentimentClassifier.__init__`, `SentimentClassifier.forward`, and `accuracy`.
They port across unchanged.

## Environment

Python 3.12 via conda. CPU only, no GPU required.

```bash
conda env create -f environment.yml
conda activate nlp705
```

Or manually:

```bash
conda create -n nlp705 -c conda-forge --override-channels python=3.12 -y
conda activate nlp705
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gensim datasets matplotlib numpy scipy black pylint
```

Versions this was run against:

| package    | version    |
|------------|------------|
| python     | 3.12.14    |
| torch      | 2.14.0+cpu |
| gensim     | 4.4.0      |
| datasets   | 5.0.1      |
| matplotlib | 3.11.2     |
| numpy      | 2.5.3      |
| scipy      | 1.18.1     |
| black      | 26.5.1     |
| pylint     | 4.0.8      |

**Python 3.13 or older is required.** gensim 4.4.0 publishes no wheels for
CPython 3.14, so a 3.14 interpreter cannot run Section 5 even though torch
supports it. The env is pinned to 3.12.

## Running

```bash
python main.py basics              # 5.1    PyTorch basics walkthrough
python main.py single_run          # 5.2.7  train vs. dev loss, one embedding
python main.py explore_embeddings  # 5.2.8  compare four embeddings
python main.py all                 # all three, in order
python oov_check.py                # supplementary: vocabulary coverage
```

Plots land in `plots/`. Per-epoch histories go to `results_single_run.json` and
`results_explore_embeddings.json`. Coverage numbers go to `results_oov.json`.

## First run is slow

Two large downloads happen once and are cached afterwards:

- IMDB via HuggingFace `datasets`, about 210 MB, cached in `~/.cache/huggingface`
- Four Gensim embedding files, about 970 MB total, cached in `~/gensim-data`

`glove-twitter-100` alone is 406 MB and takes roughly 100 seconds just to parse
into a `KeyedVectors` object. Budget 20 to 25 minutes for a cold
`python main.py all`.

Featurized splits are cached per embedding in `.cache/` as pickles, so re-runs
skip featurization. **Delete `.cache/` after changing `load_data`, the split
sizes, or the tokenizer**, because the cache key is the embedding name only and
will otherwise return stale features.

## Configuration

Set at the top of `main.py`:

| setting            | value                                            |
|--------------------|--------------------------------------------------|
| train / dev / test | 20,000 / 1,000 / 1,000                           |
| dev split source   | held out of the training pool, not the test pool |
| model              | `nn.Linear(embed_size, 2)`                       |
| loss               | `nn.CrossEntropyLoss`                            |
| optimizer          | Adam                                             |
| learning rate      | 1e-3                                             |
| batch size         | 64                                               |
| epochs             | 50                                               |
| seed               | 42                                               |
| default embedding  | `glove-wiki-gigaword-300`                        |

The four embedding configurations form a two-factor comparison:

| embedding                 | dim | corpus               | role                             |
|---------------------------|-----|----------------------|----------------------------------|
| `glove-wiki-gigaword-50`  | 50  | Wikipedia + Gigaword | dimension sweep, low             |
| `glove-wiki-gigaword-100` | 100 | Wikipedia + Gigaword | dimension sweep, mid             |
| `glove-wiki-gigaword-300` | 300 | Wikipedia + Gigaword | dimension sweep, high            |
| `glove-twitter-100`       | 100 | Twitter              | corpus swap at matched dimension |

The first three isolate embedding dimension with the corpus held fixed. The
fourth isolates training domain with dimension held fixed against
`glove-wiki-gigaword-100`.

`word2vec-google-news-300` was considered and dropped: 3M words at 300
dimensions needs roughly 3.6 GB of RAM as float32.

## Results

`single_run` on `glove-wiki-gigaword-300`, 50 epochs:

- train loss 0.6399 to 0.3767, monotone decreasing
- dev loss 0.6129 to 0.4137, minimum 0.4118 at epoch 46
- best dev accuracy 0.8380 at epoch 38
- train/dev gap widens from −0.027 at epoch 1 to +0.037 at epoch 50

`explore_embeddings`:

| embedding                 | dim | best dev loss | best dev acc |
|---------------------------|-----|---------------|--------------|
| `glove-wiki-gigaword-50`  | 50  | 0.5157        | 0.7650       |
| `glove-wiki-gigaword-100` | 100 | 0.4715        | 0.7910       |
| `glove-wiki-gigaword-300` | 300 | 0.4118        | 0.8380       |
| `glove-twitter-100`       | 100 | 0.4331        | 0.8170       |

`oov_check.py`, coverage over the 1,000-review dev set (233,012 tokens, 19,367
unique types):

| embedding               | vocab     | token coverage | type coverage |
|-------------------------|-----------|----------------|---------------|
| `glove-wiki-gigaword-*` | 400,000   | 97.43%         | 89.79%        |
| `glove-twitter-100`     | 1,193,514 | 96.45%         | 85.14%        |

This is the evidence behind the 5.2.8 claim that `glove-twitter-100` beating
`glove-wiki-gigaword-100` is a register effect rather than a coverage effect.
The Twitter vocabulary is three times larger but covers the dev set *less* well,
and it still wins on accuracy.

## Reproducibility

`torch.manual_seed(42)` is set before each model is constructed, and
`explore_embeddings` re-seeds before every configuration so weight
initialization is identical across configs and the only variable is the
representation. `load_data` shuffles with a fixed seed before slicing, which
matters because IMDB ships sorted by label; without that shuffle every split
would be single-class.

All results above reproduced to four decimal places across two machines running
different numpy, scipy, and matplotlib versions.

## Code style

Formatted with `black` and linted with `pylint`:

```bash
black basics.py model.py main.py oov_check.py
pylint basics.py model.py main.py oov_check.py --max-line-length=100 \
       --disable=import-outside-toplevel
```

Current rating: 10.00/10.
