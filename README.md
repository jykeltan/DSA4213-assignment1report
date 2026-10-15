# DSA4213 Assignment 1 — Language Models, Embeddings and Representations

Corpus: *Little Women* (Louisa May Alcott). Part III task: authorship attribution, *Little Men* (Alcott) vs *Pride and Prejudice* (Austen).

The report (PDF) is submitted separately. This repository contains the code needed to reproduce every number in it.

## Repository contents

| File | Purpose |
| --- | --- |
| `data preparation.ipynb` | Downloads the three books from Project Gutenberg, cleans *Little Women*, makes the chapter-level split and builds the vocabulary → `outputs/data_prep.pkl` |
| `part1_language model training.ipynb` | Part I: Kneser–Ney trigram, RNN, LSTM, Transformer; results table, loss curves, generated text. Also writes `lm_utils.py` |
| `part2_embeddings.ipynb` | Part II: own Skip-gram (SGNS) embeddings, GloVe 6B-100d, frozen-embedding LSTMs |
| `part3_downstream representation analysis.ipynb` | Part III: passage dataset, frozen-LM features (mean / final-state pooling), logistic regression, baselines, error analysis |
| `lm_utils.py` | Shared models, training loop, evaluation and generation code (imported by Parts I–III) |
| `requirements.txt` | Python dependencies |

The notebooks are saved **with their outputs** (tables, figures, generated text), so the results can be inspected without re-running. Re-running writes all intermediate files (processed data, trained models, result tables, figures) to an `outputs/` folder.

## Data access

No data files are included; everything is downloaded by the code.

| Data | Source | Licence |
| --- | --- | --- |
| *Little Women* | Project Gutenberg eBook #514 (`https://www.gutenberg.org/cache/epub/514/pg514.txt`) | Public domain |
| *Little Men* | Project Gutenberg eBook #2788 | Public domain |
| *Pride and Prejudice* | Project Gutenberg eBook #1342 | Public domain |
| GloVe 6B-100d | `https://huggingface.co/stanfordnlp/glove` (fallback: `https://nlp.stanford.edu/data/glove.6B.zip`), ~860 MB zip, downloaded in Part II | Stanford NLP, public |

## How to run (Google Colab, recommended)

1. Create the folder `MyDrive/DSA4213_A1/` in your Google Drive and upload the four notebooks into it.
   (To use a different location, change `BASE` in the first code cell of each notebook.)
2. In each notebook: **Runtime → Change runtime type → T4 GPU**.
3. Run the notebooks **in order**, top to bottom: `01` → `02` → `03` → `04`.
   Each notebook mounts Drive, reads what the previous one saved in `outputs/`, and saves its own results there.

Approximate run time on a T4 GPU: `01` < 1 min · `02` ~3 min · `03` ~5 min (mostly the GloVe download) · `04` ~2 min.

**Running locally instead of Colab:** remove the `drive.mount(...)` line in the first cell, set `BASE` to a local folder, install `requirements.txt`, and run in the same order. A GPU is optional (CPU works, but neural training is roughly 50× slower).

## Preprocessing (summary)

- Gutenberg licence header/footer removed; *Little Women* split into its 47 chapters.
- **Interleaved chapter-level split:** every 10th chapter → test, the one before → validation, the rest → train (39 / 4 / 4 chapters).
- Lowercased; curly quotes made straight; `_italic_` markers removed; tokens = words (contractions kept whole) + `. , ! ? ; : "`.
- Vocabulary from the **training split only**, minimum frequency 2 (5,641 types); other words → `<unk>`.
- Part III: text starts at each novel's real first line (removes contents, dedication and the Saintsbury preface in *Pride and Prejudice*); `[Illustration: …]` tags and chapter headings removed; whole sentences packed into 20–64-token passages; block split (blocks of 20 passages) and 50/50 class balancing.

## Configuration and seeds

**Random seed: 42** everywhere (Python, NumPy, PyTorch, sampling, data balancing).

| Setting | Value |
| --- | --- |
| Context length | 64 tokens |
| Embedding dim | 100 (all neural models; matches GloVe-100d) |
| RNN / LSTM | 2 layers, hidden 256, dropout 0.3 |
| Transformer | 2 pre-norm layers, d_model 256, 4 heads, feed-forward 512, dropout 0.3, learned positions, causal mask |
| Optimiser | AdamW, lr 1e-3, batch 32, gradient clipping 1.0 |
| Training budget | max 150 epochs, early stopping patience 3 on validation loss, best-validation weights restored |
| KN trigram | discount D ∈ {0.5, 0.75, 0.9, 0.95} chosen on validation (D = 0.9) |
| SGNS (Part II) | window 5, 5 negatives, subsampling t = 1e-4, 30 epochs, Adam lr 2e-3, batch 1,024 |
| Part III classifier | StandardScaler + logistic regression, C ∈ {0.01, 0.1, 1, 10} chosen on validation accuracy |
| Hardware used for reported results | Google Colab, NVIDIA Tesla T4 |

Results may differ very slightly on different GPUs or library versions because of non-deterministic CUDA kernels.

## Tool use

Code was generated and refined with an AI assistant (Claude, Anthropic); see the AI tool-use statement in the report.

## References

- Kneser & Ney (1995), interpolated Kneser–Ney smoothing.
- Mikolov et al. (2013), Skip-gram with negative sampling.
- Pennington, Socher & Manning (2014), GloVe.
- PyTorch; scikit-learn.
