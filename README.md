# Musical Instrument Classification

## TL;DR

This project classifies NSynth musical notes by instrument family using two
paths:

1. Feature-based supervised learning over extracted audio features.
2. CNN experiments over spectrogram-style image representations.

The strongest current evidence is the notebook-backed baseline report: a random
forest reached `54.20%` accuracy, and a randomized-search random forest reached
`57.57%` accuracy on the validation split captured in `4.SupervisedLearning.ipynb`.

## What is in this repo

| Artifact | Purpose |
| --- | --- |
| `1.Data_Wrangling.ipynb` | Audio feature extraction and serialized feature tables |
| `3.JSON_EDA.ipynb` | NSynth metadata exploration |
| `4.SupervisedLearning.ipynb` | Random forest baseline and tuned supervised model |
| `5.CNN_Prep.ipynb` | Spectrogram/image preparation |
| `6.CNNModel.ipynb` | CNN experiments without pretrained weights |
| `7.CNNModel_Pretrained.ipynb` | CNN experiments with pretrained fast.ai vision models |
| [docs/DATA.md](docs/DATA.md) | Dataset source, expected layout, and license note |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Environment and rerun plan |
| [reports/REPORT.md](reports/REPORT.md) | Current metric snapshot and demo plan |

## Problem

Given a short audio note, predict the instrument family. The portfolio version
of the project emphasizes:

- Clear dataset provenance.
- A reproducible feature pipeline.
- Baseline metrics before demo claims.
- Confusion-matrix review to identify class pairs that need improvement.

## Data

The project uses Google's NSynth dataset, described in the original README as a
large-scale dataset with more than 300,000 four-second audio snippets and JSON
metadata. Notebook outputs show NSynth metadata splits with 289,205 training
records, 12,678 validation records, and 4,096 test records.

See [docs/DATA.md](docs/DATA.md) for download, layout, and attribution notes.

## Method

The repo explores two modeling approaches:

- **Feature baseline:** extract audio features such as MFCC-style values and
  train scikit-learn models.
- **Image/CNN path:** convert audio to spectrogram-like images and train fast.ai
  CNN learners, including ResNet and DenseNet variants.

The original hardware and software notes were:

- AWS `g4dn.4xlarge`
- 16 vCPU
- NVIDIA T4 GPU
- Python 3.6
- Librosa 0.7.2
- fast.ai 1.0.60

## Reported metrics

The current committed metric evidence comes from `4.SupervisedLearning.ipynb`.

| Experiment | Reported value |
| --- | ---: |
| Random forest accuracy | 54.20% |
| Randomized-search random forest accuracy | 57.57% |

CNN notebooks include confusion matrix and most-confused-pair outputs, but no
clear final metric table is committed. Those metrics are intentionally marked as
pending in [reports/REPORT.md](reports/REPORT.md).

## Demo plan

A useful demo should be lightweight and honest:

1. User uploads a short WAV/MP3 note.
2. App computes the same spectrogram representation used in the notebooks.
3. App returns top-k instrument-family predictions with confidence scores.
4. App displays the spectrogram and warns when the uploaded sample is outside
   NSynth-like assumptions.
5. App links to a confusion matrix so reviewers can understand likely mistakes.

## Reproduce

The current repo is notebook-first and does not yet include raw NSynth data,
generated spectrogram folders, serialized feature tables, or trained model
weights. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the safest
rerun order and environment constraints.

## Next engineering steps

1. Add a pinned environment file for the legacy notebook stack.
2. Extract feature generation into a deterministic script.
3. Export confusion matrices and metrics into `reports/`.
4. Build a small demo using synthetic or user-provided clips rather than
   committing dataset audio.
