# Musical Instrument Classification

## TL;DR

This project classifies NSynth musical notes by instrument family using a
feature-based supervised path and a separate CNN path.

The current committed evidence is the 11-class supervised baseline:

- Random forest accuracy: `54.20%`
- Randomized-search random forest accuracy: `57.57%`

The CNN notebooks are still a separate path and use an 8-class image-folder
setup, so the two benchmark tracks should not be compared directly until the
class alignment is fixed.

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
| [environment.yml](environment.yml) | Pinned legacy environment for the notebook stack |
| [scripts/rebuild_report.py](scripts/rebuild_report.py) | Rebuilds feature tables and `reports/REPORT.md` |
| [reports/REPORT.md](reports/REPORT.md) | Current metric snapshot and demo plan |

## Problem

Given a short audio note, predict the instrument family. The repo keeps the
evaluation visible:

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

## Reproducibility

Run the notebooks in the order described in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The rebuild script follows
the notebook logic and can regenerate the feature tables plus the short report:

```bash
python scripts/rebuild_report.py --write
```

## Evidence snapshot

The current committed metric evidence comes from `4.SupervisedLearning.ipynb`.

| Experiment | Reported value |
| --- | ---: |
| Random forest accuracy | 54.20% |
| Randomized-search random forest accuracy | 57.57% |

CNN notebooks include confusion matrix and most-confused-pair outputs, but no
clear final metric table is committed. Those metrics remain pending until the
8-class CNN path is aligned with the 11-class supervised baseline.

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

1. Regenerate the metrics from the pinned legacy environment.
2. Align the CNN class set with the 11-class baseline before comparing numbers.
3. Export confusion matrices and metrics into `reports/`.
4. Keep heavy data and generated artifacts out of the main branch.
