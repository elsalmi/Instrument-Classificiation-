# Reproducibility Notes - Instrument Classification

## Current state

The repo is notebook-first, but it now has a lightweight rebuild path:

- `environment.yml` pins the legacy stack.
- `scripts/rebuild_report.py` can rebuild the feature tables and short report.

## Legacy environment

The original notebook stack was:

- Python 3.6
- Librosa 0.7.2
- fast.ai 1.0.60
- AWS `g4dn.4xlarge`
- NVIDIA T4 GPU

Use `environment.yml` as the starting point for that runtime.

## Rebuild order

1. Download NSynth locally using the layout in [docs/DATA.md](docs/DATA.md).
2. Run `scripts/rebuild_report.py` to rebuild the train/valid feature tables.
3. Inspect `reports/REPORT.md` for the current supervised baseline metrics.
4. Use `5.CNN_Prep.ipynb`, `6.CNNModel.ipynb`, and `7.CNNModel_Pretrained.ipynb`
   for the CNN path.

## Current reported metrics

`4.SupervisedLearning.ipynb` reports:

| Experiment | Accuracy |
| --- | ---: |
| Random forest | 54.20% |
| Randomized-search random forest | 57.57% |

## Class-set warning

The supervised baseline uses 11 classes, while the CNN notebooks currently use
an 8-class image-folder setup. The class sets need to be aligned before any
final cross-path benchmark claim.

## Hygiene

- Do not commit raw NSynth audio.
- Do not commit generated spectrogram folders or model checkpoints.
- Keep notebook checkpoints and other scratch output ignored.
