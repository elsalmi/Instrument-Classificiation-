# Reproducibility Notes - Instrument Classification

## Current state

The repo is notebook-first. It preserves the original exploration path but does
not yet provide a one-command pipeline. The notebooks assume local data folders
and generated pickle/image artifacts that are not committed.

## Legacy environment

The original README lists:

- Python 3.6
- Librosa 0.7.2
- fast.ai 1.0.60
- AWS `g4dn.4xlarge`
- NVIDIA T4 GPU

The pretrained CNN notebook prints:

```text
FastAI Version: 1.0.60
Librosa Version: 0.7.2
```

## Rerun order

1. Download NSynth locally.
2. Run `1.Data_Wrangling.ipynb` to generate feature tables.
3. Run `3.JSON_EDA.ipynb` to validate metadata and split assumptions.
4. Run `4.SupervisedLearning.ipynb` for random forest baselines.
5. Run `5.CNN_Prep.ipynb` to generate spectrogram images.
6. Run `6.CNNModel.ipynb` and `7.CNNModel_Pretrained.ipynb` for CNN experiments.

## Current reported metrics

`4.SupervisedLearning.ipynb` reports:

| Experiment | Accuracy |
| --- | ---: |
| Random forest | 54.20% |
| Randomized-search random forest | 57.57% |

CNN notebooks contain confusion matrix outputs and most-confused class pairs,
but no clear final accuracy table in committed output.

## Next reproducibility pass

1. Add `requirements.txt` or `environment.yml` for the legacy runtime.
2. Move feature extraction into `src/features.py`.
3. Move supervised baselines into `src/train_baseline.py`.
4. Add `src/report.py` to regenerate `reports/REPORT.md`.
5. Save confusion matrices under `reports/figures/`.
