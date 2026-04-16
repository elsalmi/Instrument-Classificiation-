# Instrument Classification Report

## Executive summary

This report summarizes the evidence currently committed in the notebook outputs.
It does not retrain models.

The strongest reproducible evidence is the supervised-learning baseline:

- Random forest accuracy: `54.20%`.
- Randomized-search random forest accuracy: `57.57%`.

CNN experiments are present and include confusion matrix/most-confused outputs,
but the repo does not currently include a clear final CNN metric table.

## Dataset

- Dataset: NSynth.
- Audio shape: four-second musical note samples.
- Metadata split counts observed in notebooks:
  - Train: 289,205.
  - Validation: 12,678.
  - Test: 4,096.

## Feature baseline evidence

`4.SupervisedLearning.ipynb` loads extracted feature tables and evaluates random
forest models.

| Experiment | Reported result |
| --- | ---: |
| Random forest accuracy | 54.20% |
| Randomized-search random forest accuracy | 57.57% |

The same notebook plots normalized confusion matrices for both the baseline and
randomized-search random forest.

## CNN evidence status

The CNN notebooks train fast.ai learners over spectrogram-style images:

- `6.CNNModel.ipynb`: non-pretrained ResNet/DenseNet experiments.
- `7.CNNModel_Pretrained.ipynb`: pretrained ResNet/DenseNet experiments.

Committed outputs show:

- FastAI `1.0.60`.
- Librosa `0.7.2`.
- An 8-class pretrained CNN folder setup.
- Confusion matrices.
- Most-confused class pairs such as `brass` vs `reed`, `flute` vs `reed`, and
  `vocal` vs `string`.

Final CNN accuracy is marked pending until a metrics table can be regenerated
from a pinned environment.

## Demo plan

Target demo behavior:

1. Upload a short WAV/MP3 note.
2. Convert the clip into the same spectrogram representation used in the
   notebooks.
3. Run top-k prediction against a small packaged checkpoint or stub model.
4. Display the spectrogram, predicted class, confidence values, and known
   confusion risks.
5. Warn that the model is trained against NSynth-like isolated notes, not
   arbitrary mixed music.

## Risks and next work

- The current repo does not include raw data, feature pickles, generated images,
  or model checkpoints.
- The supervised baseline and CNN path use different class sets in committed
  notebook evidence.
- The next release should regenerate metrics from scripts and export figures to
  `reports/figures/`.
