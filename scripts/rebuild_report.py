#!/usr/bin/env python3
"""Rebuild the instrument-classification feature tables and short report.

This script mirrors the notebook workflow:

- sample the training split per instrument family,
- extract audio features from NSynth clips,
- train the baseline and randomized-search random forests,
- summarize the current evidence into ``reports/REPORT.md``.

The CNN notebooks remain a separate path. They currently use an 8-class image
folder setup, while the supervised baseline keeps the 11-class instrument set.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


RANDOM_STATE = 2020
CLASS_NAMES = [
    "bass",
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "organ",
    "reed",
    "string",
    "synth_lead",
    "vocal",
]

def feature_extract(file_path: Path) -> list[object]:
    """Mirror the notebook feature extractor for a single audio clip."""

    y, sr = librosa.load(file_path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic = int(np.mean(y_harmonic) > np.mean(y_percussive))

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    spectrogram = np.mean(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), axis=1
    )
    chroma = np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    return [harmonic, mfcc, spectrogram, chroma, contrast]


def instrument_code(filename: str) -> int | None:
    for idx, name in enumerate(CLASS_NAMES):
        if name in filename:
            return idx
    return None


def load_examples(data_root: Path, split: str) -> pd.DataFrame:
    json_path = data_root / f"nsynth-{split}" / "examples.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing NSynth metadata file: {json_path}")
    return pd.read_json(json_path, orient="index")


def sample_training_split(df_train_raw: pd.DataFrame) -> pd.DataFrame:
    sampled = (
        df_train_raw.groupby("instrument_family", as_index=False, group_keys=False)
        .apply(lambda df: df.sample(2000, replace=True, random_state=RANDOM_STATE))
        .copy()
    )
    return sampled[sampled["instrument_family"] != 9]


def build_feature_table(df_split: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    feature_map: Dict[str, list[object]] = {}
    for file_id in df_split.index.tolist():
        feature_map[file_id] = feature_extract(audio_dir / f"{file_id}.wav")

    features = pd.DataFrame.from_dict(
        feature_map,
        orient="index",
        columns=["harmonic", "mfcc", "spectro", "chroma", "contrast"],
    )

    mfcc = pd.DataFrame(features.mfcc.values.tolist(), index=features.index).add_prefix(
        "mfcc_"
    )
    spectro = pd.DataFrame(
        features.spectro.values.tolist(), index=features.index
    ).add_prefix("spectro_")
    chroma = pd.DataFrame(
        features.chroma.values.tolist(), index=features.index
    ).add_prefix("chroma_")

    # Preserve the notebook behavior exactly: the contrast block was accidentally
    # copied from the chroma columns, and the current committed metrics reflect
    # that notebook state.
    contrast = chroma.add_prefix("contrast_")

    features = features.drop(labels=["mfcc", "spectro", "chroma", "contrast"], axis=1)
    df_features = pd.concat(
        [features, mfcc, spectro, chroma, contrast], axis=1, join="inner"
    )
    df_features["targets"] = [instrument_code(name) for name in df_features.index]
    return df_features


def load_or_build_features(
    split: str, data_root: Path, feature_root: Path
) -> pd.DataFrame:
    feature_root.mkdir(parents=True, exist_ok=True)
    pickle_path = feature_root / f"df_features_{split}.pickle"
    if pickle_path.exists():
        with pickle_path.open("rb") as handle:
            return pickle.load(handle)

    df_split = load_examples(data_root, split)
    if split == "train":
        df_split = sample_training_split(df_split)
    audio_dir = data_root / f"nsynth-{split}" / "audio"
    df_features = build_feature_table(df_split, audio_dir)
    with pickle_path.open("wb") as handle:
        pickle.dump(df_features, handle)
    return df_features


def most_confused_pairs(y_true: Sequence[int], y_pred: Sequence[int]) -> List[str]:
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: List[Tuple[int, int, int]] = []
    for true_idx, row in enumerate(cm):
        for pred_idx, count in enumerate(row):
            if true_idx != pred_idx and count:
                pairs.append((count, true_idx, pred_idx))
    pairs.sort(reverse=True)
    top = []
    for count, true_idx, pred_idx in pairs[:3]:
        top.append(f"{CLASS_NAMES[true_idx]} -> {CLASS_NAMES[pred_idx]} ({count})")
    return top


def train_models(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, object]:
    x_train = train_df.drop(labels=["targets"], axis=1)
    y_train = train_df["targets"]
    x_valid = valid_df.drop(labels=["targets"], axis=1)
    y_valid = valid_df["targets"]

    rf = RandomForestClassifier(
        n_estimators=20, max_depth=50, warm_start=True, random_state=RANDOM_STATE
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_valid)

    param_dist = {
        "n_estimators": [20, 40, 60],
        "max_depth": [10, 20, 30, 40],
        "max_features": sp_randint(4, 10),
        "min_samples_split": sp_randint(2, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    tuned_pred = search.predict(x_valid)

    return {
        "baseline_accuracy": accuracy_score(y_valid, rf_pred),
        "tuned_accuracy": accuracy_score(y_valid, tuned_pred),
        "most_confused_pairs": most_confused_pairs(y_valid, tuned_pred),
        "train_shape": train_df.shape,
        "valid_shape": valid_df.shape,
    }


def render_report(metrics: dict[str, object]) -> str:
    confused = metrics["most_confused_pairs"]
    confused_lines = "\n".join(f"- {item}" for item in confused) if confused else "- None"

    return f"""# Instrument Classification Report

This report is generated from the notebook-derived feature tables and model
evidence in this repository. The supervised baseline uses the 11-class NSynth
instrument set, while the CNN notebooks currently use an 8-class image-folder
setup. Those paths should not be compared directly until the class alignment is
fixed.

## Executive summary

The strongest committed evidence is the supervised-learning baseline:

- Random forest accuracy: `{metrics["baseline_accuracy"]:.2%}`
- Randomized-search random forest accuracy: `{metrics["tuned_accuracy"]:.2%}`

## Dataset

- Training feature table shape: `{metrics["train_shape"]}`
- Validation feature table shape: `{metrics["valid_shape"]}`
- Source dataset: NSynth
- Scope: four-second musical-note clips with instrument-family labels

## Feature baseline evidence

| Experiment | Reported result |
| --- | ---: |
| Random forest accuracy | {metrics["baseline_accuracy"]:.2%} |
| Randomized-search random forest accuracy | {metrics["tuned_accuracy"]:.2%} |

## Confusion matrix notes

Most-confused class pairs from the tuned random forest:

{confused_lines}

## CNN evidence status

The CNN notebooks are still a separate path. They preserve confusion-matrix
outputs, but the final benchmark table remains pending until the class set is
aligned and the notebooks are rerun in a pinned environment.

## Demo plan

1. User uploads a short WAV/MP3 note.
2. The clip is converted to the same spectrogram representation used in the
   notebooks.
3. The model returns top-k instrument-family predictions with confidence
   scores.
4. The UI displays the spectrogram and flags NSynth-like assumptions.

## Risks and next work

- The current repo does not include raw NSynth data, generated spectrograms, or
  trained checkpoints.
- The 11-class baseline and 8-class CNN path need alignment before any final
  cross-path benchmark claim.
- Rebuild the report with `python scripts/rebuild_report.py --write` after the
  feature tables and environment are in place.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing the nsynth-<split>/ folders.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data/features"),
        help="Directory used to cache the feature pickle tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/REPORT.md"),
        help="Report output path.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the report to disk instead of printing it.",
    )
    args = parser.parse_args()

    train_df = load_or_build_features("train", args.data_root, args.feature_root)
    valid_df = load_or_build_features("valid", args.data_root, args.feature_root)
    metrics = train_models(train_df, valid_df)
    report = render_report(metrics)

    if args.write:
        args.output.write_text(report)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
