# Data Notes - Instrument Classification

## Source

This project uses the NSynth dataset from Google Magenta. The original README
describes NSynth as more than 300,000 four-second audio snippets with JSON
metadata.

Raw audio, generated spectrogram images, serialized feature tables, and trained
weights are not committed to this repository.

## Notebook-observed split sizes

The committed notebooks include these metadata counts:

| Split | Observed count |
| --- | ---: |
| Train metadata | 289,205 |
| Validation metadata | 12,678 |
| Test metadata | 4,096 |

The supervised-learning notebook also shows extracted feature table shapes:

| Feature table | Shape |
| --- | --- |
| Train features | `(19012, 167)` |
| Validation features | `(12678, 167)` |

## Expected local layout

A future reproducibility pass should standardize the layout below:

```text
data/
  nsynth/
    train/
    valid/
    test/
    examples.json
    train.json
    valid.json
    test.json
  features/
    df_features_train.pickle
    df_features_valid.pickle
  images/
    train/
    valid/
    test/
```

The current notebooks reference historical paths such as:

- `../data/non_images2`
- `../DataWrangling/df_features_train.pickle`
- `../DataWrangling/df_features_valid.pickle`

## Labels

The supervised-learning notebook uses 11 class names:

`bass`, `brass`, `flute`, `guitar`, `keyboard`, `mallet`, `organ`, `reed`,
`string`, `synth_lead`, `vocal`.

The pretrained CNN notebook shows an 8-class image folder setup:

`brass`, `flute`, `guitar`, `keyboard`, `mallet`, `reed`, `string`, `vocal`.

This class mismatch should be resolved before making final benchmark claims.

## License and attribution

NSynth is a third-party dataset. Any public report or demo should cite the
official NSynth source and comply with the dataset license and attribution
requirements. Do not redistribute raw audio in this repository unless the
license and attribution requirements are explicitly handled.

## Privacy and storage policy

- Do not commit raw NSynth audio.
- Do not commit generated spectrogram folders at full scale.
- Do not commit large model checkpoints in the main branch.
- Use release assets or external storage for heavy artifacts if needed.
