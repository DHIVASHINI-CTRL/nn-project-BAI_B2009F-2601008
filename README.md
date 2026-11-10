# Frog Species Classification from Acoustic Call Features Using a Custom Multilayer Perceptron

BAI13143 Neural Network Basics - Individual Project
Name: Dhivashini | Student ID: BAI_B2009F-2601008

A neural network that identifies the species of a frog from 22 acoustic features (MFCCs) of its call.
Training and testing each run in their own Docker container, so the results can be reproduced on another machine.

## 1. Dataset

- **Anuran Calls (MFCCs)**, UCI Machine Learning Repository
- Source: https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs
- Licence: CC BY 4.0 (academic use allowed)
- 7,195 syllables from 60 recordings, 22 MFCC features, 10 species
- Authors: Colonna, Nakamura, Cristo, Gordo (Universidade Federal do Amazonas)
- No missing values and no duplicate rows. Outliers (24.7% of rows have one) are kept, because they reflect real variation in frog calls.

**Split:** whole recordings are assigned to train / validation / test (about 70 / 15 / 15 by samples), so syllables of the same
recording never appear in two sets. The assignment is saved in `data/processed/recording_split.json`.
Only the 22 MFCC columns are inputs. `Family`, `Genus` and `RecordID` are never used as features.

## 2. Project structure

```
nn-project-<studentID>/
├── data/
│   ├── Frogs_MFCCs.csv              raw dataset (mounted into containers, not copied into images)
│   └── processed/                   created by prepare_data.py (recording_split.json is needed for testing)
├── models/                          model.pt, scaler.pkl, label_encoder.pkl, feature_columns.json
├── outputs/                         experiment_log.csv, test_results.json, figures/
├── src/
│   ├── prepare_data.py              checks, plots, split by recording, scaling
│   ├── model.py                     the MLP (22 -> 64 -> 32 -> 10)
│   ├── train.py                     baseline + 3 experiments
│   ├── test.py                      final evaluation on the test set
│   └── predict.py                   predict one new sample
├── Dockerfile.train
├── Dockerfile.test
├── requirements-train.txt
├── requirements-test.txt
└── README.md
```

## 3. Network

| Layer | Size | Activation | Parameters |
|---|---|---|---|
| Input | 22 | - | 0 |
| Hidden 1 | 64 | ReLU (+ dropout 0.3 in the final model) | 1,472 |
| Hidden 2 | 32 | ReLU (+ dropout 0.3 in the final model) | 2,080 |
| Output | 10 | Softmax (inside CrossEntropyLoss) | 330 |
| **Total** | | | **3,882** |

Loss: cross-entropy. Optimiser: Adam, learning rate 0.001, batch size 64, 60 epochs, seed 42.
The final model is the Exp 2 (dropout 0.3) checkpoint at epoch 9, chosen by validation macro F1.

## 4. Run with Docker (recommended)

Requirements: Docker Desktop. Commands are for Windows PowerShell, run from the project folder.
The containers use Python 3.12 and torch 2.9.0 (CPU), set with the build arguments below. On my laptop I trained with
Python 3.14.6 and torch 2.14.0 and got the same best epochs and validation scores.

### 4.1 Train (training container)

```powershell
docker build -f Dockerfile.train -t nn-train --build-arg PYTHON_VERSION=3.12 --build-arg TORCH_VERSION=2.9.0 .
docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" -v "${PWD}\outputs:/app/outputs" nn-train
```

This runs `prepare_data.py` then `train.py` and saves the model, scaler and label encoder to `models/`.

### 4.2 Test (testing container, can run on a different machine)

Copy `data/Frogs_MFCCs.csv`, `data/processed/recording_split.json` and the `models/` folder to the other machine, keeping the same layout.

```powershell
docker build -f Dockerfile.test -t nn-test --build-arg PYTHON_VERSION=3.12 --build-arg TORCH_VERSION=2.9.0 .
docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" -v "${PWD}\outputs:/app/outputs" nn-test
```

Or use the published image instead of building:

```powershell
docker pull dhivashini/nn-test:1.0
docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" -v "${PWD}\outputs:/app/outputs" dhivashini/nn-test:1.0
```

### 4.3 Predict a new sample

```powershell
docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" nn-test python src/predict.py --demo
docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" nn-test python src/predict.py --features "v1,v2,...,v22"
```

`--demo` uses one real row of the test set. `--features` takes the 22 raw MFCC values (between -1 and 1).

## 5. Run without Docker

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch -r requirements-train.txt
python src/prepare_data.py
python src/train.py
python src/test.py
python src/predict.py --demo
```

## 6. Results

Experiments (each changes one thing from the baseline), compared on the validation set:

| Exp | Change | Best val macro F1 | Best epoch |
|---|---|---|---|
| 0 | Baseline (64, 32, Adam 0.001) | 0.7619 | 6 |
| 1 | Wider layers (128, 64) | 0.7949 | 3 |
| 2 | Dropout 0.3 | **0.7997** | 9 |
| 3 | SGD lr 0.01 instead of Adam | 0.7215 | 60 |

Final model on the test set (used once; 1,117 samples):

| Metric | Result |
|---|---|
| Accuracy | 0.8935 |
| Macro F1 | 0.6006 |
| Majority-class baseline accuracy | 0.4727 |
| Target (macro F1 >= 0.85) | Not met |

**Reproducibility:** library versions are pinned and the seed is fixed (42). I got the same validation and test results
on my own computer, in the Docker containers, and on a second computer.

**Second machine:** I ran the published test image on a second Windows computer that has Docker (version 29.9.2) but no
Python installed (`python --version` says "Python was not found"). I downloaded the project from GitHub, ran
`docker pull dhivashini/nn-test:1.0`, and ran the container with the `data`, `models` and `outputs` folders mounted.
The results were exactly the same:

| Result | Training machine | Second machine |
|---|---|---|
| Validation accuracy | 0.9228 | 0.9228 |
| Validation macro F1 | 0.7997 | 0.7997 |
| Test accuracy | 0.8935 | 0.8935 |
| Test macro F1 | 0.6006 | 0.6006 |
| Wrong predictions | 119 of 1,117 | 119 of 1,117 |

`python src/predict.py --demo` also worked there. It predicted AdenomeraAndre (96.0%), and the true species was
AdenomeraAndre. Screenshots of this run are in the report and the slides.

**Limitations:** several species have only 3 to 5 recordings, so their validation and test sets are tiny
(for example 3 test samples for OsteocephalusOophagus). Per-species scores are therefore noisy.
ScinaxRuber was never recognised in the test set.

## 7. Use of generative AI

Generative AI (Claude) was used for learning, planning and debugging support. See the AI declaration in the report.
