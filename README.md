# Fairy Blue · English Essay Scoring

CS4486 mini-project by Group 7. Predict six writing scores with Microsoft DeBERTa-v3-large, then use three teachers to label additional essays for student training.

| Model | Training supervision |
| --- | --- |
| S0 | Human ratings |
| S1 | Human ratings and uniformly weighted pseudo-labels |
| S2 | Human ratings and confidence-weighted pseudo-labels |

S1 and S2 use the same 4,000 auxiliary essays and a pseudo-label loss coefficient of 0.5. Ensembles average predictions from seeds 4486, 4487, and 4488 equally.

## Requirements

- Python 3.11 or newer.
- A CUDA-enabled PyTorch installation and sufficient GPU memory for full DeBERTa-v3-large fine-tuning.
- Kaggle access to download the competition datasets.

```sh
git clone https://github.com/yuangq1/fairy-blue-ell.git
cd fairy-blue-ell
python -m pip install -r requirements.txt
```

Training downloads the [pinned Microsoft checkpoint](https://huggingface.co/microsoft/deberta-v3-large/tree/64a8c8eab3e352a784c658aef62be1662607476f) on first use. Pass `--model-path /path/to/local/snapshot` to use a local copy. Prediction supports `--device cpu`.

## Data

Download the files from their original sources and place them as follows.

| Source | Local path |
| --- | --- |
| [Feedback Prize ELL](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data) | `data/ell/train.csv` and `data/ell/test.csv` |
| [Feedback Prize 2021](https://www.kaggle.com/competitions/feedback-prize-2021/data) | Extract essay text files into `data/fp1/train/`. |
| [ELLIPSE original test archive](https://github.com/scrosseye/ELLIPSE-Corpus/blob/710963bbf1acf23a28e40643d9f937d06f047dd4/ELLIPSE_Final_github_test.zip) | `data/ELLIPSE_Final_github_test.zip` |

The ELLIPSE archive identifies the 2,571 protected essays that preparation excludes from training and auxiliary sampling. Kaggle downloads require accepting the competition terms. An additional `--earlier-dir` can include extracted texts from [Feedback Prize Effectiveness](https://www.kaggle.com/competitions/feedback-prize-effectiveness/data).

Preparation reserves 391 ELL essays for validation. Teachers train on 3,129 essays and calibrate variance heads on another 391; students train on their combined 3,520 essays. Auxiliary sampling uses seed 42 after ID and text deduplication.

## Training

Run these commands from the repository root.

```sh
python data_prep.py --ell-train data/ell/train.csv --earlier-dir data/fp1/train --holdout-inputs data/ELLIPSE_Final_github_test.zip --output-dir data/prepared

python -m students.train --model S0 --seed 4486
python -m teacher.train
python -m teacher.label
python -m students.train --model S1 --seed 4486
python -m students.train --model S2 --seed 4486
```

Teacher seeds are 2006, 7, and 24. The student baseline uses seed 4486. To build an ensemble, also train S1 or S2 with `--seed 4487` and `--seed 4488`.

Defaults use up to 1,024 tokens and an effective batch size of 32. Lower `--micro-batch` if GPU memory is limited. Scoring checkpoints minimize validation MCRMSE; teacher variance heads minimize validation Gaussian NLL. Student training stops after two epochs without improvement, with a maximum of six epochs.

New training runs save their settings, selected metrics, and checkpoints under `runs/`. To retain an existing data split, pass `--split-ids path/to/role_id_lists.json` during preparation. These commands generate new pseudo-labels and train new checkpoints; the paper's exact submission scores also depend on its original split, supervision file, and selected weights.

## Prediction

Use one checkpoint directory or three checkpoint directories from the same student family.

```sh
python predict.py --checkpoints runs/S0_4486/checkpoint --input data/ell/test.csv --output outputs/S0.csv

python predict.py --checkpoints runs/S1_4486/checkpoint runs/S1_4487/checkpoint runs/S1_4488/checkpoint --input data/ell/test.csv --output outputs/S1_ensemble.csv
```

Teacher checkpoints use paths such as `runs/teachers/T2006/checkpoint`. Each checkpoint contains `backbone/`, `tokenizer/`, `heads.safetensors`, and `metadata.json`. Predictions are written as a CSV with `text_id` and the six score columns.

The scripts in `inference/` support historical self-contained Kaggle bundles. They require the matching bundle files; the three-seed script also needs an ensemble manifest supplied with `--manifest`. Use `predict.py` for checkpoints produced by this repository.

## Data analysis

```sh
python -m pip install -r figures/requirements.txt
python figures/prepare_score_statistics.py --train data/ell/train.csv --plots-dir outputs/data_analysis
```

This writes label statistics, score distributions, and a Pearson correlation plot.

## Repository layout

```text
data_prep.py        Data validation, splitting, and auxiliary sampling
teacher/           Teacher training and pseudo-label generation
students/          S0, S1, and S2 training entry point
pipeline/          Shared models, training, and prediction
predict.py         Single-model and ensemble CSV prediction
figures/           Label statistics and data plots
inference/         Historical Kaggle bundle inference
```

## Team

**Fairy Blue · Group 7**

- Gengqi Yuan — 72400647
- Lin’er Yuan — 72400893
- Zhiming Chen — 72400137

See [AUTHORS](AUTHORS) for the author list.
