# Fairy Blue · CS4486 English Essay Scoring

Train six-trait DeBERTa-v3-large scorers, label earlier Feedback Prize essays with three teachers, then train S0, S1 and S2. See [AUTHORS](AUTHORS) for the project team.

## Download

Install the Python dependencies with `python -m pip install -r requirements.txt`. Use a CUDA-enabled PyTorch build for full DeBERTa-v3-large training.

| Source | Local location |
|---|---|
| [Feedback Prize ELL](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data) | Extract `train.csv` and `test.csv` into `data/ell/`. |
| [Feedback Prize 2021](https://www.kaggle.com/competitions/feedback-prize-2021/data) | Extract the `train/` directory of essays to `data/fp1/train/`. |
| [ELLIPSE original test archive](https://github.com/scrosseye/ELLIPSE-Corpus/blob/710963bbf1acf23a28e40643d9f937d06f047dd4/ELLIPSE_Final_github_test.zip) | Download the ZIP as `data/ELLIPSE_Final_github_test.zip`. Preparation uses it only to exclude the protected essays. |

Sign in to Kaggle and accept the competition terms to download its files. Training fetches the [pinned Microsoft checkpoint](https://huggingface.co/microsoft/deberta-v3-large/tree/64a8c8eab3e352a784c658aef62be1662607476f) on first use. Pass `--model-path` to the training commands to use a local snapshot.

## Train and predict

Run from the project root.

```sh
python data_prep.py --ell-train data/ell/train.csv --earlier-dir data/fp1/train --holdout-inputs data/ELLIPSE_Final_github_test.zip --output-dir data/prepared
python -m students.train --model S0 --seed 4486
python -m teacher.train
python -m teacher.label
python -m students.train --model S1 --seed 4486
python -m students.train --model S2 --seed 4486
python predict.py --checkpoints runs/S0_4486/checkpoint --input data/ell/test.csv --output outputs/S0.csv
```

Preparation excludes gold and protected essays by ID and normalized text, then samples 4,000 earlier essays. A second `--earlier-dir` can add [Feedback Prize Effectiveness](https://www.kaggle.com/competitions/feedback-prize-effectiveness/data) training texts. Use `--split-ids` with an existing role ID file to retain a saved split. The default creates a deterministic split. This workflow regenerates pseudo labels and fits new checkpoints.

Teacher checkpoints are in `runs/teachers/T2006/checkpoint`, `T7/checkpoint`, and `T24/checkpoint` under the same parent. `teacher.label` writes `data/pseudo/pseudo.jsonl`. S1 uses uniform pseudo-label weights and S2 uses per-trait uncertainty weights. Both use a pseudo-loss coefficient of 0.5. Checkpoints are selected by validation MCRMSE. Lower `--micro-batch` if GPU memory is limited. The effective batch stays at 32 by default.

For teacher predictions, use `runs/teachers/T2006/checkpoint` with `predict.py`. For an S1 or S2 ensemble, also train the family at seeds 4487 and 4488 and pass all three checkpoint directories to `--checkpoints`. Predictions are averaged per trait. Use `--device cpu` for CPU prediction. The scripts in `inference/` also read the earlier self-contained Kaggle checkpoint format.

## Data plots

```sh
python -m pip install -r figures/requirements.txt
python figures/prepare_score_statistics.py --train data/ell/train.csv --plots-dir outputs/data_analysis
```

This writes six-trait summary statistics and two PNGs for score distributions and Pearson correlations.
