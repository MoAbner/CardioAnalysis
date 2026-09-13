**English** | [Português](README.pt-BR.md)

# CardioAnalysis

An educational machine-learning notebook exploring the Heart Failure Clinical Records dataset with XGBoost, interactive visualizations and an ipywidgets interface.

## What the project explores

- Preparing tabular data and investigating feature relationships.
- Training a classifier for the dataset's `DEATH_EVENT` outcome.
- Inspecting results with Plotly and testing inputs through notebook widgets.

**Tools:** Python · pandas · XGBoost · Plotly · ipywidgets · Jupyter.

## Run locally

```bash
git clone https://github.com/MoAbner/CardioAnalysis.git
cd CardioAnalysis
python -m venv .venv
```

Activate the environment, then install the recorded dependencies and the notebook interface:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter
python -m notebook
```

Open [HealthCare_Prediction.ipynb](HealthCare_Prediction.ipynb), check the dataset path and run the cells in order. The repository includes `heart_failure_clinical_records_dataset.csv`. Widget rendering requires a compatible notebook frontend.

## Scope and interpretation

This is an exploratory learning project, not a clinically validated prediction tool. Outputs and feature associations should not be presented as diagnoses, treatment advice or evidence of clinical effectiveness.

For future evaluation, document the train/test split, check for data leakage and whether each feature would be available at the intended prediction time, and report performance on held-out data. In particular, review how follow-up time is used before interpreting predictions prospectively.

## Possible next steps

Reproducible evaluation, hyperparameter comparison and clearer documentation of preprocessing and model limitations.
