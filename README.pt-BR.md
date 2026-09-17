[English](README.md) | **Português**

# CardioAnalysis

Aplicação modular de machine learning para o dataset Heart Failure Clinical
Records. O projeto treina XGBoost e Random Forest, persiste o XGBoost final com
seu `StandardScaler` e disponibiliza inferência por classe Python ou API FastAPI.

Este é um projeto educacional e não uma ferramenta clínica validada. O resultado
não deve ser usado como diagnóstico ou recomendação de tratamento. A variável
`time` representa acompanhamento posterior e deve ser revisada antes de qualquer
uso prospectivo para evitar vazamento temporal.

## Estrutura

```text
CardioAnalysis/
├── src/cardio_analysis/
│   ├── schema.py       # contrato e ordem das 12 features
│   ├── data.py         # leitura, tradução, split e scaling
│   ├── training.py     # treino, métricas e CLI
│   ├── artifacts.py    # bundle versionado modelo + scaler
│   ├── inference.py    # HeartFailurePredictor
│   └── api.py          # API FastAPI
├── scripts/test_pipeline.py  # teste CSV -> treino -> persistência -> predição
├── tests/test_pipeline.py     # entrada para pytest
├── pyproject.toml
└── heart_failure_clinical_records_dataset.csv
```

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Para instalar apenas as dependências de produção, também é possível executar
`python -m pip install -r requirements.txt` e definir `PYTHONPATH=src`. A
instalação editável é recomendada por registrar o pacote e o comando de treino.

## Treinar

```bash
cardio-train \
  --data heart_failure_clinical_records_dataset.csv \
  --model artifacts/model.joblib \
  --metrics artifacts/metrics.json
```

Equivalente sem o comando instalado:

```bash
python -m cardio_analysis.training \
  --data heart_failure_clinical_records_dataset.csv \
  --model artifacts/model.joblib \
  --metrics artifacts/metrics.json
```

O artefato contém o modelo, o scaler e a ordem das features. O JSON contém a
matriz de confusão, F1, ROC AUC e relatório de classificação no mesmo conjunto
de teste criado pelo notebook.

## Validar o pipeline completo

```bash
python scripts/test_pipeline.py
# ou
pytest
```

O teste treina os três modelos presentes no fluxo do notebook, salva e recarrega
o artefato em uma pasta temporária, faz uma predição e verifica que o resultado
persistido é idêntico ao resultado em memória.

## Integrar diretamente em Python

```python
from cardio_analysis import HeartFailurePredictor

predictor = HeartFailurePredictor.from_file("artifacts/model.joblib")
result = predictor.predict({
    "age": 60,
    "anaemia": 0,
    "creatinine_phosphokinase": 250,
    "diabetes": 0,
    "ejection_fraction": 38,
    "high_blood_pressure": 0,
    "platelets": 265000,
    "serum_creatinine": 1.1,
    "serum_sodium": 137,
    "sex": 1,
    "smoking": 0,
    "time": 100,
})
```

`predict()` retorna `prediction` (`0` ou `1`), `survival_probability` e
`death_probability`. A instância deve ser criada uma vez na inicialização do
serviço e reutilizada. Há também `predict_many()` e `analyze()`, que transforma
em dados estruturados os alertas e a simulação existentes no notebook.

## Executar a API

Depois do treinamento:

```bash
uvicorn cardio_analysis.api:app --host 0.0.0.0 --port 8000
```

Por padrão, a API procura `artifacts/model.joblib`. Outro caminho pode ser
definido pela variável `CARDIO_MODEL_PATH`. A documentação interativa fica em
`http://localhost:8000/docs`, e os endpoints são:

- `GET /health`: informa se o artefato está disponível.
- `POST /predict`: recebe as 12 features com os nomes originais do CSV.

Exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":60,"anaemia":0,"creatinine_phosphokinase":250,"diabetes":0,"ejection_fraction":38,"high_blood_pressure":0,"platelets":265000,"serum_creatinine":1.1,"serum_sodium":137,"sex":1,"smoking":0,"time":100}'
```

Para retreinamento contínuo, grave dados e feedback fora deste serviço, versione
o dataset e o artefato e só promova um novo `model.joblib` após uma avaliação
reprodutível. Não faça `fit` durante uma requisição web.
