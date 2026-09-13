[English](README.md) | **Português**

# CardioAnalysis

Um notebook educacional de aprendizado de máquina que explora o dataset Heart Failure Clinical Records com XGBoost, visualizações interativas e uma interface com ipywidgets.

## O que o projeto explora

- Preparação de dados tabulares e investigação das relações entre variáveis.
- Treinamento de um classificador para o desfecho `DEATH_EVENT` do dataset.
- Inspeção de resultados com Plotly e testes de entradas por widgets no notebook.

**Ferramentas:** Python · pandas · XGBoost · Plotly · ipywidgets · Jupyter.

## Executar localmente

```bash
git clone https://github.com/MoAbner/CardioAnalysis.git
cd CardioAnalysis
python -m venv .venv
```

Ative o ambiente e instale as dependências registradas e a interface de notebooks:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter
python -m notebook
```

Abra [HealthCare_Prediction.ipynb](HealthCare_Prediction.ipynb), confira o caminho do dataset e execute as células em ordem. O repositório inclui `heart_failure_clinical_records_dataset.csv`. A exibição dos widgets exige uma interface de notebook compatível.

## Escopo e interpretação

Este é um projeto exploratório de aprendizado, sem validação clínica como ferramenta preditiva. As saídas e associações entre variáveis não devem ser apresentadas como diagnósticos, recomendações de tratamento ou evidência de eficácia clínica.

Para avaliações futuras, documente a divisão de treino/teste, verifique vazamento de dados e a disponibilidade de cada variável no momento pretendido da previsão, e relate o desempenho em dados separados. Em particular, revise como o tempo de acompanhamento é utilizado antes de interpretar previsões prospectivas.

## Possíveis próximos passos

Avaliação reproduzível, comparação de hiperparâmetros e documentação mais clara do pré-processamento e das limitações do modelo.
