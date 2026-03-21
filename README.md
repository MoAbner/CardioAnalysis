# CardioAnalysis: Suporte à Decisão Clínica

Este projeto utiliza **Aprendizado de Máquina (Ensemble Learning)** para predição de risco em pacientes com insuficiência cardíaca.

---

##  Sobre o Projeto
O sistema processa 12 variáveis clínicas para calcular a probabilidade de óbito. Diferente de modelos "caixa-preta", este projeto foca na **interpretabilidade**, oferecendo um dashboard interativo onde o profissional de saúde pode simular cenários e visualizar os fatores de maior impacto.

###  Tecnologias e Algoritmos
- **Linguagem:** Python 3.x
- **Modelo:** `XGBoost` (Extreme Gradient Boosting) - Escolhido pela alta eficiência em dados tabulares.
- **Interface:** `IPywidgets` com estilização personalizada em CSS para ambiente hospitalar.
- **Gráficos:** `Plotly Express` para visualizações dinâmicas.

---

##  Principais Insights de Sobrevivência
A análise dos pesos do modelo revelou que:
1. **Tempo de Acompanhamento:** O maior preditor de sobrevivência. Estabilidade temporal indica resiliência biológica.
2. **Fração de Ejeção:** Essencial para a saúde mecânica do miocárdio (alvos > 50%).
3. **Creatinina Sérica:** Principal marcador do eixo cardiorrenal; valores elevados (< 1.1 mg/dL) são alertas críticos de falha sistêmica.

---

##  O que pode ser melhorado?
Como todo projeto de engenharia, este é um protótipo funcional que pode evoluir:

- [ ] **Ponderação de Idade:** Implementar uma trava de risco para pacientes idosos, evitando que um "Tempo de Acompanhamento" alto mascare o risco biológico da idade avançada.
- [ ] **Engenharia de Features:** Criar variáveis combinadas (ex: produto entre Idade e Creatinina) para capturar melhor a fragilidade renal.
- [ ] **Persistência de Dados:** Conectar o dashboard a um banco de dados SQL para salvar o histórico de simulações.
- [ ] **Otimização de Hiperparâmetros:** Utilizar `GridSearchCV` ou `Optuna` para refinar ainda mais a acurácia do XGBoost.

---

##  Como Rodar
1. Clone o repositório:
   ```bash
   git clone [https://github.com/MoAbner/CardioAnalysis.git](https://github.com/MoAbner/CardioAnalysis.git)
