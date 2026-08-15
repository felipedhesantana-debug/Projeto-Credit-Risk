# Projeto-Credit-Risk

<div align="center">

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/ML-Credit%20Risk-4B8BBE?style=for-the-badge" alt="Credit Risk" />
  <img src="https://img.shields.io/badge/Status-Ready-2EA043?style=for-the-badge" alt="Ready" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License" />

</div>

<p align="center">
  <strong>Plataforma de avaliação de risco de crédito com Machine Learning, API e dashboard.</strong>
</p>

## Visão geral

Este repositório implementa um pipeline completo para avaliação de risco de crédito usando o dataset alemão de crédito. A solução combina análise exploratória, pré-processamento, treinamento de modelos, explicabilidade e uma interface para uso prático.

O projeto foi pensado para demonstrar um fluxo realista de ML em produção, incluindo:

- análise exploratória de dados (EDA)
- tratamento de variáveis e encoding
- comparação de modelos de classificação
- métricas de negócio e performance
- inferência em novas propostas de crédito
- API REST e dashboard para visualização

## Stack principal

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- FastAPI
- Streamlit / Dash
- Docker

## Métricas do modelo

Os modelos avaliados foram Regressão Logística, Random Forest e XGBoost. O melhor desempenho ficou com o XGBoost, conforme métrica de ROC-AUC e recall para maus pagadores.

| Modelo | ROC-AUC | Recall (maus pagadores) |
|---|---:|---:|
| XGBoost | 0.7850 | 71.67% |
| Random Forest | 0.7739 | - |
| Regressão Logística | 0.7607 | - |

## Arquitetura do projeto

```text
Projeto Credit Risk/
├── german_credit_data.csv          # Dataset original
├── eda.py                          # Análise exploratória de dados
├── predict.py                      # Simulação e inferência em novos perfis
├── requirements.txt                # Dependências do projeto
├── Dockerfile                      # Container da aplicação
├── docker-compose.yml              # Orquestração dos serviços
├── README.md                       # Documentação do projeto
├── LICENSE                         # Licença do projeto
├── .gitignore                      # Arquivos ignorados pelo Git
├── models/
│   └── credit_risk_model.joblib   # Modelo treinado + preprocessador
├── plots/
│   ├── eda_overview.png            # Gráficos de EDA
│   ├── model_evaluation.png        # ROC e matriz de confusão
│   └── feature_importance.png      # Importância das features
├── src/
│   ├── preprocessing.py           # Pipeline de preprocessamento
│   ├── train_model.py             # Treinamento, validação e salvamento
│   ├── monitoring.py              # Monitoramento de estabilidade
│   └── risk_engine.py             # Motor de risco e decisões
├── app/
│   ├── api.py                     # API de inferência
│   └── dashboard.py               # Dashboard interativo
└── .agents/
    └── skills/
```

## Como executar

### 1. Clonar o repositório

```bash
git clone git@github.com:felipedhesantana-debug/Projeto-Credit-Risk.git
cd Projeto-Credit-Risk
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar EDA

```bash
python eda.py
```

### 4. Treinar o modelo

```bash
PYTHONPATH=src python src/train_model.py
```

### 5. Fazer predição em novos perfis

```bash
PYTHONPATH=src python predict.py
```

### 6. Executar a API ou dashboard

```bash
docker-compose up --build
```

Ou execute diretamente:

```bash
python app/api.py
```

## Exemplo de uso

```python
from predict import predict_credit_risk

cliente = {
    'Age': 30,
    'Sex': 'male',
    'Job': 2,
    'Housing': 'own',
    'Saving accounts': 'moderate',
    'Checking account': 'little',
    'Credit amount': 2500,
    'Duration': 12,
    'Purpose': 'car'
}

resultado = predict_credit_risk(cliente)
print(resultado)
```

## Casos de uso

- avaliação de crédito para fintechs e bancos
- apoio à decisão de aprovação/reprovação
- benchmarking de modelos em classificação binária
- prototipagem de sistemas de credit scoring
- visualização de risco e monitoramento de população

## Roadmap

- melhorar explicabilidade com SHAP e análise de fatores
- adicionar testes automatizados
- evoluir para deploy em nuvem
- criar monitoramento de drift e PSI

## Autor

Felipe Santana

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

