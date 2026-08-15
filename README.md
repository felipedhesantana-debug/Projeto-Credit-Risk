# Projeto-Credit-Risk

<div align="center">

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/ML-Credit%20Risk-4B8BBE?style=for-the-badge" alt="Credit Risk" />
  <img src="https://img.shields.io/badge/Status-Ready-2EA043?style=for-the-badge" alt="Ready" />

</div>

## Sobre o projeto

Este repositório contém um pipeline completo de Machine Learning para avaliação e previsão de risco de crédito, utilizando o dataset de crédito alemão. O objetivo é classificar clientes em perfis de risco e apoiar decisões de crédito com base em indicadores financeiros e comportamentais.

O projeto inclui:
- análise exploratória de dados
- tratamento de variáveis e preprocessing
- treinamento de múltiplos modelos
- comparação de métricas de avaliação
- inferência em novos perfis
- API e dashboard para uso prático

## Modelo e métricas

Os modelos avaliados incluem Regressão Logística, Random Forest e XGBoost. A comparação foi feita com foco em métricas relevantes para risco de crédito, especialmente recall para maus pagadores e área sob a curva ROC.

| Modelo | ROC-AUC | Recall (maus pagadores) |
|---|---:|---:|
| XGBoost | 0.7850 | 71.67% |
| Random Forest | 0.7739 | - |
| Regressão Logística | 0.7607 | - |

## Estrutura do repositório

```text
Projeto Credit Risk/
├── german_credit_data.csv          # Dataset original
├── eda.py                          # Análise exploratória de dados
├── predict.py                      # Simulação de inferência e predição
├── requirements.txt                # Dependências do projeto
├── Dockerfile                      # Container da aplicação
├── docker-compose.yml              # Orquestração dos serviços
├── README.md                       # Documentação do projeto
├── .gitignore                      # Arquivos ignorados pelo Git
├── models/
│   └── credit_risk_model.joblib   # Modelo treinado + pré-processador
├── plots/
│   ├── eda_overview.png            # Visualizações da EDA
│   ├── model_evaluation.png        # Métricas e avaliação do modelo
│   └── feature_importance.png      # Importância das variáveis
├── src/
│   ├── preprocessing.py           # Pipeline de preprocessing
│   ├── train_model.py             # Treinamento e validação
│   ├── monitoring.py              # Monitoramento
│   └── risk_engine.py             # Motor de risco e lógica de decisão
├── app/
│   ├── api.py                     # API para inferência
│   └── dashboard.py               # Dashboard interativo
└── LICENSE                        # Licença do projeto (se aplicável)
```

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar a análise exploratória

```bash
python eda.py
```

### 3. Treinar o modelo

```bash
PYTHONPATH=src python src/train_model.py
```

### 4. Fazer predição em novos perfis

```bash
PYTHONPATH=src python predict.py
```

### 5. Executar a API / dashboard

```bash
docker-compose up --build
```

Ou, em modo direto:

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

- análise de risco de crédito para fintechs
- apoio à decisão de aprovação/reprovação
- benchmark de modelos em problemas de classificação binária
- prototipagem para sistemas de crédito automatizados

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Docker
- GitHub

## Autor

Felipe Santana

## Licença

Este projeto está disponível para uso educacional e de estudo. Ajuste a licença conforme a sua necessidade antes de uso comercial.

