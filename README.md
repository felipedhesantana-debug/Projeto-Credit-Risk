# Projeto-Credit-Risk

# Projeto de Avaliação de Risco de Crédito (Credit Risk Assessment)

Este repositório contém um pipeline completo de Machine Learning para avaliação e previsão de risco de crédito baseado no conjunto de dados [`german_credit_data.csv`](german_credit_data.csv).

## 📁 Estrutura do Projeto

```
Projeto Credit Risk/
├── german_credit_data.csv     # Dataset original
├── eda.py                     # Script de Análise Exploratória de Dados
├── predict.py                 # Script para inferência e simulação em novas propostas
├── src/
│   ├── preprocessing.py       # Pipeline de tratamento, imputação e encoding
│   └── train_model.py         # Treinamento e avaliação de modelos (XGBoost, RF, RegLog)
├── models/
│   └── credit_risk_model.joblib # Artefato do melhor modelo treinado e preprocessor
├── app/
│   ├── api.py                 # API de inferência
│   └── dashboard.py          # Dashboard interativo
├── plots/
│   ├── eda_overview.png       # Visualizações da análise exploratória
│   ├── model_evaluation.png   # Curvas ROC e Matriz de Confusão
│   └── feature_importance.png # Gráfico de importância dos atributos
├── Dockerfile                 # Container para execução da API
├── docker-compose.yml         # Orquestração dos serviços
├── requirements.txt           # Dependências do projeto
├── README.md                 # Documentação do projeto
└── .gitignore                # Arquivos ignorados pelo Git
```

## 🚀 Como Executar

### 1. Análise Exploratória (EDA)
Gere estatísticas descritivas e relatórios visuais:
```bash
python3 eda.py
```
Os gráficos serão salvos na pasta `plots/`.

### 2. Treinamento e Avaliação de Modelos
Treine os modelos (Regressão Logística, Random Forest e XGBoost), avalie as métricas de crédito e salve o melhor modelo:
```bash
PYTHONPATH=src python3 src/train_model.py
```
Métricas geradas:
- **XGBoost**: ROC-AUC: `0.7850`, Recall (Maus Pagadores): `71.67%`
- **Random Forest**: ROC-AUC: `0.7739`
- **Regressão Logística**: ROC-AUC: `0.7607`

### 3. Simulação de Inferência (Predição de Risco)
Rode simulações com novos perfis de solicitantes de crédito:
```bash
PYTHONPATH=src python3 predict.py
```
Ou importe no seu próprio script:
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

### 4. Executar a API/Dashboard
```bash
docker-compose up --build
```
Ou, para executar direto:
```bash
python3 app/api.py
```

## 📌 Objetivo do projeto

Este projeto demonstra um pipeline completo de machine learning para classificação de risco de crédito, incluindo:
- preparação e engenharia de features
- avaliação comparativa de modelos
- otimização da decisão de crédito
- inferência em novos perfis de clientes
- pronto para publicação em GitHub e uso em ambientes de produção

