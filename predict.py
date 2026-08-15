import sys
import joblib
import pandas as pd

def load_model_pipeline(model_path='models/credit_risk_model.joblib'):
    """Carrega o artefato do modelo treinado e preprocessor."""
    try:
        pipeline_artifact = joblib.load(model_path)
        return pipeline_artifact
    except FileNotFoundError:
        raise FileNotFoundError(f"Modelo não encontrado em '{model_path}'. Execute 'python3 src/train_model.py' primeiro.")

def predict_credit_risk(applicant_data, model_path='models/credit_risk_model.joblib'):
    """
    Recebe um dicionário ou DataFrame com dados do cliente e retorna a predição e probabilidade de risco.
    """
    artifact = load_model_pipeline(model_path)
    model = artifact['model']
    preprocessor = artifact['preprocessor']
    
    if isinstance(applicant_data, dict):
        df_input = pd.DataFrame([applicant_data])
    elif isinstance(applicant_data, pd.DataFrame):
        df_input = applicant_data.copy()
    else:
        raise ValueError("applicant_data deve ser um dicionário ou pandas DataFrame.")
        
    # Tratamento de NAs para contas bancárias
    df_input['Saving accounts'] = df_input['Saving accounts'].fillna('unknown')
    df_input['Checking account'] = df_input['Checking account'].fillna('unknown')
    
    # Aplica o pré-processamento salvo
    X_processed = preprocessor.transform(df_input)
    
    # Predição e probabilidade de inadimplência (Classe 1 = Bad)
    bad_risk_prob = model.predict_proba(X_processed)[:, 1][0]
    prediction_class = model.predict(X_processed)[0]
    
    risk_label = "RUIM (Inadimplência Provável)" if prediction_class == 1 else "BOM (Baixo Risco)"
    
    return {
        'risk_label': risk_label,
        'bad_risk_probability': float(bad_risk_prob),
        'good_risk_probability': float(1 - bad_risk_prob),
        'model_used': artifact['model_name']
    }

if __name__ == '__main__':
    print("="*60)
    print("TESTE DE SIMULAÇÃO DE INFERÊNCIA DE RISCO DE CRÉDITO")
    print("="*60)
    
    # Exemplo 1: Cliente de Alto Risco (Jovem, aluguel, conta zerada/pouco saldo, valor alto e prazo longo)
    high_risk_client = {
        'Age': 21,
        'Sex': 'female',
        'Job': 2,
        'Housing': 'rent',
        'Saving accounts': 'little',
        'Checking account': 'little',
        'Credit amount': 8500,
        'Duration': 48,
        'Purpose': 'car'
    }
    
    # Exemplo 2: Cliente de Baixo Risco (Mais idoso, casa própria, conta rica, valor moderado e prazo curto)
    low_risk_client = {
        'Age': 52,
        'Sex': 'male',
        'Job': 2,
        'Housing': 'own',
        'Saving accounts': 'rich',
        'Checking account': 'rich',
        'Credit amount': 1500,
        'Duration': 12,
        'Purpose': 'radio/TV'
    }
    
    res_high = predict_credit_risk(high_risk_client)
    res_low = predict_credit_risk(low_risk_client)
    
    print("\n--- Cliente 1 (Perfil de Alto Risco) ---")
    print(f"Resultado: {res_high['risk_label']}")
    print(f"Probabilidade de Inadimplência: {res_high['bad_risk_probability']:.2%}")
    
    print("\n--- Cliente 2 (Perfil de Baixo Risco) ---")
    print(f"Resultado: {res_low['risk_label']}")
    print(f"Probabilidade de Inadimplência: {res_low['bad_risk_probability']:.2%}")
    print("="*60)
