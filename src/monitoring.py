import numpy as np
import pandas as pd

def calculate_psi(expected, actual, num_buckets=10):
    """
    Calcula o Population Stability Index (PSI) entre uma distribuição de referência (expected)
    e uma nova safra/amostra (actual).
    
    Parâmetros:
    - expected: Array/Series com os valores de referência (ex: dados de treino)
    - actual: Array/Series com os novos valores (ex: nova safra de propostas)
    - num_buckets: Número de decis/bins para discretização
    
    Retorna:
    - psi_value: Valor escalar do PSI
    - psi_df: DataFrame detalhado por bucket
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # Criar quantis/bins baseados na distribuição de referência (expected)
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    # Remover duplicatas em caso de valores muito repetidos
    bins = np.unique(bins)
    
    # Frequência de ocorrência nos bins
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Percentual de população por bucket
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Evitar divisão por zero ou log de zero na fórmula
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # Componente PSI por bucket: (Actual% - Expected%) * ln(Actual% / Expected%)
    bucket_psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(bucket_psi))
    
    # DataFrame detalhado
    psi_df = pd.DataFrame({
        'Bucket': [f"Bin {i+1}" for i in range(len(bucket_psi))],
        'Expected_Pct': expected_pct,
        'Actual_Pct': actual_pct,
        'Bucket_PSI': bucket_psi
    })
    
    # Classificação do PSI
    if total_psi < 0.10:
        status = "ESTÁVEL (Sem Drift Significativo)"
        color = "green"
    elif total_psi <= 0.25:
        status = "VARIAÇÃO MODERADA (Atenção / Monitoramento)"
        color = "warning"
    else:
        status = "DRIFT CRÍTICO (Necessário Re-treinamento do Modelo)"
        color = "red"
        
    return {
        'psi_value': round(total_psi, 4),
        'status': status,
        'color': color,
        'psi_table': psi_df
    }

if __name__ == '__main__':
    print("="*60)
    print("TESTE DO MÓDULO DE POPULATION STABILITY INDEX (PSI)")
    print("="*60)
    
    np.random.seed(42)
    # Distribuição base de treinos (ex: Scores de crédito de 300 a 850)
    ref_scores = np.random.normal(loc=650, scale=80, size=1000)
    
    # Teste 1: Nova safra idêntica (Sem drift)
    recent_scores_stable = np.random.normal(loc=650, scale=80, size=300)
    res_stable = calculate_psi(ref_scores, recent_scores_stable)
    
    # Teste 2: Nova safra com deterioração de risco (Drift severo)
    recent_scores_drift = np.random.normal(loc=550, scale=90, size=300)
    res_drift = calculate_psi(ref_scores, recent_scores_drift)
    
    print(f"\n1. Safra Estável - PSI: {res_stable['psi_value']} | Status: {res_stable['status']}")
    print(f"2. Safra com Drift - PSI: {res_drift['psi_value']} | Status: {res_drift['status']}")
    print("="*60)
