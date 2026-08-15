import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path='german_credit_data.csv', test_size=0.2, random_state=42):
    """
    Carrega o dataset, trata valores ausentes, aplica codificação categórica
    e padronização numérica, dividindo em conjuntos de treino e teste.
    """
    df = pd.read_csv(data_path)
    
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
        
    # Preenchimento de valores ausentes em contas bancárias com 'unknown'
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
    
    # Target encoding: 1 para 'bad' (Risco de Inadimplência), 0 para 'good'
    df['Risk_Binary'] = df['Risk'].map({'good': 0, 'bad': 1})
    
    X = df.drop(columns=['Risk', 'Risk_Binary'])
    y = df['Risk_Binary']
    
    # Identificar colunas numéricas e categóricas
    num_features = ['Age', 'Credit amount', 'Duration', 'Job']
    cat_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    
    # Criar transformador de colunas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )
    
    # Split Treino / Teste estratificado pela variável alvo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Ajustar preprocessor no conjunto de treino e transformar ambos
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Obter nomes das colunas pós One-Hot Encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_features).tolist()
    feature_names = num_features + encoded_cat_cols
    
    return {
        'X_train': X_train_proc,
        'X_test': X_test_proc,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'raw_df': df
    }

if __name__ == '__main__':
    data_dict = load_and_preprocess_data()
    print("Pré-processamento concluído com sucesso!")
    print(f"X_train shape: {data_dict['X_train'].shape}")
    print(f"X_test shape: {data_dict['X_test'].shape}")
    print(f"Número de Features pós-encoding: {len(data_dict['feature_names'])}")
