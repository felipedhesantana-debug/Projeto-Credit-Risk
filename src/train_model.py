import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from preprocessing import load_and_preprocess_data

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def train_and_evaluate():
    print("="*60)
    print("TREINAMENTO E AVALIAÇÃO DE MODELOS - RISCO DE CRÉDITO")
    print("="*60)
    
    # 1. Carregar dados
    data_dict = load_and_preprocess_data()
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_names = data_dict['feature_names']
    preprocessor = data_dict['preprocessor']
    
    # 2. Definir Modelos
    models = {
        'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', scale_pos_weight=2.33)
    }
    
    results = []
    trained_models = {}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    best_auc = 0
    best_model_name = None
    best_model_obj = None
    
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Modelo': name,
            'Acurácia': acc,
            'Precisão': prec,
            'Recall (Maus Pagadores)': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
        trained_models[name] = model
        
        # Plota curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_obj = model
            
    # Configuração do gráfico ROC
    axes[0].plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.500)')
    axes[0].set_title('Curvas ROC - Comparação de Modelos')
    axes[0].set_xlabel('Taxa de Falsos Positivos (FPR)')
    axes[0].set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Matriz de Confusão do melhor modelo
    best_y_pred = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, best_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bom (0)', 'Ruim (1)'])
    disp.plot(ax=axes[1], cmap='Blues', values_format='d')
    axes[1].set_title(f'Matriz de Confusão - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig('plots/model_evaluation.png', dpi=300)
    plt.close()
    
    # Tabela comparativa
    df_results = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    print("\n" + "="*60)
    print("RESUMO DE DESEMPENHO DOS MODELOS")
    print("="*60)
    print(df_results.to_string(index=False))
    
    print(f"\n--> Melhor Modelo Selecionado: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    
    # Feature Importance (se disponível)
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Importância das Features ({best_model_name})', fontsize=14, fontweight='bold')
        plt.barh([feature_names[i] for i in indices[:10]][::-1], importances[indices[:10]][::-1], color='#3498db')
        plt.xlabel('Importância Relativa')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300)
        plt.close()
        print("--> Gráfico de Importância das Features salvo em 'plots/feature_importance.png'")
        
    # 3. Salvar o melhor modelo e o preprocessor
    pipeline_artifact = {
        'model_name': best_model_name,
        'model': best_model_obj,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }
    
    artifact_path = 'models/credit_risk_model.joblib'
    joblib.dump(pipeline_artifact, artifact_path)
    print(f"--> Modelo e pipeline salvos com sucesso em '{artifact_path}'!\n")

if __name__ == '__main__':
    train_and_evaluate()
