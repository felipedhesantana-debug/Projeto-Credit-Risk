import os
import joblib
import pandas as pd
import numpy as np
import shap

# Tabela de LGD (Loss Given Default) estimada por tipo de finalidade/colateral (Prática bancária)
LGD_BY_PURPOSE = {
    'car': 0.35,                  # Recuperação parcial por garantia veicular
    'furniture/equipment': 0.50, # Recuperação média
    'radio/TV': 0.60,
    'domestic appliances': 0.60,
    'business': 0.55,
    'repairs': 0.65,
    'education': 0.70,           # Sem garantia real
    'vacation/others': 0.75      # Sem garantia real
}
DEFAULT_LGD = 0.50

class CreditRiskEngine:
    def __init__(self, model_path='models/credit_risk_model.joblib'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em '{model_path}'. Treine o modelo primeiro.")
        
        self.artifact = joblib.load(model_path)
        self.model = self.artifact['model']
        self.preprocessor = self.artifact['preprocessor']
        self.feature_names = self.artifact['feature_names']
        
        # Inicializa o explicador SHAP para a árvore XGBoost
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None

    def calculate_score(self, pd_val):
        """
        Converte a Probabilidade de Default (PD) em um Credit Score padronizado (300 a 850).
        Fórmula Log-Odds: Score = 600 + 50 * ln((1-PD)/PD)
        """
        pd_clamped = np.clip(pd_val, 1e-5, 1 - 1e-5)
        odds = (1 - pd_clamped) / pd_clamped
        score = 600 + 50 * np.log(odds)
        return int(np.clip(score, 300, 850))

    def estimate_lgd(self, purpose):
        """Retorna o Loss Given Default estimado com base na finalidade do empréstimo."""
        return LGD_BY_PURPOSE.get(str(purpose).lower(), DEFAULT_LGD)

    def evaluate_applicant(self, applicant_data):
        """
        Avalia uma proposta de crédito completa e calcula PD, LGD, EAD, EL, Score e SHAP values.
        """
        if isinstance(applicant_data, dict):
            df_input = pd.DataFrame([applicant_data])
        elif isinstance(applicant_data, pd.DataFrame):
            df_input = applicant_data.copy()
        else:
            raise ValueError("applicant_data deve ser dicionário ou DataFrame.")

        # Imputação de nulos para contas bancárias
        df_input['Saving accounts'] = df_input['Saving accounts'].fillna('unknown')
        df_input['Checking account'] = df_input['Checking account'].fillna('unknown')
        
        # Pré-processamento
        X_proc = self.preprocessor.transform(df_input)
        
        # 1. PD - Probability of Default
        pd_val = float(self.model.predict_proba(X_proc)[:, 1][0])
        
        # 2. EAD - Exposure at Default (Valor total do crédito solicitado)
        ead_val = float(df_input['Credit amount'].iloc[0])
        
        # 3. LGD - Loss Given Default
        purpose_val = df_input['Purpose'].iloc[0]
        lgd_val = float(self.estimate_lgd(purpose_val))
        
        # 4. Expected Loss (EL) = PD * LGD * EAD
        expected_loss = float(pd_val * lgd_val * ead_val)
        
        # 5. Credit Scorecard
        score = self.calculate_score(pd_val)
        
        # Decisão de Crédito
        if score >= 650:
            decision = "APROVADO"
            decision_detail = "Proposta com baixo risco financeiro dentro dos limites de crédito."
        elif score >= 550:
            decision = "ALERTA / ANÁLISE MANUAL"
            decision_detail = "Risco moderado. Recomendada análise manual de comprovante de renda."
        else:
            decision = "NEGADO"
            decision_detail = "Proposta excede a política de risco e limite de Perda Esperada (EL)."
            
        # 6. SHAP / Explicabilidade (XAI)
        shap_factors = []
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X_proc)[0]
                # Criar pares de (feature, shap_value) ordenados pelo impacto absoluto
                feat_impact = list(zip(self.feature_names, shap_vals))
                feat_impact.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for fname, val in feat_impact[:5]:
                    impact_type = "Aumenta o Risco" if val > 0 else "Reduz o Risco"
                    shap_factors.append({
                        'feature': fname,
                        'shap_value': round(float(val), 4),
                        'impact': impact_type
                    })
            except Exception as e:
                shap_factors = [{'error': str(e)}]
                
        return {
            'credit_score': score,
            'decision': decision,
            'decision_detail': decision_detail,
            'metrics': {
                'pd_probability_of_default': round(pd_val, 4),
                'lgd_loss_given_default': round(lgd_val, 4),
                'ead_exposure_at_default': round(ead_val, 2),
                'expected_loss': round(expected_loss, 2)
            },
            'top_risk_factors': shap_factors
        }

if __name__ == '__main__':
    engine = CreditRiskEngine()
    
    sample = {
        'Age': 24,
        'Sex': 'female',
        'Job': 2,
        'Housing': 'rent',
        'Saving accounts': 'little',
        'Checking account': 'little',
        'Credit amount': 6000,
        'Duration': 36,
        'Purpose': 'car'
    }
    
    res = engine.evaluate_applicant(sample)
    print("="*60)
    print("AVALIAÇÃO QUANTITATIVA DE RISCO DE CRÉDITO")
    print("="*60)
    print(f"Credit Score: {res['credit_score']}")
    print(f"Decisão: {res['decision']} - {res['decision_detail']}")
    print(f"PD (Prob. Default): {res['metrics']['pd_probability_of_default']:.2%}")
    print(f"LGD (Taxa Perda): {res['metrics']['lgd_loss_given_default']:.2%}")
    print(f"EAD (Exposição): R$ {res['metrics']['ead_exposure_at_default']:,.2f}")
    print(f"Expected Loss (EL): R$ {res['metrics']['expected_loss']:,.2f}")
    print("\nFatores Principais (SHAP XAI):")
    for f in res['top_risk_factors']:
        print(f" - {f['feature']}: {f['shap_value']} ({f['impact']})")
