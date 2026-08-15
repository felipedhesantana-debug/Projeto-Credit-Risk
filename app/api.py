import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_engine import CreditRiskEngine
from monitoring import calculate_psi

app = FastAPI(
    title="Credit Risk & Quantitative Analytics API",
    description="API para Avaliação de Risco de Crédito, Cálculo de PD/LGD/EAD/EL, Scorecard e Explicabilidade SHAP.",
    version="2.0.0"
)

# Inicializa o motor de risco
try:
    risk_engine = CreditRiskEngine()
except Exception as e:
    risk_engine = None
    print(f"Aviso: Não foi possível carregar o motor de risco: {e}")

class CreditApplicantSchema(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Idade do proponente em anos", example=35)
    Sex: str = Field(..., description="Gênero: 'male' ou 'female'", example="male")
    Job: int = Field(..., ge=0, le=3, description="Nível de emprego/qualificação (0 a 3)", example=2)
    Housing: str = Field(..., description="Moradia: 'own', 'rent', ou 'free'", example="own")
    Saving_accounts: Optional[str] = Field(None, alias="Saving accounts", description="Conta poupança: 'little', 'moderate', 'quite rich', 'rich' ou None", example="little")
    Checking_account: Optional[str] = Field(None, alias="Checking account", description="Conta corrente: 'little', 'moderate', 'rich' ou None", example="moderate")
    Credit_amount: float = Field(..., alias="Credit amount", gt=0, description="Valor do crédito em Euros/Reais", example=4500.0)
    Duration: int = Field(..., gt=0, le=72, description="Duração do contrato em meses", example=24)
    Purpose: str = Field(..., description="Finalidade: 'car', 'radio/TV', 'furniture/equipment', 'business', 'education', 'repairs', etc.", example="car")

    model_config = ConfigDict(populate_by_name=True)

class PSICheckSchema(BaseModel):
    recent_scores: List[float] = Field(..., description="Lista de scores de crédito da nova safra para avaliar drift")

@app.get("/")
def read_root():
    return {
        "service": "Credit Risk & Quantitative Analytics API",
        "version": "2.0.0",
        "status": "Online",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    if risk_engine is None:
        return {"status": "Unhealthy", "error": "Motor de risco não carregado."}
    return {
        "status": "Healthy",
        "model_loaded": risk_engine.artifact['model_name'],
        "features_count": len(risk_engine.feature_names)
    }

@app.post("/api/v1/credit-risk/evaluate", status_code=status.HTTP_200_OK)
def evaluate_credit_application(applicant: CreditApplicantSchema):
    """
    Avalia a proposta de crédito de um cliente e calcula:
    - Credit Score (300 a 850)
    - PD (Probability of Default)
    - LGD (Loss Given Default)
    - EAD (Exposure at Default)
    - Expected Loss (EL = PD * LGD * EAD)
    - SHAP Top Risk Factors (Explicabilidade / XAI)
    """
    if risk_engine is None:
        raise HTTPException(status_code=500, detail="Motor de risco de crédito indisponível.")
    
    applicant_dict = applicant.model_dump(by_alias=True)
    
    try:
        result = risk_engine.evaluate_applicant(applicant_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar avaliação: {str(e)}")

@app.post("/api/v1/monitoring/psi", status_code=status.HTTP_200_OK)
def check_population_stability(payload: PSICheckSchema):
    """
    Calcula o Population Stability Index (PSI) entre a safra de treino do modelo e a nova safra.
    """
    if risk_engine is None:
        raise HTTPException(status_code=500, detail="Motor de risco indisponível.")
        
    try:
        # Carrega dataset base para obter distribuição de scores de referência
        raw_df = risk_engine.preprocessor.transform(risk_engine.artifact.get('raw_df', pd.DataFrame())) if 'raw_df' in risk_engine.artifact else None
        
        # Como referência rápida, geramos scores sintéticos da base de treino
        y_proba_train = risk_engine.model.predict_proba(risk_engine.preprocessor.transform(pd.read_csv('german_credit_data.csv').fillna('unknown')))[:, 1]
        ref_scores = [risk_engine.calculate_score(p) for p in y_proba_train]
        
        psi_result = calculate_psi(ref_scores, payload.recent_scores)
        
        return {
            'psi_value': psi_result['psi_value'],
            'status': psi_result['status'],
            'color': psi_result['color'],
            'buckets_detail': psi_result['psi_table'].to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao calcular PSI: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
