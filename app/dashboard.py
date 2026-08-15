import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Adiciona diretórios src e app ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_engine import CreditRiskEngine
from monitoring import calculate_psi

# Configuração da Página do Streamlit
st.set_page_config(
    page_title="Credit Risk Analytics & Decisioning",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def get_risk_engine():
    try:
        return CreditRiskEngine(model_path='models/credit_risk_model.joblib')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de risco: {e}")
        return None

engine = get_risk_engine()

# Título Principal
st.title("🏦 Plataforma Empresarial de Credit Risk & Risk Analytics")
st.markdown("##### Avaliação de Crédito Quantitativa, Explicabilidade SHAP (XAI) e Monitoramento de Drift (PSI)")

# Tabs da Aplicação
tab1, tab2, tab3 = st.tabs([
    "📋 Simulador de Crédito e Decisão",
    "📊 Visão Geral de Risco e Desempenho do Modelo",
    "🛡️ Monitoramento de Deriva da População (PSI)"
])

# ==============================================================================
# TAB 1: SIMULADOR DE PROPOSTA DE CRÉDITO & DECISIONING
# ==============================================================================
with tab1:
    st.subheader("Simulação em Tempo Real de Proposta de Empréstimo")
    
    col_input, col_output = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("#### Dados do Solicitante")
        with st.form("credit_application_form"):
            age = st.slider("Idade do Cliente (anos)", 18, 80, 32)
            sex = st.selectbox("Gênero", ["male", "female"], format_func=lambda x: "Masculino" if x=="male" else "Feminino")
            job = st.selectbox("Nível de Qualificação do Emprego", [0, 1, 2, 3], index=2, 
                             format_func=lambda x: {0: "0 - Não qualificado (residente)", 1: "1 - Não qualificado (não residente)", 2: "2 - Qualificado / Funcionário", 3: "3 - Altamente qualificado / Executivo"}[x])
            housing = st.selectbox("Tipo de Moradia", ["own", "rent", "free"], 
                                 format_func=lambda x: {"own": "Própria", "rent": "Alugada", "free": "Cedida / Gratuita"}[x])
            saving_acc = st.selectbox("Saldo da Conta Poupança", ["little", "moderate", "quite rich", "rich", None], index=0,
                                    format_func=lambda x: {"little": "Pouco (< €100)", "moderate": "Moderado (€100 - €500)", "quite rich": "Alto (€500 - €1000)", "rich": "Muito Alto (> €1000)", None: "Sem informação / Desconhecido"}[x])
            checking_acc = st.selectbox("Saldo da Conta Corrente", ["little", "moderate", "rich", None], index=1,
                                      format_func=lambda x: {"little": "Pouco (< €0)", "moderate": "Moderado (€0 - €200)", "rich": "Alto (> €200)", None: "Sem informação / Desconhecido"}[x])
            credit_amount = st.number_input("Valor Solicitado (€)", min_value=250.0, max_value=20000.0, value=3500.0, step=250.0)
            duration = st.slider("Prazo de Pagamento (Meses)", 4, 72, 24)
            purpose = st.selectbox("Finalidade do Empréstimo", ["car", "radio/TV", "furniture/equipment", "business", "education", "repairs", "domestic appliances", "vacation/others"],
                                 format_func=lambda x: {"car": "Veículo", "radio/TV": "Eletrônicos / TV", "furniture/equipment": "Móveis / Equipamentos", "business": "Negócios", "education": "Educação", "repairs": "Reformas", "domestic appliances": "Eletrodomésticos", "vacation/others": "Férias / Outros"}[x])
            
            submit_btn = st.form_submit_button("🚀 Avaliar Proposta de Crédito")

    with col_output:
        st.markdown("#### Resultado da Análise de Risco")
        
        if submit_btn or 'last_eval' in st.session_state:
            if submit_btn:
                applicant_data = {
                    'Age': age, 'Sex': sex, 'Job': job, 'Housing': housing,
                    'Saving accounts': saving_acc, 'Checking account': checking_acc,
                    'Credit amount': credit_amount, 'Duration': duration, 'Purpose': purpose
                }
                res = engine.evaluate_applicant(applicant_data)
                st.session_state['last_eval'] = res
            else:
                res = st.session_state['last_eval']
                
            score = res['credit_score']
            decision = res['decision']
            metrics = res['metrics']
            
            # Badge de Decisão em container isolado para evitar erros de renderização no React DOM
            with st.container(border=True):
                if decision == "APROVADO":
                    st.success(f"**Decisão: {decision}** — {res['decision_detail']}")
                elif decision == "ALERTA / ANÁLISE MANUAL":
                    st.warning(f"**Decisão: {decision}** — {res['decision_detail']}")
                else:
                    st.error(f"**Decisão: {decision}** — {res['decision_detail']}")
                
            # Scorecard Metric
            st.metric(label="Pontuação de Crédito (Escala 300 - 850)", value=score, delta=f"{score - 600} pontos em relação à linha de base")
            st.progress((score - 300) / 550)
            
            # Métricas Financeiras de Risco
            st.markdown("---")
            st.markdown("##### Métricas de Risco de Crédito (Quadro de Basileia)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PD (Probabilidade de Inadimplência)", f"{metrics['pd_probability_of_default']:.1%}")
            m2.metric("LGD (Perda Dada)", f"{metrics['lgd_loss_given_default']:.1%}")
            m3.metric("EAD (Exposição)", f"€ {metrics['ead_exposure_at_default']:,.0f}")
            m4.metric("Perda Esperada (PE)", f"€ {metrics['expected_loss']:,.2f}")
            
            # Explicabilidade SHAP
            st.markdown("---")
            st.markdown("##### Explicabilidade da Decisão (SHAP XAI - Top Drivers)")
            factors = res['top_risk_factors']
            if factors and 'error' not in factors[0]:
                df_shap = pd.DataFrame(factors)
                fig, ax = plt.subplots(figsize=(7, 3))
                colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in df_shap['shap_value']]
                ax.barh(df_shap['feature'], df_shap['shap_value'], color=colors)
                ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
                ax.set_xlabel("Impacto no Risco (SHAP Value)")
                ax.set_title("Fatores que Aumentam (Vermelho) ou Reduzem (Verde) o Risco")
                st.pyplot(fig)
            else:
                st.info("Explicação SHAP não disponível no momento.")
        else:
            st.info("Preencha o formulário ao lado e clique em **Avaliar Proposta de Crédito** para visualizar os resultados.")

# ==============================================================================
# TAB 2: VISÃO GERAL DE RISCO & MODEL PERFORMANCE
# ==============================================================================
with tab2:
    st.subheader("Desempenho dos Modelos Preditivos e Análise de Risco")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### Matriz de Confusão e Curvas ROC")
        if os.path.exists('plots/model_evaluation.png'):
            st.image('plots/model_evaluation.png', caption="Avaliação Preditiva - Curvas ROC e Matriz de Confusão")
        else:
            st.warning("Gráfico de avaliação não encontrado. Execute o treinamento.")
            
    with col_m2:
        st.markdown("#### Importância Global das Variáveis (Feature Importance)")
        if os.path.exists('plots/feature_importance.png'):
            st.image('plots/feature_importance.png', caption="Top Features Globais no Modelo XGBoost")
        else:
            st.warning("Gráfico de importância das variáveis não encontrado.")
            
    st.markdown("---")
    st.markdown("#### Análise Exploratória do Dataset (EDA)")
    if os.path.exists('plots/eda_overview.png'):
        st.image('plots/eda_overview.png', caption="Distribuições de Idade, Crédito, Moradia e Propósito vs Risco")

# ==============================================================================
# TAB 3: MONITORAMENTO DE DRIFT DA POPULAÇÃO (PSI)
# ==============================================================================
with tab3:
    st.subheader("Monitoramento de Population Stability Index (PSI)")
    st.markdown("""
    O **Population Stability Index (PSI)** mede o deslocamento entre a distribuição dos scores de crédito da safra de treinamento 
    e novas safras de propostas recebidas.
    
    - **PSI < 0,10:** Estável (Sem Drift) 🟢
    - **0,10 ≤ PSI ≤ 0,25:** Variação Moderada (Atenção) 🟡
    - **PSI > 0,25:** Drift Crítico (Re-treinamento Necessário) 🔴
    """)
    
    col_psi1, col_psi2 = st.columns([1, 1.2])
    
    with col_psi1:
        st.markdown("#### Simular Nova Safra de Clientes")
        drift_scenario = st.radio(
            "Cenário de Teste de Safra:",
            ["Safra Normal (Perfil Semelhante)", "Safra Com Deterioração de Crédito"],
            index=0
        )
        
        if st.button("🔄 Simular e Calcular PSI"):
            np.random.seed(123)
            # Scores da base de referência
            df_ref = pd.read_csv('german_credit_data.csv').fillna('unknown')
            y_proba_ref = engine.model.predict_proba(engine.preprocessor.transform(df_ref))[:, 1]
            ref_scores = [engine.calculate_score(p) for p in y_proba_ref]
            
            if "Normal" in drift_scenario:
                # Safra estável (ligeira variação aleatória)
                actual_scores = np.random.normal(loc=np.mean(ref_scores), scale=np.std(ref_scores), size=400)
            else:
                # Safra com perfil deteriorado (scores menores)
                actual_scores = np.random.normal(loc=np.mean(ref_scores) - 85, scale=np.std(ref_scores) * 1.1, size=400)
                
            psi_res = calculate_psi(ref_scores, actual_scores)
            st.session_state['psi_result'] = psi_res
            st.session_state['ref_scores'] = ref_scores
            st.session_state['actual_scores'] = actual_scores

    with col_psi2:
        if 'psi_result' in st.session_state:
            psi_res = st.session_state['psi_result']
            val = psi_res['psi_value']
            status_txt = psi_res['status']
            
            st.markdown("#### Resultado da Análise de PSI")
            with st.container(border=True):
                if val < 0.10:
                    st.success(f"**Valor do PSI:** {val:.4f} | **Status:** {status_txt}")
                elif val <= 0.25:
                    st.warning(f"**Valor do PSI:** {val:.4f} | **Status:** {status_txt}")
                else:
                    st.error(f"**Valor do PSI:** {val:.4f} | **Status:** {status_txt}")
                
            st.markdown("##### Tabela por Bucket de Decil")
            st.dataframe(psi_res['psi_table'])
            
            # Gráfico de comparação de distribuição
            fig_psi, ax_psi = plt.subplots(figsize=(6, 3))
            ax_psi.hist(st.session_state['ref_scores'], bins=15, alpha=0.5, label='Treino (Esperado)', color='blue', density=True)
            ax_psi.hist(st.session_state['actual_scores'], bins=15, alpha=0.5, label='Nova Safra (Atual)', color='orange', density=True)
            ax_psi.set_xlabel("Credit Score")
            ax_psi.set_ylabel("Densidade")
            ax_psi.legend()
            ax_psi.set_title("Comparação de Distribuição de Scores")
            st.pyplot(fig_psi)
        else:
            st.info("Clique em **Simular e Calcular PSI** para executar o teste de estabilidade.")
