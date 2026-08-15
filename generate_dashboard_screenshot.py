"""
Script para gerar uma captura visual do dashboard para documentação
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import sys
import os

# Configurar o diretório
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from risk_engine import CreditRiskEngine

# Criar figura com tema escuro (como Streamlit padrão)
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0e1117')

# ==============================================================================
# PAINEL ESQUERDO - FORMULÁRIO
# ==============================================================================
ax_left.set_facecolor('#0d1117')
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 12)
ax_left.axis('off')

# Título do painel
ax_left.text(0.5, 11.3, 'Idade do Cliente (anos)', fontsize=11, color='white', weight='bold')
ax_left.text(0.5, 10.7, '21', fontsize=14, color='red', weight='bold')
ax_left.add_patch(Rectangle((0.5, 10.3), 9, 0.3, fill=True, facecolor='#30363d', edgecolor='#30363d'))
ax_left.plot([0.5, 4.5], [10.3, 10.3], color='red', linewidth=3)

# Campos do formulário
fields = [
    ('Gênero', 'Feminino', 9.5),
    ('Nível de Qualificação do Emprego', '0 - Não qualificado (residente)', 8.5),
    ('Tipo de Moradia', 'Alugada', 7.5),
    ('Saldo da Conta Poupança', 'Alto (€500 - €1000)', 6.5),
    ('Saldo da Conta Corrente', 'Sem informação / Desconhecido', 5.5),
    ('Valor Solicitado (€)', '5000,00', 4.5),
    ('Prazo de Pagamento (Meses)', '40', 3.5),
    ('Finalidade do Empréstimo', 'Veículo', 2.5),
]

for i, (label, value, y_pos) in enumerate(fields):
    ax_left.text(0.5, y_pos + 0.3, label, fontsize=10, color='#c9d1d9')
    ax_left.text(0.5, y_pos - 0.2, value, fontsize=9, color='#8b949e')
    ax_left.add_patch(Rectangle((0.5, y_pos - 0.4), 9, 0.5, fill=True, facecolor='#161b22', 
                                 edgecolor='#30363d', linewidth=0.5))

# Botão de avaliação
ax_left.add_patch(FancyBboxPatch((0.5, 0.2), 9, 0.8, boxstyle="round,pad=0.1", 
                                 facecolor='#238636', edgecolor='#238636', linewidth=1))
ax_left.text(5, 0.6, '🔥 Avaliar Proposta de Crédito', fontsize=11, color='white', 
            weight='bold', ha='center', va='center')

# ==============================================================================
# PAINEL DIREITO - RESULTADO
# ==============================================================================
ax_right.set_facecolor('#0d1117')
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 12)
ax_right.axis('off')

# Alerta de decisão
ax_right.add_patch(FancyBboxPatch((0.3, 10.5), 9.4, 1, boxstyle="round,pad=0.05",
                                  facecolor='#6d5d3f', edgecolor='#d4a574', linewidth=2))
ax_right.text(5, 11, 'Decisão: ALERTA / ANÁLISE MANUAL — Risco moderado. Recomendada análise manual de comprovante de renda.', 
             fontsize=10, color='#f0e68c', ha='center', va='center', weight='bold', wrap=True)

# Score de crédito
ax_right.text(0.5, 9.8, 'Pontuação de Crédito (Escala 300 - 850)', fontsize=11, color='white', weight='bold')
ax_right.text(0.5, 9.2, '600', fontsize=24, color='#00e676', weight='bold')
ax_right.text(0.5, 8.6, '↑ 0 pontos em relação à linha de base', fontsize=9, color='#00e676')

# Barra de score
score_pct = (600 - 300) / (850 - 300) * 9.2
ax_right.add_patch(Rectangle((0.5, 8.2), 9.2, 0.2, fill=True, facecolor='#30363d'))
ax_right.add_patch(Rectangle((0.5, 8.2), score_pct, 0.2, fill=True, facecolor='#0ea5e9'))

# Métricas de Risco
ax_right.text(0.5, 7.7, 'Métricas de Risco de Crédito (Quadro de Basileia)', fontsize=11, color='white', weight='bold')

metrics = [
    ('PD (Probabilidade de Inadimplência)', '49.6%', 6.5),
    ('LGD (Perda Dada)', '35.0%', 5.5),
    ('EAD (Exposição)', '€ 5,000', 4.5),
    ('Perda Esperada (PE)', '€ 867.51', 3.5),
]

for label, value, y_pos in metrics:
    ax_right.text(0.5, y_pos + 0.3, label, fontsize=9, color='#8b949e')
    ax_right.text(0.5, y_pos - 0.3, value, fontsize=13, color='#ffffff', weight='bold')

# SHAP Explicabilidade
ax_right.text(0.5, 2.7, 'Explicabilidade da Decisão (SHAP XAI - Top Drivers)', fontsize=11, color='white', weight='bold')

# Simulação simples de SHAP bar plot
factors = ['Housing_own', 'Saving_quite_rich', 'Age', 'Duration', 'Checking_unknown']
impacts = [0.35, -0.25, 0.45, 0.60, -0.30]
colors_shap = ['#ef4444' if x > 0 else '#22c55e' for x in impacts]

y_shap = 2.2
for factor, impact, color in zip(factors, impacts, colors_shap):
    x_start = 5 if impact < 0 else 5
    bar_width = abs(impact) * 3.5
    
    if impact > 0:
        ax_right.add_patch(Rectangle((5, y_shap - 0.1), bar_width, 0.2, fill=True, facecolor=color))
        ax_right.text(4.8, y_shap, factor, fontsize=8, color='#c9d1d9', ha='right', va='center')
    else:
        ax_right.add_patch(Rectangle((5 - bar_width, y_shap - 0.1), bar_width, 0.2, fill=True, facecolor=color))
        ax_right.text(5.2, y_shap, factor, fontsize=8, color='#c9d1d9', ha='left', va='center')
    
    y_shap -= 0.35

# Legenda SHAP
ax_right.text(2, 0.3, 'Fatores que Aumentam (Vermelho) ou Reduzem (Verde) o Risco', fontsize=8, color='#8b949e')

plt.tight_layout()
plt.savefig('plots/dashboard_screenshot.png', dpi=150, facecolor='#0e1117', edgecolor='none', bbox_inches='tight')
print("✅ Dashboard screenshot salvo em: plots/dashboard_screenshot.png")
plt.close()
