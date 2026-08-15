"""
Script para gerar uma captura visual da tela de PSI (Population Stability Index)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Criar figura com tema escuro
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#0e1117')

# Criar grid para organizar os elementos
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.93, bottom=0.08)

# ==============================================================================
# TÍTULO
# ==============================================================================
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(0.5, 0.6, '🛡️ Monitoramento de Deriva da População (PSI)', 
             fontsize=18, color='white', weight='bold', ha='center', transform=ax_title.transAxes)
ax_title.text(0.5, 0.1, 'Acompanhamento da estabilidade do modelo em produção', 
             fontsize=12, color='#8b949e', ha='center', transform=ax_title.transAxes)

# ==============================================================================
# RESULTADO DO PSI (com status)
# ==============================================================================
ax_result = fig.add_subplot(gs[1, 0])
ax_result.set_facecolor('#0d1117')
ax_result.set_xlim(0, 10)
ax_result.set_ylim(0, 10)
ax_result.axis('off')

ax_result.text(0.5, 9, 'Resultado da Análise de PSI', fontsize=12, color='white', weight='bold')

# PSI status - ESTÁVEL (verde)
ax_result.add_patch(FancyBboxPatch((0.3, 6.5), 9.4, 2, boxstyle="round,pad=0.1",
                                  facecolor='#164b35', edgecolor='#23d366', linewidth=2))
ax_result.text(5, 8.2, '✓ ESTÁVEL', fontsize=14, color='#23d366', weight='bold', ha='center')
ax_result.text(5, 7.2, 'Valor do PSI: 0.0345', fontsize=12, color='#23d366', ha='center')
ax_result.text(5, 6.7, 'Distribuição do modelo mantém-se consistente', fontsize=10, color='#79c0ff', ha='center')

# Tabela PSI por Decil
ax_result.text(0.5, 5.8, 'Tabela por Bucket de Decil', fontsize=11, color='#c9d1d9', weight='bold')

table_data = [
    ['Decil', 'Treino %', 'Atual %', 'PSI Parcial'],
    ['0-10%', '10.2%', '9.8%', '0.004'],
    ['10-20%', '10.1%', '10.3%', '0.002'],
    ['20-30%', '9.9%', '10.0%', '0.001'],
    ['30-40%', '10.0%', '10.1%', '0.001'],
    ['40-50%', '10.2%', '10.0%', '0.002'],
]

y_pos = 5.2
for i, row in enumerate(table_data):
    if i == 0:
        # Header
        for j, cell in enumerate(row):
            x_pos = 0.5 + j * 2.3
            ax_result.text(x_pos, y_pos, cell, fontsize=9, color='#79c0ff', weight='bold')
        ax_result.plot([0.3, 9.7], [5.0, 5.0], color='#30363d', linewidth=1)
    else:
        # Data
        for j, cell in enumerate(row):
            x_pos = 0.5 + j * 2.3
            ax_result.text(x_pos, y_pos - i*0.35, cell, fontsize=8, color='#c9d1d9')
    y_pos -= 0.4

# ==============================================================================
# GRÁFICO DE DISTRIBUIÇÃO
# ==============================================================================
ax_hist = fig.add_subplot(gs[1:, 1])
ax_hist.set_facecolor('#0d1117')

# Simular dados de distribuição
np.random.seed(42)
treino = np.random.normal(650, 80, 1000)  # Score de treino
atual = np.random.normal(648, 82, 1000)   # Score atual (ligeiramente diferente)

# Histogram
bins = np.linspace(400, 850, 20)
ax_hist.hist(treino, bins=bins, alpha=0.6, label='Treino (Esperado)', color='#0ea5e9', density=True)
ax_hist.hist(atual, bins=bins, alpha=0.6, label='Nova Safra (Atual)', color='#f97316', density=True)

ax_hist.set_xlabel('Credit Score', fontsize=11, color='#c9d1d9')
ax_hist.set_ylabel('Densidade', fontsize=11, color='#c9d1d9')
ax_hist.set_title('Comparação de Distribuição de Scores', fontsize=12, color='white', weight='bold', pad=15)
ax_hist.legend(fontsize=10, loc='upper right', facecolor='#161b22', edgecolor='#30363d')
ax_hist.tick_params(colors='#8b949e')
ax_hist.spines['bottom'].set_color('#30363d')
ax_hist.spines['left'].set_color('#30363d')
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.grid(True, alpha=0.1, color='#30363d')

# ==============================================================================
# INTERPRETAÇÃO E AÇÕES
# ==============================================================================
ax_info = fig.add_subplot(gs[2, 0])
ax_info.set_facecolor('#0d1117')
ax_info.set_xlim(0, 10)
ax_info.set_ylim(0, 10)
ax_info.axis('off')

ax_info.text(0.5, 9.2, 'O que significa cada status?', fontsize=11, color='white', weight='bold')

statuses = [
    ('✓ ESTÁVEL', '🟢 PSI < 0.10', 'Modelo continua confiável. Nenhuma ação necessária.', 7.5),
    ('⚠ ATENÇÃO', '🟡 0.10 ≤ PSI ≤ 0.25', 'Drift detectado. Monitorar de perto, considerar retraining.', 5.5),
    ('✗ CRÍTICO', '🔴 PSI > 0.25', 'Drift severo. Retraining urgente recomendado.', 3.5),
]

for label, range_text, desc, y in statuses:
    ax_info.text(0.5, y, f'{label} — {range_text}', fontsize=9, color='#79c0ff', weight='bold')
    ax_info.text(0.5, y - 0.5, desc, fontsize=8, color='#8b949e')

plt.savefig('plots/psi_monitoring.png', dpi=150, facecolor='#0e1117', edgecolor='none', bbox_inches='tight')
print("✅ PSI monitoring screenshot salvo em: plots/psi_monitoring.png")
plt.close()
