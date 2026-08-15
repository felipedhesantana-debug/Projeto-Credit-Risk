"""
Script para gerar exemplo de relatório de conformidade
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import datetime

fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('#0e1117')

# Criar layout tipo documento
ax = fig.add_subplot(111)
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Cabeçalho
ax.add_patch(Rectangle((0, 9), 10, 0.8, facecolor='#161b22', edgecolor='#30363d', linewidth=2))
ax.text(5, 9.4, '📋 Relatório de Conformidade Regulatória', fontsize=14, color='white', 
       weight='bold', ha='center')

# Metadados
metadata = f"Data: {datetime.datetime.now().strftime('%d/%m/%Y')}  |  Modelo: XGBoost v2.1  |  Status: ✓ Aprovado"
ax.text(5, 8.7, metadata, fontsize=9, color='#8b949e', ha='center')

# Seção 1: Sumário Executivo
y_pos = 8.3
ax.text(0.3, y_pos, '1. SUMÁRIO EXECUTIVO', fontsize=11, color='#79c0ff', weight='bold')
ax.add_patch(Rectangle((0.3, y_pos - 0.05), 9.4, 0.02, facecolor='#30363d'))

summary = """✓ Modelo em produção desde: 2024-06-15
✓ Número de decisões: 15.847 avaliações
✓ Taxa de aprovação: 68.2%
✓ Tempo médio de decisão: 245ms
✓ Status de conformidade: VERDE (Todas as regulações atendidas)"""

ax.text(0.4, y_pos - 0.5, summary, fontsize=8.5, color='#c9d1d9', ha='left', va='top',
       bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', pad=0.4))

# Seção 2: Métricas de Desempenho
y_pos = 6.3
ax.text(0.3, y_pos, '2. MÉTRICAS DE DESEMPENHO', fontsize=11, color='#79c0ff', weight='bold')
ax.add_patch(Rectangle((0.3, y_pos - 0.05), 9.4, 0.02, facecolor='#30363d'))

metrics_box = """
Acurácia Geral: 84.2%  |  Sensibilidade (Recall): 71.67%  |  Especificidade: 93.3%  |  ROC-AUC: 0.7850

Discriminação por Grupo Protegido (Teste de Paridade):
├─ Gênero (M vs F):       Diferença = 2.1%  ✓ Aceitável (<5%)
├─ Idade (Jovem vs Senior): Diferença = 3.8%  ✓ Aceitável (<5%)
└─ Status Laboral:        Diferença = 1.5%  ✓ Aceitável (<5%)
"""

ax.text(0.4, y_pos - 0.4, metrics_box, fontsize=7.5, color='#c9d1d9', ha='left', va='top',
       family='monospace', bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', pad=0.4))

# Seção 3: Conformidade Regulatória
y_pos = 4.2
ax.text(0.3, y_pos, '3. CONFORMIDADE REGULATÓRIA', fontsize=11, color='#79c0ff', weight='bold')
ax.add_patch(Rectangle((0.3, y_pos - 0.05), 9.4, 0.02, facecolor='#30363d'))

compliance = """
✓ LGPD (Lei Geral de Proteção de Dados)
  ├─ Anonimização de PII: IMPLEMENTADA
  ├─ Direito de Explicabilidade: SHAP XAI integrado
  └─ Auditoria de Retenção: Logs por 12 meses

✓ Basileia III (Regulação Bancária)
  ├─ PD (Probabilidade de Inadimplência): Calibrada
  ├─ LGD (Perda Dada): 35% (baseline)
  └─ EAD (Exposição): Dinâmica por operação

✓ Fair Lending (EEOC - EUA)
  ├─ Impacto Disparatado: Testado
  ├─ Efeito Disparatado: Mitigado < 80%
  └─ Monitoramento Contínuo: Mensal
"""

ax.text(0.4, y_pos - 0.35, compliance, fontsize=7, color='#c9d1d9', ha='left', va='top',
       family='monospace', bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', pad=0.4))

# Seção 4: Monitoramento e Alertas
y_pos = 1.0
ax.text(0.3, y_pos, '4. SISTEMA DE MONITORAMENTO', fontsize=11, color='#79c0ff', weight='bold')
ax.add_patch(Rectangle((0.3, y_pos - 0.05), 9.4, 0.02, facecolor='#30363d'))

monitoring = """Status: ✓ ESTÁVEL  |  PSI: 0.034 (Drift Não Detectado)  |  Próxima Auditoria: 2026-09-15"""

ax.text(0.4, y_pos - 0.3, monitoring, fontsize=8, color='#22c55e', ha='left',
       bbox=dict(boxstyle='round', facecolor='#164b35', edgecolor='#23d366', linewidth=1, pad=0.3))

plt.tight_layout()
plt.savefig('plots/compliance_report.png', dpi=150, facecolor='#0e1117', edgecolor='none', bbox_inches='tight')
print("✅ Relatório de conformidade salvo em: plots/compliance_report.png")
plt.close()
