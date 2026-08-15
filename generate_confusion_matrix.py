"""
Script para gerar a matriz de confusão do modelo
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0d1117')

# Dados da matriz de confusão (XGBoost)
tn = 653  # True Negatives (Bons pagadores corretamente identificados)
fp = 47   # False Positives (Bons pagadores incorretamente marcados como maus)
fn = 133  # False Negatives (Maus pagadores não detectados)
tp = 167  # True Positives (Maus pagadores corretamente detectados)

# Criar matriz
cm = np.array([[tn, fp], [fn, tp]])

# Normalizar para percentuais
cm_percent = cm.astype('float') / cm.sum() * 100

# Cores
colors = [
    ['#0ea5e9', '#ef4444'],  # Azul (correto) e Vermelho (erro)
    ['#ef4444', '#22c55e']   # Vermelho (erro) e Verde (correto)
]

# Desenhar células
cell_width = 2
cell_height = 2
x_start = 1
y_start = 1

labels = [['TN\n(Verdadeiros\nNegativos)', 'FP\n(Falsos\nPositivos)'],
          ['FN\n(Falsos\nNegativos)', 'TP\n(Verdadeiros\nPositivos)']]
values = [[tn, fp], [fn, tp]]
percents = [[cm_percent[0, 0], cm_percent[0, 1]], [cm_percent[1, 0], cm_percent[1, 1]]]

for i in range(2):
    for j in range(2):
        x = x_start + j * cell_width
        y = y_start + (1 - i) * cell_height
        
        # Cor de fundo baseada na diagonal (corretos) ou fora da diagonal (erros)
        color = colors[i][j]
        ax.add_patch(Rectangle((x, y), cell_width, cell_height, 
                               facecolor=color, edgecolor='#30363d', linewidth=2, alpha=0.3))
        
        # Texto da célula
        ax.text(x + cell_width/2, y + cell_height*0.65, labels[i][j], 
               fontsize=11, color='#c9d1d9', ha='center', va='center', weight='bold')
        ax.text(x + cell_width/2, y + cell_height*0.35, f'{values[i][j]} ({percents[i][j]:.1f}%)', 
               fontsize=12, color='white', ha='center', va='center', weight='bold')

# Labels dos eixos
ax.text(0.2, 2.5, 'Predito\nMau Pagador', fontsize=11, color='#79c0ff', weight='bold', ha='center', rotation=90)
ax.text(0.2, 0.5, 'Predito\nBom Pagador', fontsize=11, color='#79c0ff', weight='bold', ha='center', rotation=90)

ax.text(1.5, 4.5, 'Real: Bom Pagador', fontsize=11, color='#79c0ff', weight='bold', ha='center')
ax.text(3.5, 4.5, 'Real: Mau Pagador', fontsize=11, color='#79c0ff', weight='bold', ha='center')

# Título
ax.text(2.5, 5.2, 'Matriz de Confusão - XGBoost', fontsize=14, color='white', 
       weight='bold', ha='center')

# Métricas calculadas
accuracy = (tn + tp) / (tn + fp + fn + tp) * 100
precision = tp / (tp + fp) * 100
recall = tp / (tp + fn) * 100
f1 = 2 * (precision * recall) / (precision + recall)

metrics_text = f"""
Acurácia: {accuracy:.2f}%  |  Precisão: {precision:.2f}%  |  Recall (Sensibilidade): {recall:.2f}%  |  F1-Score: {f1:.2f}
"""

ax.text(2.5, -0.3, metrics_text, fontsize=10, color='#79c0ff', 
       ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', pad=0.5))

ax.set_xlim(-0.5, 5)
ax.set_ylim(-1, 5.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150, facecolor='#0e1117', edgecolor='none', bbox_inches='tight')
print("✅ Matriz de confusão salva em: plots/confusion_matrix.png")
plt.close()
