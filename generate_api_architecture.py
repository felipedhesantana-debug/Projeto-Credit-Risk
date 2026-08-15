"""
Script para gerar diagrama de API endpoints
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Título
ax.text(5, 9.5, 'API REST - Arquitetura de Endpoints', fontsize=16, color='white', 
       weight='bold', ha='center')
ax.text(5, 9.0, 'FastAPI com Documentação Automática (Swagger)', fontsize=11, color='#8b949e', 
       ha='center', style='italic')

# Cliente/Frontend
ax.add_patch(FancyBboxPatch((0.3, 7.5), 2, 1, boxstyle="round,pad=0.1",
                           facecolor='#238636', edgecolor='#2ea043', linewidth=2))
ax.text(1.3, 8, 'Cliente/Frontend', fontsize=10, color='white', weight='bold', ha='center', va='center')

# Seta
arrow1 = FancyArrowPatch((2.5, 8), (3.5, 8), arrowstyle='->', mutation_scale=20, 
                        color='#79c0ff', linewidth=2)
ax.add_patch(arrow1)
ax.text(3, 8.3, 'HTTP Request', fontsize=9, color='#79c0ff', ha='center')

# API Gateway
ax.add_patch(FancyBboxPatch((3.5, 7.3), 3, 1.4, boxstyle="round,pad=0.1",
                           facecolor='#0ea5e9', edgecolor='#0ea5e9', linewidth=2))
ax.text(5, 8.1, 'FastAPI Server', fontsize=11, color='white', weight='bold', ha='center')
ax.text(5, 7.6, 'http://localhost:8000', fontsize=9, color='white', ha='center')

# Endpoints
endpoints = [
    {
        'title': '📊 POST /predict',
        'desc': 'Avalia proposta\nde crédito',
        'color': '#0ea5e9',
        'x': 0.3, 'y': 5.5
    },
    {
        'title': '📈 GET /metrics',
        'desc': 'Retorna métricas\ndo modelo',
        'color': '#22c55e',
        'x': 3.3, 'y': 5.5
    },
    {
        'title': '🛡️ POST /psi',
        'desc': 'Calcula PSI e\nmonitoramento',
        'color': '#f97316',
        'x': 6.3, 'y': 5.5
    },
    {
        'title': '📖 GET /docs',
        'desc': 'Swagger UI\nDocumentação',
        'color': '#8b5cf6',
        'x': 9.2, 'y': 5.5
    }
]

for ep in endpoints:
    # Box do endpoint
    ax.add_patch(FancyBboxPatch((ep['x'], ep['y']), 2.8, 1.8, boxstyle="round,pad=0.08",
                               facecolor=ep['color'], edgecolor=ep['color'], linewidth=1.5, alpha=0.3))
    
    # Título
    ax.text(ep['x'] + 1.4, ep['y'] + 1.45, ep['title'], fontsize=10, color='white', 
           weight='bold', ha='center')
    
    # Descrição
    ax.text(ep['x'] + 1.4, ep['y'] + 0.6, ep['desc'], fontsize=8, color='#8b949e', 
           ha='center', va='center')
    
    # Seta apontando para API
    if ep['x'] < 5:
        arrow_x = ep['x'] + 1.4
    else:
        arrow_x = ep['x'] + 1.4
    
    arrow = FancyArrowPatch((arrow_x, ep['y'] + 1.8), (5, 8.7),
                           arrowstyle='->', mutation_scale=15, color='#30363d', 
                           linewidth=1, linestyle='--', alpha=0.5)
    ax.add_patch(arrow)

# Request/Response Models
ax.text(5, 4.5, 'Modelos de Requisição e Resposta', fontsize=11, color='white', weight='bold', ha='center')

# Request box
ax.add_patch(FancyBboxPatch((0.3, 1.5), 4.5, 2.7, boxstyle="round,pad=0.1",
                           facecolor='#161b22', edgecolor='#30363d', linewidth=1.5))
ax.text(2.55, 4.0, '📥 Request Body (POST /predict)', fontsize=10, color='#79c0ff', weight='bold', ha='center')

request_text = """{
  "age": 30,
  "sex": "male",
  "credit_amount": 5000,
  "duration": 40,
  "housing": "own"
}"""

ax.text(0.5, 2.8, request_text, fontsize=7.5, color='#c9d1d9', ha='left', va='top',
       family='monospace', bbox=dict(boxstyle='round', facecolor='#0d1117', edgecolor='#30363d', pad=0.3))

# Response box
ax.add_patch(FancyBboxPatch((5.2, 1.5), 4.5, 2.7, boxstyle="round,pad=0.1",
                           facecolor='#161b22', edgecolor='#30363d', linewidth=1.5))
ax.text(7.45, 4.0, '📤 Response (200 OK)', fontsize=10, color='#79c0ff', weight='bold', ha='center')

response_text = """{
  "score": 600,
  "decision": "ALERTA",
  "probability": 0.496,
  "shap_values": {...},
  "timestamp": "2026-08-15"
}"""

ax.text(5.4, 2.8, response_text, fontsize=7.5, color='#c9d1d9', ha='left', va='top',
       family='monospace', bbox=dict(boxstyle='round', facecolor='#0d1117', edgecolor='#30363d', pad=0.3))

# Status codes legenda
ax.text(5, 0.9, '✓ 200 OK  |  ⚠️  400 Bad Request  |  ⚠️  422 Validation Error  |  ❌ 500 Server Error', 
       fontsize=9, color='#8b949e', ha='center')

plt.tight_layout()
plt.savefig('plots/api_architecture.png', dpi=150, facecolor='#0e1117', edgecolor='none', bbox_inches='tight')
print("✅ Arquitetura de API salva em: plots/api_architecture.png")
plt.close()
