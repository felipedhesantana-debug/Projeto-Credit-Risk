import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs('plots', exist_ok=True)

# 1. Load Dataset
data_path = 'german_credit_data.csv'
df = pd.read_csv(data_path)

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

print("="*50)
print("1. RESUMO DO CONJUNTO DE DADOS DE RISCO DE CRÉDITO")
print("="*50)
print(f"Formato do dataset: {df.shape[0]} linhas, {df.shape[1]} colunas\n")

print("--- Primeiras 5 linhas ---")
print(df.head())

print("\n--- Informações do Dataset ---")
print(df.info())

print("\n--- Valores Ausentes por Coluna ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Distribuição da Variável Alvo (Risk) ---")
risk_counts = df['Risk'].value_counts()
print(risk_counts)
print(f"Proporção de Bons Pagadores: {risk_counts.get('good', 0) / len(df):.2%}")
print(f"Proporção de Maus Pagadores: {risk_counts.get('bad', 0) / len(df):.2%}\n")

# 2. Visualizações
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot 1: Distribuição de Risco
counts = df['Risk'].value_counts()
axes[0, 0].bar(counts.index, counts.values, color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Distribuição de Risco (Good vs Bad)')
axes[0, 0].set_ylabel('Quantidade')
for i, v in enumerate(counts.values):
    axes[0, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Plot 2: Idade vs Valor do Crédito por Risco
good_df = df[df['Risk'] == 'good']
bad_df = df[df['Risk'] == 'bad']
axes[0, 1].scatter(good_df['Age'], good_df['Credit amount'], alpha=0.5, label='Good', color='#2ecc71')
axes[0, 1].scatter(bad_df['Age'], bad_df['Credit amount'], alpha=0.6, label='Bad', color='#e74c3c')
axes[0, 1].set_title('Idade vs. Valor do Crédito')
axes[0, 1].set_xlabel('Idade (anos)')
axes[0, 1].set_ylabel('Valor do Crédito')
axes[0, 1].legend()

# Plot 3: Propósito do Empréstimo vs Risco
purpose_risk = pd.crosstab(df['Purpose'], df['Risk'])
purpose_risk.plot(kind='bar', stacked=True, ax=axes[1, 0], color=['#e74c3c', '#2ecc71'])
axes[1, 0].set_title('Propósito do Empréstimo por Risco')
axes[1, 0].set_xlabel('Propósito')
axes[1, 0].set_ylabel('Quantidade')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Moradia (Housing) vs Risco
housing_risk = pd.crosstab(df['Housing'], df['Risk'])
housing_risk.plot(kind='bar', ax=axes[1, 1], color=['#e74c3c', '#2ecc71'])
axes[1, 1].set_title('Tipo de Moradia por Risco')
axes[1, 1].set_xlabel('Moradia')
axes[1, 1].set_ylabel('Quantidade')

plt.suptitle('Análise Exploratória - German Credit Risk Dataset', fontsize=16, fontweight='bold')
plt.savefig('plots/eda_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print("--> Gráfico de EDA salvo com sucesso em 'plots/eda_overview.png'\n")
