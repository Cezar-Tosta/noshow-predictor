# noshow-predictor / notebooks / 07_interpretabilidade.py
# Parte 8 — Interpretabilidade com SHAP
# Execute com: python notebooks/07_interpretabilidade.py
# Compatível com Windows e Linux

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CAMINHO_TESTE  = Path('data')  / 'processed' / 'test.csv'
CAMINHO_MODELO = Path('model') / 'model.pkl'
CAMINHO_FEATS  = Path('model') / 'feature_names.pkl'
PASTA_ASSETS   = Path('assets')
PASTA_ASSETS.mkdir(exist_ok=True)

# ── 1. CARREGAR MODELO E DADOS ───────────────────────────────────
print('=' * 60)
print('INTERPRETABILIDADE COM SHAP — PARTE 8')
print('=' * 60)

for caminho in [CAMINHO_TESTE, CAMINHO_MODELO, CAMINHO_FEATS]:
    if not caminho.exists():
        print(f'ERRO: arquivo não encontrado em {caminho}')
        exit(1)

modelo        = joblib.load(CAMINHO_MODELO)
feature_names = joblib.load(CAMINHO_FEATS)
test_df       = pd.read_csv(CAMINHO_TESTE)

# Alinhar colunas do teste com as do modelo
X_teste = test_df.reindex(columns=feature_names, fill_value=0)
y_teste = test_df['noshow']

print(f'\nConjunto de teste: {X_teste.shape[0]:,} linhas × {X_teste.shape[1]} features')

# ── 2. CALCULAR VALORES SHAP ─────────────────────────────────────
print('\n--- Calculando valores SHAP (pode levar alguns minutos) ---')
explainer   = shap.TreeExplainer(modelo)
shap_values = explainer(X_teste)
print('  Cálculo concluído!')
print(f'  Shape dos valores SHAP: {shap_values.values.shape}')

# ── 3. GRÁFICO 1 — SUMMARY PLOT ──────────────────────────────────
print('\n--- Gerando gráfico 1: Summary plot ---')
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_teste,
    max_display=15,
    show=False
)
plt.title(
    'SHAP — Impacto das Features na Previsão de No-Show',
    fontsize=13, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig(
    PASTA_ASSETS / 'shap_01_summary_plot.png',
    bbox_inches='tight', dpi=150
)
plt.close()
print('  Salvo: assets/shap_01_summary_plot.png')

# ── 4. GRÁFICO 2 — BAR PLOT (IMPORTÂNCIA MÉDIA) ──────────────────
print('\n--- Gerando gráfico 2: Bar plot ---')
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_teste,
    plot_type='bar',
    max_display=15,
    show=False
)
plt.title(
    'SHAP — Importância Média das Features',
    fontsize=13, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig(
    PASTA_ASSETS / 'shap_02_bar_plot.png',
    bbox_inches='tight', dpi=150
)
plt.close()
print('  Salvo: assets/shap_02_bar_plot.png')

# ── 5. GRÁFICO 3 — WATERFALL PLOT (PACIENTE INDIVIDUAL) ──────────
print('\n--- Gerando gráfico 3: Waterfall plot ---')

# Selecionar um paciente com alta probabilidade de no-show
y_pred_prob  = modelo.predict_proba(X_teste)[:, 1]
idx_noshow   = np.where(y_teste.values == 1)[0]  # apenas reais no-shows
probs_noshow = y_pred_prob[idx_noshow]
idx_paciente = idx_noshow[np.argmax(probs_noshow)]  # maior probabilidade

prob_paciente = y_pred_prob[idx_paciente]
print(f'  Paciente selecionado: índice {idx_paciente}')
print(f'  Probabilidade de no-show: {prob_paciente:.1%}')
print(f'  No-show real: {"Sim" if y_teste.iloc[idx_paciente] == 1 else "Não"}')

plt.figure(figsize=(10, 7))
shap.waterfall_plot(
    shap_values[idx_paciente],
    max_display=12,
    show=False
)
plt.title(
    f'SHAP — Explicação Individual (Prob. no-show: {prob_paciente:.1%})',
    fontsize=12, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig(
    PASTA_ASSETS / 'shap_03_waterfall_plot.png',
    bbox_inches='tight', dpi=150
)
plt.close()
print('  Salvo: assets/shap_03_waterfall_plot.png')

# ── 6. RANKING DAS FEATURES NO TERMINAL ──────────────────────────
print('\n--- Ranking de importância (SHAP médio absoluto) ---')
importancia_media = np.abs(shap_values.values).mean(axis=0)
ranking = pd.Series(importancia_media, index=feature_names)
ranking = ranking.sort_values(ascending=False).head(15)
print()
for i, (feat, val) in enumerate(ranking.items(), 1):
    print(f'  {i:>2}. {feat:<35} {val:.4f}')

# ── 7. ANÁLISE DO PACIENTE SELECIONADO ───────────────────────────
print('\n--- Análise do paciente selecionado ---')
dados_paciente = X_teste.iloc[idx_paciente]
shap_paciente  = shap_values.values[idx_paciente]
contribuicoes  = pd.Series(shap_paciente, index=feature_names)
contribuicoes  = contribuicoes.reindex(
    contribuicoes.abs().sort_values(ascending=False).index
)
print(f'\n  Top 5 fatores que mais influenciaram a previsão:')
for feat, val in contribuicoes.head(5).items():
    direcao = 'aumentou' if val > 0 else 'reduziu'
    print(f'  {feat}: valor={dados_paciente[feat]:.2f} → {direcao} no-show ({val:+.4f})')

print()
print('=' * 60)
print('Interpretabilidade concluída! Prossiga para a Parte 9.')
print('=' * 60)