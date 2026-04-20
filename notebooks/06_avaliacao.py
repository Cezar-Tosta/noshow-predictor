# noshow-predictor / notebooks / 06_avaliacao.py
# Parte 7 — Avaliação do Modelo
# Execute com: python notebooks/06_avaliacao.py
# Compatível com Windows e Linux

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)

CAMINHO_TESTE  = Path('data')  / 'processed' / 'test.csv'
CAMINHO_MODELO = Path('model') / 'model.pkl'
CAMINHO_FEATS  = Path('model') / 'feature_names.pkl'
PASTA_ASSETS   = Path('assets')
PASTA_ASSETS.mkdir(exist_ok=True)

# ── 1. CARREGAR MODELO E DADOS ───────────────────────────────────
print('=' * 60)
print('AVALIAÇÃO DO MODELO — PARTE 7')
print('=' * 60)

for caminho in [CAMINHO_TESTE, CAMINHO_MODELO, CAMINHO_FEATS]:
    if not caminho.exists():
        print(f'ERRO: arquivo não encontrado em {caminho}')
        exit(1)

modelo        = joblib.load(CAMINHO_MODELO)
feature_names = joblib.load(CAMINHO_FEATS)
test_df       = pd.read_csv(CAMINHO_TESTE)

print(f'\nConjunto de teste: {test_df.shape[0]:,} linhas')

# ── 2. ALINHAR FEATURES DO TESTE COM AS DO MODELO ────────────────
# Garantir que o teste tenha exatamente as mesmas colunas do treino
X_teste = test_df.reindex(columns=feature_names, fill_value=0)
y_teste = test_df['noshow']

# ── 3. FAZER PREVISÕES ────────────────────────────────────────────
print('\n--- Realizando previsões ---')
y_pred      = modelo.predict(X_teste)
y_pred_prob = modelo.predict_proba(X_teste)[:, 1]

# ── 4. CALCULAR MÉTRICAS ──────────────────────────────────────────
print('\n--- Métricas de avaliação ---')
acuracia = accuracy_score(y_teste, y_pred)
precisao = precision_score(y_teste, y_pred)
recall   = recall_score(y_teste, y_pred)
f1       = f1_score(y_teste, y_pred)
auc      = roc_auc_score(y_teste, y_pred_prob)

print(f'  Acurácia  : {acuracia:.4f}  ({acuracia*100:.1f}%)')
print(f'  Precisão  : {precisao:.4f}  ({precisao*100:.1f}%)')
print(f'  Recall    : {recall:.4f}  ({recall*100:.1f}%)')
print(f'  F1-score  : {f1:.4f}  ({f1*100:.1f}%)')
print(f'  AUC-ROC   : {auc:.4f}')

print('\n--- Relatório completo ---')
print(classification_report(
    y_teste, y_pred,
    target_names=['Compareceu (0)', 'Não compareceu (1)']
))

# ── 5. MATRIZ DE CONFUSÃO ─────────────────────────────────────────
print('\n--- Matriz de confusão ---')
cm = confusion_matrix(y_teste, y_pred)
vn, fp, fn, vp = cm.ravel()
print(f'  Verdadeiro Negativo (VN) : {vn:,}')
print(f'  Falso Positivo      (FP) : {fp:,}')
print(f'  Falso Negativo      (FN) : {fn:,}')
print(f'  Verdadeiro Positivo (VP) : {vp:,}')

# ── 6. ANÁLISE CLÍNICA ────────────────────────────────────────────
print('\n--- Análise clínica ---')
total_noshow_real = fn + vp
noshow_detectados = vp
taxa_deteccao     = vp / total_noshow_real * 100
print(f'  No-shows reais no teste        : {total_noshow_real:,}')
print(f'  No-shows detectados pelo modelo: {noshow_detectados:,}')
print(f'  Taxa de detecção (recall)      : {taxa_deteccao:.1f}%')
print(f'  No-shows não detectados (FN)   : {fn:,}')
print(f'  Alarmes falsos (FP)            : {fp:,}')

# ── 7. GRÁFICO — MATRIZ DE CONFUSÃO ──────────────────────────────
print('\n--- Gerando gráficos ---')
fig, ax = plt.subplots(figsize=(7, 6))
cores   = [['#2E86C1', '#FDEDEC'], ['#FDEDEC', '#2E86C1']]
rotulos = [['VN', 'FP'], ['FN', 'VP']]
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle(
            (j, 1 - i), 1, 1,
            color=cores[i][j], alpha=0.7
        ))
        ax.text(
            j + 0.5, 1.5 - i,
            f'{rotulos[i][j]}\n{cm[i, j]:,}',
            ha='center', va='center',
            fontsize=14, fontweight='bold'
        )
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(
    ['Previsto: Compareceu', 'Previsto: Não compareceu'],
    fontsize=11
)
ax.set_yticklabels(
    ['Real: Não compareceu', 'Real: Compareceu'],
    fontsize=11
)
ax.set_title('Matriz de Confusão', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'avaliacao_01_matriz_confusao.png', bbox_inches='tight')
plt.close()

# ── 8. GRÁFICO — CURVA ROC ────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_teste, y_pred_prob)
fig, ax     = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#2E86C1', lw=2,
        label=f'Modelo (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--',
        label='Aleatório (AUC = 0.500)')
ax.fill_between(fpr, tpr, alpha=0.1, color='#2E86C1')
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos (Recall)', fontsize=12)
ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'avaliacao_02_curva_roc.png', bbox_inches='tight')
plt.close()

# ── 9. GRÁFICO — BARRAS DE MÉTRICAS ──────────────────────────────
metricas  = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
valores   = [acuracia, precisao, recall, f1]
cores_bar = ['#5D6D7E', '#2E86C1', '#C0392B', '#1E8449']
fig, ax   = plt.subplots(figsize=(8, 5))
bars      = ax.bar(metricas, valores, color=cores_bar,
                   edgecolor='white', width=0.5)
ax.set_ylim(0, 1)
ax.set_ylabel('Valor', fontsize=12)
ax.set_title('Resumo das Métricas de Avaliação',
             fontsize=14, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--',
           alpha=0.5, label='Linha base (0.50)')
for bar, v in zip(bars, valores):
    ax.text(
        bar.get_x() + bar.get_width() / 2, v + 0.01,
        f'{v:.3f}', ha='center', fontweight='bold', fontsize=11
    )
ax.legend()
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'avaliacao_03_metricas.png', bbox_inches='tight')
plt.close()
print('  3 gráficos salvos em assets/')

# ── 10. SALVAR RELATÓRIO JSON ─────────────────────────────────────
relatorio = {
    'acuracia' : round(float(acuracia), 4),
    'precisao' : round(float(precisao), 4),
    'recall'   : round(float(recall),   4),
    'f1_score' : round(float(f1),       4),
    'auc_roc'  : round(float(auc),      4),
    'matriz_confusao': {
        'VN': int(vn), 'FP': int(fp),
        'FN': int(fn), 'VP': int(vp)
    },
    'analise_clinica': {
        'noshow_reais'      : int(total_noshow_real),
        'noshow_detectados' : int(noshow_detectados),
        'taxa_deteccao_pct' : round(taxa_deteccao, 1),
        'nao_detectados_FN' : int(fn),
        'alarmes_falsos_FP' : int(fp)
    }
}
caminho_json = PASTA_ASSETS / 'metrics_report.json'
with open(caminho_json, 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, indent=4, ensure_ascii=False)
print(f'  Relatório salvo em: {caminho_json}')

print()
print('=' * 60)
print('Avaliação concluída! Prossiga para a Parte 8.')
print('=' * 60)