# noshow-predictor / notebooks / 05_treinamento.py
# Parte 6 — Treinamento do Modelo com XGBoost
# Execute com: python notebooks/05_treinamento.py
# Compatível com Windows e Linux

import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

CAMINHO_TREINO = Path('data') / 'processed' / 'train.csv'
CAMINHO_VAL    = Path('data') / 'processed' / 'val.csv'
CAMINHO_MODELO = Path('model') / 'model.pkl'
CAMINHO_FEATS  = Path('model') / 'feature_names.pkl'

# ── 1. CARREGAMENTO ────────────────────────────────────────────── 
print('=' * 60)
print('TREINAMENTO DO MODELO — PARTE 6')
print('=' * 60)

for caminho in [CAMINHO_TREINO, CAMINHO_VAL]:
    if not caminho.exists():
        print(f'ERRO: arquivo não encontrado em {caminho}') 
        print('Execute primeiro o script da Parte 5.')
        exit(1)
        
treino = pd.read_csv(CAMINHO_TREINO)
val    = pd.read_csv(CAMINHO_VAL)

print(f'\nTreino    : {treino.shape[0]:,} linhas × {treino.shape[1]} colunas')
print(f'Validação : {val.shape[0]:,} linhas × {val.shape[1]} colunas')

# ── 2. SEPARAR FEATURES E ALVO ─────────────────────────────────── 
X_treino = treino.drop(columns=['noshow'])
y_treino = treino['noshow']
X_val    = val.drop(columns=['noshow'])
y_val    = val['noshow']

feature_names = X_treino.columns.tolist()
print(f'\nFeatures utilizadas: {len(feature_names)}')
print(f'  {feature_names}')

# ── 3. CALCULAR PESO PARA COMPENSAR DESEQUILÍBRIO DE CLASSES ───── 
negativos        = (y_treino == 0).sum()
positivos        = (y_treino == 1).sum()
scale_pos_weight = negativos / positivos

print(f'\nDesequilíbrio de classes:')
print(f'  Negativos (compareceu)     : {negativos:,}')
print(f'  Positivos (não compareceu) : {positivos:,}')
print(f'  scale_pos_weight           : {scale_pos_weight:.2f}')

# ── 4. CONFIGURAR O MODELO ──────────────────────────────────────── 
print('\n--- Configurando o modelo XGBoost ---')
modelo = xgb.XGBClassifier(
    n_estimators         = 300,
    max_depth            = 4,
    learning_rate        = 0.05,
    subsample            = 0.8,
    colsample_bytree     = 0.8,
    scale_pos_weight     = scale_pos_weight,
    early_stopping_rounds= 30,
    eval_metric          = 'auc',
    random_state         = 42,
    n_jobs               = -1,   # usa todos os núcleos disponíveis
)

# ── 5. TREINAR O MODELO ─────────────────────────────────────────── 
print('\n--- Treinando o modelo ---')
print('(Aguarde — pode levar alguns minutos...)')

modelo.fit(
    X_treino, y_treino,
    eval_set=[(X_treino, y_treino), (X_val, y_val)],
    verbose=50,   # exibe progresso a cada 50 rodadas
)

# ── 6. RESULTADOS DO TREINAMENTO ───────────────────────────────── 
melhor_rodada = modelo.best_iteration
print(f'\n--- Resultados do treinamento ---')
print(f'  Melhor rodada (early stopping) : {melhor_rodada}')
print(f'  n_estimators configurado       : 300')

resultados = modelo.evals_result()
auc_treino_final = resultados['validation_0']['auc'][melhor_rodada] 
auc_val_final    = resultados['validation_1']['auc'][melhor_rodada] 
print(f'  AUC no treino     : {auc_treino_final:.4f}')
print(f'  AUC na validação  : {auc_val_final:.4f}')

diferenca = auc_treino_final - auc_val_final
print(f'  Diferença treino-validação : {diferenca:.4f}')
if diferenca < 0.05:
    print('  ✓ Diferença pequena — sem sinais evidentes de overfitting.') 
else:
    print('  ⚠ Diferença alta — possível overfitting. Considere reduzir max_depth ou n_estimators.')
    
# ── 7. SALVAR O MODELO E OS NOMES DAS FEATURES ─────────────────── 
print('\n--- Salvando modelo e features ---') 
CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True) 
joblib.dump(modelo,        CAMINHO_MODELO)
joblib.dump(feature_names, CAMINHO_FEATS)
print(f'  Modelo salvo em  : {CAMINHO_MODELO}')
print(f'  Features salvas em: {CAMINHO_FEATS}')

print()
print('=' * 60)
print('Treinamento concluído! Prossiga para a Parte 7.')
print('=' * 60)