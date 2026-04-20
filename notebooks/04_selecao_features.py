# noshow-predictor / notebooks / 04_selecao_features.py
# Parte 5 — Seleção de Features e Divisão dos Dados
# Execute com: python notebooks/04_selecao_features.py
# Compatível com Windows e Linux

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CAMINHO_ENTRADA = Path('data') / 'processed' / 'dados_processados.csv' 
CAMINHO_TREINO  = Path('data') / 'processed' / 'train.csv'
CAMINHO_VAL     = Path('data') / 'processed' / 'val.csv'
CAMINHO_TESTE   = Path('data') / 'processed' / 'test.csv'
LIMIAR_CORR     = 0.01  # correlação mínima para manter a coluna

# ── 1. CARREGAMENTO ──────────────────────────────────────────────
print('=' * 60)
print('SELEÇÃO DE FEATURES E DIVISÃO DOS DADOS — PARTE 5')
print('=' * 60)

if not CAMINHO_ENTRADA.exists():
    print(f'ERRO: arquivo não encontrado em {CAMINHO_ENTRADA}') 
    print('Execute primeiro o script da Parte 4.')
    exit(1)
    
df = pd.read_csv(CAMINHO_ENTRADA)
print(f'\nDataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas')

# ── 2. CORRELAÇÃO COM A VARIÁVEL-ALVO ──────────────────────────── 
print('\n--- Correlação com a variável-alvo (noshow) ---')
corrs = df.corr()['noshow'].abs().drop('noshow')
corrs_ordenadas = corrs.sort_values(ascending=False)

print('\n  Top 15 features mais correlacionadas:') 
print(corrs_ordenadas.head(15).round(4).to_string())

# ── 3. FILTRAGEM DAS FEATURES ──────────────────────────────────── 
print(f'\n--- Filtrando colunas com correlação > {LIMIAR_CORR} ---') 
features_selecionadas = corrs[corrs > LIMIAR_CORR].index.tolist() 
colunas_finais        = ['noshow'] + features_selecionadas
df_filtrado           = df[colunas_finais]

print(f'  Colunas antes da filtragem : {df.shape[1]}')
print(f'  Colunas após a filtragem   : {df_filtrado.shape[1]}')
print(f'  Features selecionadas      : {len(features_selecionadas)}')

# ── 4. DIVISÃO TREINO / VALIDAÇÃO / TESTE ──────────────────────── 
print('\n--- Dividindo os dados em treino, validação e teste ---')

# Separar features (X) e variável-alvo (y)
X = df_filtrado.drop(columns=['noshow'])
y = df_filtrado['noshow']

# Primeira divisão: 70% treino, 30% temporário
X_treino, X_temp, y_treino, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y  # garante proporção de classes em cada conjunto
)

# Segunda divisão: 20% validação, 10% teste (do 30% temporário) 
X_val, X_teste, y_val, y_teste = train_test_split(
    X_temp, y_temp,
    test_size=0.333,  # 0.333 × 30% ≈ 10% do total
    random_state=42,
    stratify=y_temp
)

print(f'  Treino     : {len(X_treino):>6,} linhas  ({len(X_treino)/len(X)*100:.0f}%)')
print(f'  Validação  : {len(X_val):>6,} linhas  ({len(X_val)/len(X)*100:.0f}%)')
print(f'  Teste      : {len(X_teste):>6,} linhas  ({len(X_teste)/len(X)*100:.0f}%)')

# Verificar proporção da classe-alvo em cada conjunto
print('\n  Taxa de no-show por conjunto (stratify=y):')
print(f'  Treino    : {y_treino.mean()*100:.1f}%')
print(f'  Validação : {y_val.mean()*100:.1f}%')
print(f'  Teste     : {y_teste.mean()*100:.1f}%')

# ── 5. RECOMBINAR X E Y PARA SALVAR ────────────────────────────── 
train_df = X_treino.copy()
train_df['noshow'] = y_treino.values

val_df = X_val.copy()
val_df['noshow'] = y_val.values

test_df = X_teste.copy()
test_df['noshow'] = y_teste.values

# ── 6. SALVAMENTO DOS CSVs ──────────────────────────────────────── 
print('\n--- Salvando conjuntos em data/processed/ ---') 
train_df.to_csv(CAMINHO_TREINO, index=False) 
val_df.to_csv(CAMINHO_VAL,      index=False) 
test_df.to_csv(CAMINHO_TESTE,   index=False)

print(f'  train.csv : {CAMINHO_TREINO}')
print(f'  val.csv   : {CAMINHO_VAL}')
print(f'  test.csv  : {CAMINHO_TESTE}')

# ── 7. RESUMO FINAL ─────────────────────────────────────────────── 
print('\n--- Resumo final ---')
print(f'  Features no modelo : {len(features_selecionadas)}') 
print(f'  Total de amostras  : {len(X):,}')
print(f'  Treino             : {len(train_df):,}')
print(f'  Validação          : {len(val_df):,}')
print(f'  Teste              : {len(test_df):,}')

print()
print('=' * 60)
print('Seleção de features concluída! Prossiga para a Parte 6.') 
print('=' * 60)