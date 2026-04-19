# noshow-predictor / notebooks / 03_preprocessamento.py
# Parte 4 — Pré-processamento e Engenharia de Features
# Execute com: python notebooks/03_preprocessamento.py
# Compatível com Windows e Linux

import pandas as pd
from pathlib import Path
CAMINHO_DATASET   = Path('data') / 'raw'       / 'KaggleV2-May-2016.csv' 
CAMINHO_SAIDA     = Path('data') / 'processed' / 'dados_processados.csv'

# ── 1. CARREGAMENTO ──────────────────────────────────────────────
print('=' * 60)
print('PRÉ-PROCESSAMENTO — PARTE 4')
print('=' * 60)

if not CAMINHO_DATASET.exists():
    print(f'ERRO: dataset não encontrado em {CAMINHO_DATASET}')
    print('Execute a partir da raiz do projeto.')
    exit(1)
    
df = pd.read_csv(CAMINHO_DATASET)
print(f'\nDataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas')

# ── 2. LIMPEZA — REGISTROS INVÁLIDOS ───────────────────────────── 
print('\n--- Limpeza: removendo registros inválidos ---')

# Remover idades negativas (identificadas na Parte 2)
antes = len(df)
df = df[df['Age'] >= 0]
removidos = antes - len(df)
print(f'  Registros com Age < 0 removidos: {removidos}')

# Remover agendamentos com data da consulta anterior à data do agendamento 
df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
antes = len(df)
df = df[df['AppointmentDay'] >= df['ScheduledDay']]
removidos_datas = antes - len(df)
print(f'  Registros com data inválida removidos: {removidos_datas}')
print(f'  Registros restantes: {len(df):,}')

# ── 3. VARIÁVEL-ALVO — CONVERSÃO PARA NUMÉRICO ─────────────────── 
print('\n--- Variável-alvo: Yes → 1, No → 0 ---')
df['noshow'] = (df['No-show'] == 'Yes').astype(int)
print(f'  Compareceu     (0): {(df["noshow"] == 0).sum():,}')
print(f'  Não compareceu (1): {(df["noshow"] == 1).sum():,}')

# ── 4. ENGENHARIA DE FEATURES — DIAS DE ANTECEDÊNCIA ───────────── 
print('\n--- Feature: dias_antecedencia ---')
df['dias_antecedencia'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
print(f'  Mínimo  : {df["dias_antecedencia"].min()} dias')
print(f'  Máximo  : {df["dias_antecedencia"].max()} dias')
print(f'  Média   : {df["dias_antecedencia"].mean():.1f} dias')
print(f'  Mediana : {df["dias_antecedencia"].median():.0f} dias')

# ── 5. REMOÇÃO DE COLUNAS IRRELEVANTES ─────────────────────────── 
print('\n--- Removendo colunas irrelevantes ---')
colunas_remover = [
    'PatientId',       # apenas identificador
    'AppointmentID',   # apenas identificador
    'ScheduledDay',    # substituída por dias_antecedencia 'AppointmentDay',  # substituída por dias_antecedencia
    'No-show',         # substituída pela coluna noshow (numérica)
]
df = df.drop(columns=colunas_remover)
print(f'  Colunas removidas: {colunas_remover}')
print(f'  Colunas restantes: {df.shape[1]}')

# ── 6. ENCODING CATEGÓRICO — GET_DUMMIES ───────────────────────── 
print('\n--- Encoding categórico com get_dummies ---')
colunas_antes = df.shape[1]
df = pd.get_dummies(df, drop_first=False)
colunas_depois = df.shape[1]
print(f'  Colunas antes do encoding : {colunas_antes}')
print(f'  Colunas após o encoding   : {colunas_depois}')
print(f'  Novas colunas criadas     : {colunas_depois - colunas_antes}')

# Garantir que a coluna-alvo é inteira após o get_dummies
df['noshow'] = df['noshow'].astype(int)

# ── 7. RESUMO FINAL ─────────────────────────────────────────────── 
print('\n--- Resumo do dataset processado ---')
print(f'  Linhas  : {df.shape[0]:,}')
print(f'  Colunas : {df.shape[1]}')
print(f'  Taxa de no-show: {df["noshow"].mean()*100:.1f}%')
print()
print('  Primeiras colunas:')
print(f'  {list(df.columns[:8])}')

# ── 8. SALVAMENTO ───────────────────────────────────────────────── 
print('\n--- Salvando dataset processado ---') 
CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True) 
df.to_csv(CAMINHO_SAIDA, index=False)
print(f'  Arquivo salvo em: {CAMINHO_SAIDA}')

print()
print('=' * 60)
print('Pré-processamento concluído! Prossiga para a Parte 5.')
print('=' * 60)