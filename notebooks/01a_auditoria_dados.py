# Entendendo o Problema e os Dados (Auditoria de Qualidade)
# Execute com: python notebooks/01a_auditoria_dados.py
# Compatível com Windows e Linux
import pandas as pd
from pathlib import Path

# Caminho relativo compatível com Windows e Linux
CAMINHO_DATASET = Path('data') / 'raw' / 'KaggleV2-May-2016.csv'

# ── 1. CARREGAMENTO E STATUS ──────────────────────────────────────
print('=' * 60)
print('AUDITORIA DE QUALIDADE DOS DADOS (PASSO 01a)')
print('=' * 60)

if not CAMINHO_DATASET.exists():
    print(f'ERRO: arquivo não encontrado em {CAMINHO_DATASET}')
    print('Certifique-se de executar o script a partir da raiz do projeto.')
    exit(1)

df = pd.read_csv(CAMINHO_DATASET)
print(f'\nDataset carregado para auditoria!')
print(f'  Total de registros para análise: {df.shape[0]:,}')

# ── 2. AUDITORIA DE IDADES (OUTLIERS) ─────────────────────────────
print('\n--- Verificação de idades inconsistentes ---')
idades_negativas = df[df['Age'] < 0]
idades_centenarias = df[df['Age'] > 105]

print(f'  Idades negativas encontradas : {len(idades_negativas)}')
if len(idades_negativas) > 0:
    print(idades_negativas[['PatientId', 'Age']].to_string(index=False))

print(f'  Idades acima de 105 anos     : {len(idades_centenarias)}')
if len(idades_centenarias) > 0:
    print(f'  → Alerta: {len(idades_centenarias)} pacientes com idade muito avançada.')

# ── 3. LÓGICA TEMPORAL (DATAS) ────────────────────────────────────
print('\n--- Verificação de inconsistência de datas ---')
# Convertendo para comparar apenas as datas (sem considerar horas)
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date

datas_invalidas = df[df['AppointmentDay'] < df['ScheduledDay']]
print(f'  Consultas retroativas encontradas: {len(datas_invalidas)}')

if len(datas_invalidas) > 0:
    print('  (Consultas marcadas para antes do dia do agendamento)')
    print(datas_invalidas[['AppointmentID', 'ScheduledDay', 'AppointmentDay']].head().to_string(index=False))

# ── 4. INTEGRIDADE DE IDENTIFICADORES ─────────────────────────────
print('\n--- Verificação de AppointmentID (Chave Primária) ---')
duplicados_id = df['AppointmentID'].duplicated().sum()
print(f'  IDs de agendamento duplicados : {duplicados_id}')

# ── 5. RESUMO DE QUALIDADE ────────────────────────────────────────
print('\n--- Resumo para o Pré-processamento ---')
total_erros = len(idades_negativas) + len(datas_invalidas)
print(f'  Total de registros com erros críticos: {total_erros}')
print('  → Estes dados devem ser filtrados no script 03_preprocessamento.py')

print('\n' + '=' * 60)
print('Auditoria de qualidade concluída!')
print('=' * 60)