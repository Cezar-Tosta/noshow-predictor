# Entendendo o Problema e os Dados
# Execute com: python notebooks/01_exploracao_dados.py
# Compatível com Windows e Linux
import pandas as pd
from pathlib import Path

# Caminho relativo compatível com Windows e Linux
CAMINHO_DATASET = Path('data') / 'raw' / 'KaggleV2-May-2016.csv'

# ── 1. CARREGAMENTO ──────────────────────────────────────────────
print('=' * 60)
print('EXPLORAÇÃO INICIAL DO DATASET')
print('=' * 60)
if not CAMINHO_DATASET.exists():
    print(f'ERRO: arquivo não encontrado em {CAMINHO_DATASET}') 
    print('Certifique-se de executar o script a partir da raiz do projeto.') 
    print('Exemplo: python notebooks/01_exploracao_dados.py')
    exit(1)
    
df = pd.read_csv(CAMINHO_DATASET)
print(f'\nDataset carregado com sucesso!')
print(f'  Linhas  : {df.shape[0]:>10,}')
print(f'  Colunas : {df.shape[1]:>10}')

# ── 2. PRIMEIRAS LINHAS ────────────────────────────────────────── 
print('\n--- Primeiras 5 linhas ---')
print(df.head().to_string())

# ── 3. TIPOS DE DADOS E NULOS ──────────────────────────────────── 
print('\n--- Tipos de dados e contagem de não-nulos ---')
df.info()

# ── 4. VALORES NULOS ───────────────────────────────────────────── 
print('\n--- Valores nulos por coluna ---')
nulos = df.isnull().sum()
if nulos.sum() == 0:
    print('  Nenhum valor nulo encontrado.')
else:
    print(nulos[nulos > 0])
    
# ── 5. DUPLICATAS ──────────────────────────────────────────────── 
print('\n--- Verificação de duplicatas ---')
duplicatas = df.duplicated().sum()
print(f'  Linhas completamente duplicadas : {duplicatas}')
ids_unicos = df['AppointmentID'].nunique()
print(f'  AppointmentIDs únicos           : {ids_unicos:,} de {df.shape[0]:,} total')

# ── 6. DISTRIBUIÇÃO DA VARIÁVEL-ALVO ───────────────────────────── 
print('\n--- Distribuição da variável-alvo (No-show) ---')
contagem   = df['No-show'].value_counts()
percentual = df['No-show'].value_counts(normalize=True) * 100
print(f'  Compareceu     (No)  : {contagem["No"]:>6,}  ({percentual["No"]:>5.1f}%)')
print(f'  Não compareceu (Yes) : {contagem["Yes"]:>6,}  ({percentual["Yes"]:>5.1f}%)')
print()
print('  Lembrete: "No" = compareceu = 0 | "Yes" = não compareceu = 1')

# ── 7. ESTATÍSTICAS DESCRITIVAS ────────────────────────────────── 
print('\n--- Estatísticas descritivas (variáveis numéricas) ---') 
print(df.describe().to_string())

# ── 8. IDADES INVÁLIDAS ─────────────────────────────────────────── 
print('\n--- Verificação de idades inválidas (Age < 0) ---')
idades_invalidas = df[df['Age'] < 0]
print(f'  Registros com idade negativa: {len(idades_invalidas)}')
if len(idades_invalidas) > 0:
    print(idades_invalidas[['PatientId', 'Age', 'No-show']].to_string()) 
    print('  → Este registro será corrigido  quando fizermos o pré-processamento.')

# ── 9. TAXA DE NO-SHOW POR GÊNERO ──────────────────────────────── 
print('\n--- Taxa de no-show por gênero ---')
taxa_genero = df.groupby('Gender')['No-show'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(1)
print(taxa_genero.to_string())

# ── 10. TAXA DE NO-SHOW POR RECEBIMENTO DE SMS ─────────────────── 
print('\n--- Taxa de no-show por recebimento de SMS ---')
taxa_sms = df.groupby('SMS_received')['No-show'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(1)
print(f'  SMS = 0 (não recebeu) : {taxa_sms[0]}%')
print(f'  SMS = 1 (recebeu)     : {taxa_sms[1]}%')
print('  → Resultado aparentemente contraintuitivo — será explicado mais adiante.')

# ── 11. VALORES ÚNICOS EM VARIÁVEIS CATEGÓRICAS ────────────────── 
print('\n--- Valores únicos em variáveis categóricas ---')
categoricas = ['Gender', 'No-show', 'Neighbourhood']
for col in categoricas:
    print(f'  {col:<20}: {df[col].nunique()} valores únicos')

print('\n' + '=' * 60)
print('Exploração inicial dos dados concluída!')
print('=' * 60)




# Observações Maicon sobre os dados:

# Uma informação que me surpreendeu foi na análise Taxa de no-show por gẽnero, eu imaginava que homens teriam a tendência de faltar mais que mulheres, porém, os números são bem próximos.
# Outra informação que me surpreendeu foi a Verificação de idades inválidas, não imaginava que poderia ter esse tipo de erro dentro do banco de dados.
# os números de no-show de pessoas que recebem SMS e que não recebem são bem aproximadas.

# Sugestão Maicon:

# Alguns termos me causaram dúvidas pela falta de conhecimento de inglês, por exemplo, a coluna 'Scholarship' que significa Bolsa Familia, ou, a coluna 'Neighbourhood' que significa Bairro, deixo como sugestão de melhoria traduzirmos todas as colunas para facilitar a compreensão.
