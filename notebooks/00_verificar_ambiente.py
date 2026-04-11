# noshow-predictor / notebooks / 00_verificar_ambiente.py
# Verificação do ambiente de desenvolvimento
# Execute com: python notebooks/00_verificar_ambiente.py
import sys

print('=' * 55)
print('VERIFICAÇÃO DO AMBIENTE — NOSHOW PREDICTOR')
print('=' * 55)
print(f'Python : {sys.version}')
print()

bibliotecas = [
('pandas',     'pandas'),
('numpy',      'numpy'),
('matplotlib', 'matplotlib'),
('seaborn',    'seaborn'),
('sklearn',    'scikit-learn'),
('xgboost',    'xgboost'),
('shap',       'shap'),
('streamlit',  'streamlit'),
('joblib',     'joblib'),
]

todas_ok = True
for modulo, nome_exibido in bibliotecas:
    try:
        mod    = __import__(modulo)
        versao = getattr(mod, '__version__', 'OK')
        print(f'  {nome_exibido:<15} {versao}')
    except ImportError:
        print(f'  {nome_exibido:<15} *** NÃO INSTALADO ***')
        todas_ok = False
        
print()
if todas_ok:
    print('Ambiente configurado com sucesso!')
else:
    print('ATENÇÃO: algumas bibliotecas estão faltando.')
    print('Execute: pip install pandas numpy matplotlib seaborn')
    print('         scikit-learn xgboost shap streamlit joblib') 

print('=' * 55)