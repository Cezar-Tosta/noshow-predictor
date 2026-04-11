# Previsão de No-Show em Consultas Médicas
![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Status](https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow)
## Objetivo
Sistema de Machine Learning para prever a probabilidade de um paciente
não comparecer à sua consulta médica agendada (no-show), com base em
dados históricos de agendamentos do sistema público de saúde.
## Problema
O não comparecimento de pacientes gera desperdício de recursos, ociosidade
de profissionais e impede o atendimento de outros pacientes em lista de espera. Este modelo permite ações proativas para reduzir o índice de no-show.
## Stack Tecnológica
- Python 3.10+
- pandas, numpy, scikit-learn, XGBoost
- matplotlib, seaborn (visualizações)
- SHAP (interpretabilidade)
- joblib (persistência do modelo)
- Streamlit (app web)
- GitHub Desktop (versionamento)
## Estrutura do Repositório
```
noshow-predictor/
├── data/
│   ├── raw/            ← dataset original (baixar manualmente, ver abaixo)
│   └── processed/      ← CSVs gerados pelo pré-processamento
├── notebooks/          ← scripts .py
├── model/              ← modelo treinado (gerado localmente)
├── assets/             ← gráficos .png gerados pela EDA e SHAP
├── app.py              ← aplicação Streamlit
├── requirements.txt    ← dependências do projeto
└── README.md
```
## Como Configurar o Projeto Localmente
### 1. Pré-requisitos
- Python 3.10 ou superior instalado
- VS Code instalado com a extensão Python
- GitHub Desktop instalado
### 2. Clonar o repositório
No GitHub Desktop: File → Clone repository → noshow-predictor.
Escolha uma pasta local e clique em Clone.
### 3. Criar e ativar o ambiente virtual
No terminal do VS Code (PowerShell), dentro da pasta do projeto:
```bash
python -m venv .venv
.venv\Scripts\Activate
```
### 4. Instalar as dependências
```bash
pip install -r requirements.txt
```
### 5. Baixar o dataset
O dataset original não está no repositório (arquivo grande).<br>
Baixe em: https://www.kaggle.com/datasets/joniarroba/noshowappointments<br> 
Salve o arquivo <b>KaggleV2-May-2016.csv</b> em: `data/raw/`
### 6. Verificar o ambiente
```bash
python notebooks/00_verificar_ambiente.py
```
Todas as bibliotecas devem aparecer com versão, sem erros.
### 7. Selecionar o interpretador no VS Code
Ctrl+Shift+P → Python: Select Interpreter → escolha .venv
## Dataset
Medical Appointment No Shows — Kaggle<br>
110.527 registros | 14 variáveis | Target: No-show (Yes/No)<br>
Fonte: https://www.kaggle.com/datasets/joniarroba/noshowappointments 
## App
*(URL pública será adicionada)*
## Autor
Cezar Tosta