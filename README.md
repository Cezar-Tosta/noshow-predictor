# Previsão de No-Show em Consultas Médicas

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow)
![Parte](https://img.shields.io/badge/Progresso-Parte%202%20de%2010-informational)

## Objetivo

Sistema de Machine Learning para prever a probabilidade de um paciente
não comparecer à sua consulta médica agendada (no-show), com base em
dados históricos de agendamentos do sistema público de saúde.

## Problema

O não comparecimento de pacientes gera desperdício de recursos, ociosidade
de profissionais e impede o atendimento de outros pacientes em lista de espera.
Este modelo permite ações proativas para reduzir o índice de no-show.

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
├── notebooks/          ← scripts .py por parte
│   ├── 00_verificar_ambiente.py
│   └── 01_exploracao_dados.py
├── model/              ← modelo treinado (gerado localmente na Parte 6)
├── assets/             ← gráficos .png gerados pela EDA e SHAP
├── app.py              ← aplicação Streamlit (Parte 9)
├── requirements.txt    ← dependências do projeto
└── README.md
```

## Como Configurar o Projeto Localmente

### 1. Pré-requisitos

- Python 3.10 ou superior instalado
- VS Code instalado com a extensão Python
- GitHub Desktop instalado

### 2. Clonar o repositório

No GitHub Desktop: **File → Clone repository → noshow-predictor**.
Escolha uma pasta local e clique em Clone.

### 3. Criar e ativar o ambiente virtual

No terminal do VS Code, dentro da pasta do projeto:

**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Se aparecer erro de política de execução, rode antes:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux / macOS (Bash):**
```bash
python -m venv .venv
source .venv/bin/activate
```

Em ambos os sistemas, o terminal exibirá `(.venv)` no início da linha quando o ambiente estiver ativo.

### 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 5. Baixar o dataset

O dataset original não está no repositório (arquivo grande).

Baixe em: https://www.kaggle.com/datasets/joniarroba/noshowappointments

Salve o arquivo **KaggleV2-May-2016.csv** em: `data/raw/`

> ⚠️ Adicione a linha `data/raw/*.csv` ao seu `.gitignore` para garantir que o dataset não seja commitado acidentalmente.

### 6. Verificar o ambiente

```bash
python notebooks/00_verificar_ambiente.py
```

Todas as bibliotecas devem aparecer com versão, sem erros.

### 7. Selecionar o interpretador no VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter** → escolha `.venv`

### 8. Executar os scripts

Execute sempre a partir da raiz do projeto:

```bash
python notebooks/01_exploracao_dados.py
```

> Os scripts usam `pathlib.Path` para montar caminhos — compatíveis com Windows e Linux.

## Dataset

**Medical Appointment No Shows — Kaggle**
110.527 registros | 14 variáveis | Target: No-show (Yes/No)

Fonte: https://www.kaggle.com/datasets/joniarroba/noshowappointments

| Variável | Tipo | Descrição |
|---|---|---|
| No-show | TARGET | `"Yes"` = não compareceu (1) \| `"No"` = compareceu (0) |
| Age | Numérico | Idade do paciente em anos |
| Gender | Categórico | F (feminino) ou M (masculino) |
| ScheduledDay | Data/hora | Data em que o agendamento foi feito |
| AppointmentDay | Data/hora | Data da consulta agendada |
| Neighbourhood | Categórico | Bairro da unidade de saúde (81 valores) |
| Scholarship | Binário | 1 = inscrito no Bolsa Família |
| Hipertension | Binário | 1 = tem hipertensão |
| Diabetes | Binário | 1 = tem diabetes |
| SMS_received | Binário | 1 = recebeu SMS de lembrete |

> ⚠️ A coluna `No-show` tem lógica invertida: `"Yes"` significa **não compareceu**.

## Progresso do Projeto

| Parte | Título | Status |
|---|---|---|
| Parte 1 | Visão Geral, Ambiente Local e Repositório | ✅ Concluída |
| Parte 2 | Entendendo o Problema e os Dados | ✅ Concluída |
| Parte 3 | Análise Exploratória de Dados (EDA) | 🔄 Em andamento |
| Parte 4 | Pré-processamento e Engenharia de Features | ⏳ Pendente |
| Parte 5 | Seleção de Features e Divisão dos Dados | ⏳ Pendente |
| Parte 6 | Treinamento do Modelo com XGBoost | ⏳ Pendente |
| Parte 7 | Avaliação do Modelo | ⏳ Pendente |
| Parte 8 | Interpretabilidade com SHAP | ⏳ Pendente |
| Parte 9 | Construção do App com Streamlit | ⏳ Pendente |
| Parte 10 | Deploy no Streamlit Cloud e Documentação Final | ⏳ Pendente |

## Resultados

*(será atualizado após a Parte 7)*

## App

*(URL pública será adicionada após a Parte 10)*

## Contribuindo

Este projeto é desenvolvido em regime de mentoria. O mentorado trabalha em um fork próprio e contribui via Pull Requests e Issues.

1. Faça um fork do repositório
2. Clone o fork localmente com o GitHub Desktop
3. Configure o ambiente seguindo os passos acima
4. Abra uma Issue de confirmação no repositório original

## Autor

Cezar Tosta