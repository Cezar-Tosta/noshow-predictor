# Previsão de No-Show em Consultas Médicas

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Sobre o Projeto

Sistema de Machine Learning construído do zero para prever a probabilidade de um paciente não comparecer à sua consulta médica agendada (**no-show**), com base em dados históricos de agendamentos do sistema público de saúde.

O projeto foi desenvolvido em **10 partes**, cada uma documentada e commitada separadamente. O repositório foi construído como material de mentoria — permitindo que qualquer pessoa acompanhe o processo completo de construção de um pipeline de ML, do dado bruto ao deploy.

## O Problema

Em unidades públicas de saúde, o não comparecimento de pacientes gera:

- Desperdício de vagas e recursos
- Ociosidade de profissionais de saúde
- Prejuízo a outros pacientes em lista de espera

Este modelo permite ações proativas — como envio de lembretes direcionados, overbooking controlado e priorização de lista de espera — com base na probabilidade prevista de no-show para cada agendamento.

## O Pipeline — Visão Geral

O projeto percorre cinco etapas, cada uma correspondendo a um conjunto de partes:

```
Dado bruto → Exploração → Pré-processamento → Modelo → App → Deploy
 (Partes 2-3)  (Parte 4-5)     (Partes 6-8)    (Parte 9) (Parte 10)
```

| Parte | Título | Script gerado |
|---|---|---|
| 1 | Visão Geral, Ambiente Local e Repositório | `00_verificar_ambiente.py` |
| 2 | Entendendo o Problema e os Dados | `01_exploracao_dados.py` | `01a_auditoria_dados.py` |
| 3 | Análise Exploratória de Dados (EDA) | `02_eda.py` |
| 4 | Pré-processamento e Engenharia de Features | `03_preprocessamento.py` |
| 5 | Seleção de Features e Divisão dos Dados | `04_selecao_features.py` |
| 6 | Treinamento do Modelo com XGBoost | `05_treinamento.py` |
| 7 | Avaliação do Modelo | `06_avaliacao.py` |
| 8 | Interpretabilidade com SHAP | `07_interpretabilidade.py` |
| 9 | Construção do App com Streamlit | `app.py` |
| 10 | Deploy no Streamlit Cloud e Documentação Final | `pipeline_completo.py` |

## Stack Tecnológica

| Camada | Ferramenta | Função |
|---|---|---|
| Linguagem | Python 3.10+ | Linguagem principal |
| Ambiente | venv | Isolamento de dependências |
| Editor | VS Code | Edição local dos scripts |
| Versionamento | Git + GitHub Desktop | Controle de versão sem terminal |
| Dados | pandas + numpy | Manipulação e transformação |
| Visualização | matplotlib + seaborn | Gráficos da EDA |
| ML | scikit-learn + XGBoost | Pré-processamento e treinamento |
| Interpretabilidade | SHAP | Explicabilidade das previsões |
| Persistência | joblib | Salvar e carregar o modelo |
| App web | Streamlit | Interface interativa |
| Deploy | Streamlit Cloud | Publicação gratuita via GitHub |

## Estrutura do Repositório

```
noshow-predictor/
├── data/
│   ├── raw/                  ← dataset original (baixar manualmente, ver abaixo)
│   └── processed/            ← CSVs gerados na Parte 5
├── notebooks/
│   ├── 00_verificar_ambiente.py
│   ├── 01_exploracao_dados.py
│   ├── 01a_auditoria_dados.py
│   ├── 02_eda.py
│   ├── 03_preprocessamento.py
│   ├── 04_selecao_features.py
│   ├── 05_treinamento.py
│   ├── 06_avaliacao.py
│   ├── 07_interpretabilidade.py
│   └── pipeline_completo.py
├── model/                    ← modelo treinado — gerado localmente na Parte 6
├── assets/                   ← gráficos .png gerados pela EDA e SHAP
├── app.py                    ← aplicação Streamlit
├── requirements.txt          ← dependências do projeto
├── .gitignore
└── README.md
```

> **Arquivos não versionados** (listados no `.gitignore`):
> - `.venv/` — ambiente virtual, criado localmente por cada colaborador
> - `data/raw/*.csv` — dataset original (~9 MB), baixar do Kaggle
> - `model/*.pkl` — modelo treinado, gerado ao executar a Parte 6

## Dicionário de Dados

Abaixo estão as descrições das colunas presentes no dataset original para facilitar o entendimento do negócio:

| Coluna | Descrição |
| :--- | :--- |
| **PatientId** | Identificação única do paciente. |
| **AppointmentID** | Identificação única de cada agendamento. |
| **Gender** | Sexo do paciente (F = Feminino, M = Masculino). |
| **ScheduledDay** | O dia em que o paciente marcou a consulta. |
| **AppointmentDay** | O dia real da consulta médica. |
| **Age** | Idade do paciente. |
| **Neighbourhood** | O bairro onde a unidade de saúde está localizada. |
| **Scholarship** | Indica se o paciente é beneficiário do Bolsa Família (0 = Não, 1 = Sim). |
| **Hipertension** | Se o paciente possui diagnóstico de hipertensão (0 ou 1). |
| **Diabetes** | Se o paciente possui diagnóstico de diabetes (0 ou 1). |
| **Alcoholism** | Se o paciente possui diagnóstico de alcoolismo (0 ou 1). |
| **Handcap** | Indica se o paciente possui alguma deficiência (0 a 4). |
| **SMS_received** | Se 1 ou mais mensagens SMS foram enviadas ao paciente. |
| **No-show** | **Alvo (Target):** "No" (Compareceu) ou "Yes" (Faltou). |

## Como Configurar o Projeto Localmente

### 1. Pré-requisitos

- Python 3.10 ou superior
- VS Code com a extensão Python instalada
- GitHub Desktop instalado

### 2. Fazer fork e clonar

1. Clique em **Fork** no canto superior direito desta página
2. Abra o GitHub Desktop
3. **File → Clone repository → aba GitHub.com → selecione seu fork**
4. Escolha uma pasta local e clique em **Clone**

### 3. Abrir no VS Code

No GitHub Desktop: **Repository → Open in Visual Studio Code**

### 4. Criar e ativar o ambiente virtual

No terminal integrado do VS Code, a partir da raiz do projeto:

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

O terminal exibirá `(.venv)` no início da linha quando o ambiente estiver ativo.

### 5. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 6. Baixar o dataset

O dataset não está no repositório — ele é ignorado pelo `.gitignore` (regra `data/raw/*.csv`).

Baixe em: https://www.kaggle.com/datasets/joniarroba/noshowappointments

Salve o arquivo **KaggleV2-May-2016.csv** em: `data/raw/`

### 7. Verificar o ambiente

```bash
python notebooks/00_verificar_ambiente.py
```

Todas as bibliotecas devem aparecer com versão, sem erros.

### 8. Selecionar o interpretador no VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter** → selecione `.venv`

### 9. Executar os scripts

Execute sempre a partir da raiz do projeto. Os scripts usam `pathlib.Path` para montar caminhos compatíveis com Windows e Linux:

```bash
python notebooks/01_exploracao_dados.py
python notebooks/01a_auditoria_dados.py
python notebooks/02_eda.py
# ... e assim por diante, em ordem
```

## Como Navegar pelo Histórico e Acompanhar a Construção do Projeto

O repositório foi construído parte por parte, com commits padronizados. Isso permite que qualquer pessoa acompanhe a evolução completa do projeto — do ambiente inicial ao deploy — como se fosse um livro com capítulos.

### Visualizando o histórico no GitHub

1. Acesse o repositório no GitHub
2. Clique na aba **Commits** (ou no número de commits exibido abaixo dos botões Code / Issues / Pull requests)
3. Você verá a lista completa de commits em ordem cronológica inversa (mais recente primeiro)

Para ler na ordem de construção, role até o commit mais antigo ou use o botão **Older** no final da lista.

### Visualizando o histórico no GitHub Desktop

1. Abra o GitHub Desktop com o repositório selecionado
2. Clique na aba **History** no painel esquerdo
3. Clique em qualquer commit para ver exatamente quais arquivos foram criados ou modificados naquele momento
4. Clique em um arquivo dentro do commit para ver as linhas adicionadas (em verde) e removidas (em vermelho)

### Entendendo a convenção de commits

Cada commit segue o formato `tipo: descrição`, onde o tipo indica o que aconteceu:

| Prefixo | Significado | Exemplo |
|---|---|---|
| `chore:` | Configuração inicial, dependências, infra | `chore: configuração inicial do ambiente` |
| `feat:` | Novo script ou funcionalidade | `feat: adiciona EDA com 7 gráficos exploratórios` |
| `fix:` | Correção de erro | `fix: corrige cálculo de dias_antecedencia negativos` |
| `docs:` | Atualização de documentação | `docs: atualiza README após Parte 3` |
| `refactor:` | Melhoria sem mudança de comportamento | `refactor: simplifica função de pré-processamento` |

### Reconstruindo o projeto a partir do zero com o histórico

Para estudar o projeto como se o estivesse construindo do zero, você pode fazer checkout de qualquer commit específico e ver o estado exato do repositório naquele ponto:

**Pelo GitHub (navegador):**
1. Vá até a aba **Commits**
2. Clique no ícone `<>` (Browse the repository at this point in history) ao lado do commit desejado
3. O GitHub mostrará o repositório exatamente como estava naquele momento

**Pelo GitHub Desktop:**
1. Na aba **History**, clique com o botão direito no commit desejado
2. Selecione **Checkout commit**
3. O repositório local voltará para aquele estado
4. Para retornar ao estado atual: **Branch → main**

### Mapa de commits por parte

Ao navegar pelo histórico, você encontrará os commits na seguinte sequência:

```
chore: configuração inicial do ambiente e estrutura do projeto   ← Parte 1
feat: adiciona script de exploração inicial do dataset           ← Parte 2
feat: adiciona EDA com 7 gráficos exploratórios                  ← Parte 3
feat: adiciona script de pré-processamento                       ← Parte 4
feat: adiciona seleção de features e divisão dos dados           ← Parte 5
feat: adiciona treinamento do modelo com XGBoost                 ← Parte 6
feat: adiciona avaliação do modelo e relatório de métricas       ← Parte 7
feat: adiciona interpretabilidade com SHAP                       ← Parte 8
feat: adiciona aplicação Streamlit                               ← Parte 9
feat: deploy no Streamlit Cloud e documentação final             ← Parte 10
```

Cada commit é um ponto de parada — você pode ler o script adicionado naquele commit, executá-lo, entender o que ele faz e só então avançar para o próximo.

## Dataset

**Medical Appointment No Shows — Kaggle**

- 110.527 registros de agendamentos reais
- Coletado em Vitória (ES), Brasil, entre 2015 e 2016
- 14 variáveis — demográficas, clínicas e de agendamento
- Target: coluna `No-show` (`"Yes"` = não compareceu = 1 | `"No"` = compareceu = 0)

> ⚠️ A coluna `No-show` tem lógica invertida: `"Yes"` significa **não compareceu**.

Fonte: https://www.kaggle.com/datasets/joniarroba/noshowappointments

## App

*(URL pública será adicionada após o deploy na Parte 10)*

## Contribuindo

Este projeto é desenvolvido em regime de mentoria. O mentorado trabalha em um fork próprio e contribui via Pull Requests e Issues abertas no repositório original.

## Autor

Cezar Tosta
