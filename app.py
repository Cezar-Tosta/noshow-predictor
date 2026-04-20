# noshow-predictor / app.py
# Parte 9 — Aplicação Streamlit
# Execute com: streamlit run app.py
# Compatível com Windows e Linux

import json
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ── CONFIGURAÇÃO DA PÁGINA ───────────────────────────────────────
st.set_page_config(
    page_title='Previsão de No-Show',
    page_icon='🏥',
    layout='wide'
)

# ── CARREGAMENTO DO MODELO (COM CACHE) ───────────────────────────
@st.cache_resource
def carregar_modelo():
    modelo        = joblib.load(Path('model') / 'model.pkl')
    feature_names = joblib.load(Path('model') / 'feature_names.pkl')
    return modelo, feature_names

@st.cache_data
def carregar_metricas():
    caminho = Path('assets') / 'metrics_report.json'
    if caminho.exists():
        with open(caminho, encoding='utf-8') as f:
            return json.load(f)
    return None

try:
    modelo, feature_names = carregar_modelo()
    modelo_carregado = True
except Exception as e:
    modelo_carregado = False
    st.error(f'Erro ao carregar o modelo: {e}')
    st.info('Execute o script da Parte 6 para gerar o modelo.')
    st.stop()

metricas = carregar_metricas()

# ── CABEÇALHO ────────────────────────────────────────────────────
st.title('🏥 Previsão de No-Show em Consultas Médicas')
st.markdown(
    'Sistema de Machine Learning para prever a probabilidade de um '
    'paciente **não comparecer** à sua consulta médica agendada.'
)
st.divider()

# ── ABAS PRINCIPAIS ──────────────────────────────────────────────
aba_previsao, aba_metricas, aba_sobre = st.tabs([
    '🔮 Previsão', '📊 Métricas do Modelo', 'ℹ️ Sobre'
])

# ══════════════════════════════════════════════════════════════════
# ABA 1 — PREVISÃO
# ══════════════════════════════════════════════════════════════════
with aba_previsao:

    # Sidebar — formulário de entrada
    with st.sidebar:
        st.header('📋 Dados do Paciente')
        st.markdown('Preencha os campos abaixo e clique em **Prever**.')
        st.divider()

        genero = st.selectbox(
            'Gênero', options=['F', 'M'],
            format_func=lambda x: 'Feminino' if x == 'F' else 'Masculino'
        )
        idade = st.number_input(
            'Idade (anos)', min_value=0, max_value=115, value=35, step=1
        )
        dias_antecedencia = st.number_input(
            'Dias de antecedência do agendamento',
            min_value=0, max_value=200, value=10, step=1
        )
        sms_recebido = st.checkbox('Recebeu SMS de lembrete')

        st.markdown('**Condições de saúde:**')
        hipertensao   = st.checkbox('Hipertensão')
        diabetes      = st.checkbox('Diabetes')
        alcoolismo    = st.checkbox('Alcoolismo')
        deficiencia   = st.checkbox('Deficiência registrada')
        bolsa_familia = st.checkbox('Inscrito no Bolsa Família')

        st.divider()
        btn_prever = st.button('🔮 Prever No-Show', use_container_width=True)

    # Área principal — resultado
    if btn_prever:

        # Montar dicionário com todos os campos do formulário
        entrada_raw = {
            'Age'              : int(idade),
            'dias_antecedencia': int(dias_antecedencia),
            'SMS_received'     : int(sms_recebido),
            'Hipertension'     : int(hipertensao),
            'Diabetes'         : int(diabetes),
            'Alcoholism'       : int(alcoolismo),
            'Handcap'          : int(deficiencia),
            'Scholarship'      : int(bolsa_familia),
            'Gender_F'         : int(genero == 'F'),
            'Gender_M'         : int(genero == 'M'),
        }

        # Criar DataFrame e alinhar com as features do modelo
        entrada_df = pd.DataFrame([entrada_raw])
        entrada_df = entrada_df.reindex(columns=feature_names, fill_value=0)

        # Fazer a previsão
        prob     = modelo.predict_proba(entrada_df)[0][1]
        predicao = modelo.predict(entrada_df)[0]

        # Exibir resultado
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label='Probabilidade de No-Show',
                value=f'{prob:.1%}'
            )
        with col2:
            if prob < 0.30:
                st.success('✅ Risco BAIXO de no-show')
            elif prob < 0.55:
                st.warning('⚠️ Risco MÉDIO de no-show')
            else:
                st.error('🚨 Risco ALTO de no-show')
        with col3:
            st.metric(
                label='Previsão do Modelo',
                value='Não comparecerá' if predicao == 1 else 'Comparecerá'
            )

        st.divider()

        # Gráfico SHAP waterfall
        st.subheader('🔍 Por que o modelo fez essa previsão?')
        st.markdown(
            'O gráfico abaixo mostra quais fatores **aumentaram** (vermelho) '
            'ou **diminuíram** (azul) a probabilidade de no-show para este paciente.'
        )

        explainer   = shap.TreeExplainer(modelo)
        shap_values = explainer(entrada_df)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close()

        # Resumo textual dos principais fatores
        st.subheader('📝 Resumo dos principais fatores')
        contrib = pd.Series(
            shap_values.values[0], index=feature_names
        ).reindex(
            pd.Series(
                shap_values.values[0], index=feature_names
            ).abs().sort_values(ascending=False).index
        )
        for feat, val in contrib.head(5).items():
            direcao    = '🔴 Aumentou' if val > 0 else '🔵 Reduziu'
            valor_feat = entrada_df[feat].values[0]
            st.markdown(
                f'- **{feat}** = {valor_feat:.0f} → {direcao} o risco ({val:+.3f})'
            )

    else:
        st.info(
            '👈 Preencha os dados do paciente na barra lateral '
            'e clique em **Prever No-Show**.'
        )

# ══════════════════════════════════════════════════════════════════
# ABA 2 — MÉTRICAS DO MODELO
# ══════════════════════════════════════════════════════════════════
with aba_metricas:
    st.subheader('📊 Desempenho do Modelo no Conjunto de Teste')

    if metricas:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Acurácia', f'{metricas["acuracia"]:.1%}')
        col2.metric('Precisão', f'{metricas["precisao"]:.1%}')
        col3.metric('Recall',   f'{metricas["recall"]:.1%}')
        col4.metric('F1-score', f'{metricas["f1_score"]:.1%}')
        col5.metric('AUC-ROC',  f'{metricas["auc_roc"]:.3f}')

        st.divider()
        st.subheader('Matriz de Confusão')
        cm = metricas['matriz_confusao']
        df_cm = pd.DataFrame(
            [[cm['VN'], cm['FP']], [cm['FN'], cm['VP']]],
            index  =['Real: Compareceu', 'Real: Não compareceu'],
            columns=['Previsto: Compareceu', 'Previsto: Não compareceu']
        )
        st.dataframe(df_cm, use_container_width=True)

        st.divider()
        st.subheader('Análise Clínica')
        ac = metricas['analise_clinica']
        col1, col2, col3 = st.columns(3)
        col1.metric(
            'No-shows reais detectados',
            f'{ac["noshow_detectados"]:,}',
            f'{ac["taxa_deteccao_pct"]}% dos casos reais'
        )
        col2.metric('Não detectados (FN)', f'{ac["nao_detectados_FN"]:,}')
        col3.metric('Alarmes falsos (FP)', f'{ac["alarmes_falsos_FP"]:,}')
    else:
        st.warning(
            'Arquivo metrics_report.json não encontrado. '
            'Execute o script da Parte 7.'
        )

# ══════════════════════════════════════════════════════════════════
# ABA 3 — SOBRE
# ══════════════════════════════════════════════════════════════════
with aba_sobre:
    st.subheader('Sobre o Projeto')
    st.markdown('''
    Este projeto foi desenvolvido como material de mentoria em Machine Learning.
    O objetivo é demonstrar o pipeline completo de um projeto de ML aplicado à saúde pública:
    da exploração dos dados ao deploy de uma aplicação web interativa.

    **Dataset:** Medical Appointment No Shows — Kaggle
    (110.527 registros coletados em Vitória/ES, 2015-2016)

    **Algoritmo:** XGBoost (Extreme Gradient Boosting)

    **Interpretabilidade:** SHAP (SHapley Additive exPlanations)

    **Repositório:** https://github.com/seu-usuario/noshow-predictor
    ''')