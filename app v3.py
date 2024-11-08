import os
import pdfplumber
import json
import re
import pandas as pd
import streamlit as st
from typing import Tuple
import time
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from gtts import gTTS
import tempfile

# Configurações da página do Streamlit
st.set_page_config(
    page_title="Consultor de PDFs + IA",
    page_icon="logo.png",
    layout="wide",
)

# Definição de constantes
FILEPATH = "agents.json"
CHAT_HISTORY_FILE = 'chat_history.json'
API_USAGE_FILE = 'api_usage.json'

MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Definição das chaves de API
API_KEYS = {
    "fetch": ["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", "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"],
    "refine": ["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", "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"],
    "evaluate": ["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", "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"]
}

# Variáveis para manter o estado das chaves de API
CURRENT_API_KEY_INDEX = {
    "fetch": 0,
    "refine": 0,
    "evaluate": 0
}

# Função para obter a próxima chave de API disponível
def get_next_api_key(action: str) -> str:
    global CURRENT_API_KEY_INDEX
    keys = API_KEYS.get(action, [])
    if keys:
        key_index = CURRENT_API_KEY_INDEX[action]
        api_key = keys[key_index]
        CURRENT_API_KEY_INDEX[action] = (key_index + 1) % len(keys)
        return api_key
    else:
        raise ValueError(f"No API keys available for action: {action}")

# [As demais funções permanecem inalteradas]

# Interface Principal com Streamlit

if 'resposta_assistente' not in st.session_state:
    st.session_state.resposta_assistente = ""
if 'descricao_especialista_ideal' not in st.session_state:
    st.session_state.descricao_especialista_ideal = ""
if 'resposta_refinada' not in st.session_state:
    st.session_state.resposta_refinada = ""
if 'resposta_original' not in st.session_state:
    st.session_state.resposta_original = ""
if 'rag_resposta' not in st.session_state:
    st.session_state.rag_resposta = ""
if 'references_df' not in st.session_state:
    st.session_state.references_df = pd.DataFrame()

agent_options = load_agent_options()

st.image('updating (2).gif', width=100, caption='Consultor de PDFs + IA', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Consultor de PDFs</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Utilize nossa plataforma para consultas detalhadas em PDFs.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100, 150, 300, 450])

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    interaction_number = len(load_api_usage()) + 1

    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar")

    references_file = st.file_uploader("Upload do arquivo JSON ou PDF com referências (opcional)", type=["json", "pdf"], key="arquivo_referencias")

with col2:
    container_saida = st.container()

    chat_history = load_chat_history()[-memory_selection:]

    if fetch_clicked:
        if references_file:
            df = upload_and_extract_references(references_file)
            if isinstance(df, pd.DataFrame):
                st.write("### Dados Extraídos do PDF")
                st.dataframe(df)
                st.session_state.references_path = "references.csv"
                st.session_state.references_df = df

        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, chat_history, interaction_number, st.session_state.get('references_df'))
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""
        save_chat_history(user_input, user_prompt, st.session_state.resposta_assistente)

    if refine_clicked:
        if st.session_state.resposta_assistente:
            references_context = ""
            if not st.session_state.references_df.empty:
                for index, row in st.session_state.references_df.iterrows():
                    titulo = row.get('titulo', row['Text'][:50] + '...')
                    autor = row.get('autor', 'Autor Desconhecido')
                    ano = row.get('ano', 'Ano Desconhecido')
                    paginas = row.get('Page', 'Página Desconhecida')
                    references_context += f"Título: {titulo}\nAutor: {autor}\nAno: {ano}\nPágina: {paginas}\n\n"
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, references_context, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.resposta_refinada)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.rag_resposta)
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")

    with container_saida:
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")

        if st.session_state.resposta_original:
            # Converter a resposta em fala
            tts = gTTS(st.session_state.resposta_original, lang='pt')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                # Reproduzir o áudio na aplicação
                st.audio(fp.name, format='audio/mp3')

        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
            # Converter a resposta refinada em fala
            tts_refinada = gTTS(st.session_state.resposta_refinada, lang='pt')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp_refinada:
                tts_refinada.save(fp_refinada.name)
                st.audio(fp_refinada.name, format='audio/mp3')

        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")
            # Converter a avaliação com RAG em fala
            tts_rag = gTTS(st.session_state.rag_resposta, lang='pt')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp_rag:
                tts_rag.save(fp_rag.name)
                st.audio(fp_rag.name, format='audio/mp3')

    st.markdown("### Histórico do Chat")
    if chat_history:
        tab_titles = [f"Interação {i+1}" for i in range(len(chat_history))]
        tabs = st.tabs(tab_titles)
        
        for i, entry in enumerate(chat_history):
            with tabs[i]:
                st.write(f"**Entrada do Usuário:** {entry['user_input']}")
                st.write(f"**Prompt do Usuário:** {entry['user_prompt']}")
                st.write(f"**Resposta do Especialista:** {entry['expert_response']}")
                st.markdown("---")

if refresh_clicked:
    clear_chat_history()
    st.session_state.clear()
    st.rerun()

# [O restante do código permanece inalterado]

