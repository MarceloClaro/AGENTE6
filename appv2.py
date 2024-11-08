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
from gtts import gTTS  # Importação adicionada para vocalização
import tempfile        # Importação adicionada para manuseio de arquivos temporários
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

# Função para manipular limites de taxa
def handle_rate_limit(error_message: str, action: str):
    wait_time = 80  # Tempo padrão de espera
    match = re.search(r'Aguardando (\d+\.?\d*) segundos', error_message)
    if match:
        wait_time = float(match.group(1))
    st.warning(f"Limite de taxa atingido. Aguardando {wait_time} segundos...")
    time.sleep(wait_time)
    # Alterna a chave de API para a próxima disponível
    new_key = get_next_api_key(action)
    st.info(f"Usando nova chave de API para {action}: {new_key}")
# Função para carregar as opções de agentes
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de Agentes. Por favor, verifique o formato.")
    return agent_options

# Função para extrair texto de um arquivo PDF
def extrair_texto_pdf(file):
    texto_paginas = []
    with pdfplumber.open(file) as pdf:
        for num_pagina in range(len(pdf.pages)):
            pagina = pdf.pages[num_pagina]
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto_paginas.append({'page': num_pagina + 1, 'text': texto_pagina})
    return texto_paginas

# Função para converter o texto extraído em um DataFrame
def text_to_dataframe(texto_paginas):
    dados = {'Page': [], 'Text': []}
    for entrada in texto_paginas:
        dados['Page'].append(entrada['page'])
        dados['Text'].append(entrada['text'])
    return pd.DataFrame(dados)
# Função para fazer upload e extrair referências
def upload_and_extract_references(uploaded_file):
    references = {}
    try:
        if uploaded_file.name.endswith('.json'):
            references = json.load(uploaded_file)
            with open("references.json", 'w') as file:
                json.dump(references, file, indent=4)
            return "references.json"
        elif uploaded_file.name.endswith('.pdf'):
            texto_paginas = extrair_texto_pdf(uploaded_file)
            if not texto_paginas:
                st.error("Nenhum texto extraído do PDF.")
                return pd.DataFrame()
            df = text_to_dataframe(texto_paginas)
            if not df.empty:
                df.to_csv("references.csv", index=False)
                return df
            else:
                st.error("Nenhum texto extraído do PDF.")
                return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar e extrair referências: {e}")
        return pd.DataFrame()

# Função para obter o número máximo de tokens com base no modelo
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Função para registrar o uso da API
def log_api_usage(action: str, interaction_number: int, tokens_used: int, time_taken: float, user_input: str, user_prompt: str, api_response: str, agent_used: str, agent_description: str):
    entry = {
        'action': action,
        'interaction_number': interaction_number,
        'tokens_used': tokens_used,
        'time_taken': time_taken,
        'user_input': user_input,
        'user_prompt': user_prompt,
        'api_response': api_response,
        'agent_used': agent_used,
        'agent_description': agent_description
    }
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r+') as file:
            api_usage = json.load(file)
            api_usage.append(entry)
            file.seek(0)
            json.dump(api_usage, file, indent=4)
    else:
        with open(API_USAGE_FILE, 'w') as file:
            json.dump([entry], file, indent=4)
# Função para salvar o histórico do chat
def save_chat_history(user_input, user_prompt, expert_response, chat_history_file=CHAT_HISTORY_FILE):
    chat_entry = {
        'user_input': user_input,
        'user_prompt': user_prompt,
        'expert_response': expert_response
    }
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r+') as file:
            try:
                chat_history = json.load(file)
            except json.JSONDecodeError:
                chat_history = []
            chat_history.append(chat_entry)
            file.seek(0)
            json.dump(chat_history, file, indent=4)
    else:
        with open(chat_history_file, 'w') as file:
            json.dump([chat_entry], file, indent=4)

# Função para carregar o histórico do chat
def load_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r') as file:
            try:
                chat_history = json.load(file)
            except json.JSONDecodeError:
                chat_history = []
        return chat_history
    return []

# Função para limpar o histórico do chat
def clear_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        os.remove(chat_history_file)

# Função para carregar o uso da API
def load_api_usage():
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r') as file:
            try:
                api_usage = json.load(file)
            except json.JSONDecodeError:
                api_usage = []
        return api_usage
    return []

# Função para resetar o uso da API
def reset_api_usage():
    if os.path.exists(API_USAGE_FILE):
        os.remove(API_USAGE_FILE)
    st.success("Os dados de uso da API foram resetados.")
# Função para buscar a resposta do assistente
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float,
                             agent_selection: str, chat_history: list, interaction_number: int,
                             references_df: pd.DataFrame = None) -> Tuple[str, str]:
    phase_two_response = ""
    expert_title = ""
    expert_description = ""
    try:
        client = Groq(api_key=get_next_api_key('fetch'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            backoff_time = 1
            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=get_max_tokens(model_name),
                        top_p=1,
                        stop=None,
                        stream=False
                    )
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content if completion.choices else ""
                    log_api_usage('fetch', interaction_number, tokens_used, time_taken, user_input, user_prompt,
                                  api_response, expert_title, expert_description)
                    return api_response
                except Exception as e:
                    if "503" in str(e):
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")
                        return ""
                    handle_rate_limit(str(e), 'fetch')
                    backoff_time = min(backoff_time * 2, 84)
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")
                    time.sleep(backoff_time)

        if agent_selection == "Escolher um especialista...":
            # Código para gerar a descrição do especialista ideal
            phase_one_prompt = (
                f"Descreva o especialista ideal para responder à seguinte solicitação: {user_input} e {user_prompt}."
                # Instruções adicionais para o modelo
            )
            phase_one_response = get_completion(phase_one_prompt)
            # Processamento da resposta para extrair título e descrição do especialista
            # Salvar o especialista gerado
            save_expert(expert_title, expert_description)
        else:
            # Carregar o especialista selecionado
            if os.path.exists(FILEPATH):
                with open(FILEPATH, 'r') as file:
                    agents = json.load(file)
                    agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                    if agent_found:
                        expert_title = agent_found["agente"]
                        expert_description = agent_found["descricao"]
                    else:
                        raise ValueError("Especialista selecionado não encontrado no arquivo.")
            else:
                raise FileNotFoundError(f"Arquivo {FILEPATH} não encontrado.")

        # Montar o contexto do histórico de chat
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Montar o contexto das referências
        references_context = ""
        if references_df is not None and not references_df.empty:
            for index, row in references_df.iterrows():
                titulo = row.get('titulo', 'Título Desconhecido')
                autor = row.get('autor', 'Autor Desconhecido')
                ano = row.get('ano', 'Ano Desconhecido')
                paginas = row.get('Page', 'Página Desconhecida')
                references_context += f"Título: {titulo}\nAutor: {autor}\nAno: {ano}\nPáginas: {paginas}\n\n"

        # Montar o prompt para a segunda fase
        phase_two_prompt = (
            f"{expert_title}, por favor responda detalhadamente e em português à seguinte solicitação: {user_input} e {user_prompt}."
            f"\n\nHistórico de chat: {history_context}"
            f"\n\nReferências:\n{references_context}"
            # Instruções adicionais para o modelo
        )

        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

# Função para refinar a resposta
def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str,
                    model_name: str, temperature: float, references_context: str, chat_history: list,
                    interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_next_api_key('refine'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            backoff_time = 1
            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=get_max_tokens(model_name),
                        top_p=1,
                        stop=None,
                        stream=False
                    )
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content if completion.choices else ""
                    log_api_usage('refine', interaction_number, tokens_used, time_taken, user_input, user_prompt,
                                  api_response, expert_title, "")
                    return api_response
                except Exception as e:
                    if "503" in str(e):
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")
                        return ""
                    handle_rate_limit(str(e), 'refine')
                    backoff_time = min(backoff_time * 2, 64)
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")
                    time.sleep(backoff_time)

        # Montar o contexto do histórico de chat
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Montar o prompt para o refinamento
        refine_prompt = (
            f"{expert_title}, por favor refine a seguinte resposta: {phase_two_response}. Solicitação original: {user_input} e {user_prompt}."
            f"\n\nHistórico de chat: {history_context}"
            f"\n\nReferências:\n{references_context}"
            # Instruções adicionais para o modelo
        )

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""
# Função para avaliar a resposta com RAG
def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_title: str, expert_description: str,
                               assistant_response: str, model_name: str, temperature: float, chat_history: list,
                               interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_next_api_key('evaluate'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            backoff_time = 1
            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=get_max_tokens(model_name),
                        top_p=1,
                        stop=None,
                        stream=False
                    )
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content if completion.choices else ""
                    log_api_usage('evaluate', interaction_number, tokens_used, time_taken, user_input, user_prompt,
                                  api_response, expert_title, expert_description)
                    return api_response
                except Exception as e:
                    if "503" in str(e):
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")
                        return ""
                    handle_rate_limit(str(e), 'evaluate')
                    backoff_time = min(backoff_time * 2, 64)
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")
                    time.sleep(backoff_time)

        # Montar o contexto do histórico de chat
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Montar o prompt para avaliação com RAG
        rag_prompt = (
            f"{expert_title}, por favor avalie a seguinte resposta: {assistant_response}. Solicitação original: {user_input} e {user_prompt}."
            f"\n\nHistórico do chat: {history_context}"
            f"\n\nInstruções para avaliação:\n"
            f"Por favor, avalie a resposta fornecida quanto à precisão, completude e relevância. Identifique pontos fortes e áreas que podem ser melhoradas. Forneça sugestões específicas para aprimoramento."
            f"\n\n---\n"
            f"gen_id: [gerado automaticamente]\n"
            f"seed: [gerado automaticamente]\n"
        )

        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Função para salvar o especialista
def save_expert(expert_title: str, expert_description: str):
    new_expert = {
        "agente": expert_title,
        "descricao": expert_description
    }
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r+') as file:
            try:
                agents = json.load(file)
            except json.JSONDecodeError:
                agents = []
            agents.append(new_expert)
            file.seek(0)
            json.dump(agents, file, indent=4)
    else:
        with open(FILEPATH, 'w') as file:
            json.dump([new_expert], file, indent=4)
# Interface Principal com Streamlit

# Inicialização do estado da sessão
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

# Carregamento das opções de agentes
agent_options = load_agent_options()

# Configuração da interface do usuário
st.image('updating (2).gif', width=100, caption='Consultor de PDFs + IA', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Consultor de PDFs</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Utilize nossa plataforma para consultas detalhadas em PDFs.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Seleção da quantidade de interações para lembrar
memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100, 150, 300, 450])

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

# Criação de colunas para organizar a interface
col1, col2 = st.columns(2)
# Continuação da Interface Principal com Streamlit

with col1:
    # Campos de entrada do usuário
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):",
                               height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                            key="temperatura")
    interaction_number = len(load_api_usage()) + 1

    # Botões de interação
    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar")

    # Upload de referências
    references_file = st.file_uploader("Upload do arquivo JSON ou PDF com referências (opcional)", type=["json", "pdf"],
                                       key="arquivo_referencias")

with col2:
    container_saida = st.container()

    # Carregar o histórico do chat limitado ao número de interações selecionadas
    chat_history = load_chat_history()[-memory_selection:]

    if fetch_clicked:
        if references_file:
            df = upload_and_extract_references(references_file)
            if isinstance(df, pd.DataFrame):
                st.write("### Dados Extraídos do PDF")
                st.dataframe(df)
                st.session_state.references_path = "references.csv"
                st.session_state.references_df = df

        # Buscar a resposta do assistente
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(
            user_input, user_prompt, model_name, temperature, agent_selection, chat_history, interaction_number,
            st.session_state.get('references_df'))
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
            st.session_state.resposta_refinada = refine_response(
                st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input,
                user_prompt, model_name, temperature, references_context, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.resposta_refinada)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(
                user_input, user_prompt, st.session_state.descricao_especialista_ideal,
                st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name,
                temperature, chat_history, interaction_number)
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

    # Exibir o histórico do chat
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

# Botão para limpar o histórico e reiniciar a sessão
if refresh_clicked:
    clear_chat_history()
    st.session_state.clear()
    st.experimental_rerun()

# Sidebar com informações adicionais e gráficos
st.sidebar.image("logo.png", width=200)
with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    **[Conteúdo dos insights do código]**
    """)

st.sidebar.image("eu.ico", width=80)
st.sidebar.write("""
Projeto Consultor de PDFs + IA 
- Professor: Marcelo Claro.

Contatos: marceloclaro@gmail.com

Whatsapp: (88)981587145

Instagram: [https://www.instagram.com/marceloclaro.consultorpdfs/](https://www.instagram.com/marceloclaro.consultorpdfs/)
""")

# Carregar e plotar o uso da API
api_usage = load_api_usage()
if api_usage:
    plot_api_usage(api_usage)

if st.sidebar.button("Resetar Gráficos"):
    reset_api_usage()

# Funções adicionais para carregar referências e atualizar o histórico
def carregar_referencias():
    if os.path.exists('references.csv'):
        return pd.read_csv('references.csv')
    else:
        return pd.DataFrame()

def referencias_para_historico(df_referencias, chat_history_file=CHAT_HISTORY_FILE):
    if not df_referencias.empty:
        for _, row in df_referencias.iterrows():
            titulo = row.get('titulo', row['Text'][:50] + '...')
            autor = row.get('autor', 'Autor Desconhecido')
            ano = row.get('ano', 'Ano Desconhecido')
            paginas = row.get('Page', 'Página Desconhecida')

            chat_entry = {
                'user_input': f"Título: {titulo}",
                'user_prompt': f"Autor: {autor}\nAno: {ano}\nPágina: {paginas}\nTexto: {row['Text']}",
                'expert_response': 'Informação adicionada ao histórico de chat como referência.'
            }

            if os.path.exists(chat_history_file):
                with open(chat_history_file, 'r+') as file:
                    try:
                        chat_history = json.load(file)
                    except json.JSONDecodeError:
                        chat_history = []
                    chat_history.append(chat_entry)
                    file.seek(0)
                    json.dump(chat_history, file, indent=4)
            else:
                with open(chat_history_file, 'w') as file:
                    json.dump([chat_entry], file, indent=4)

# Atualizar o histórico com referências
df_referencias = carregar_referencias()
referencias_para_historico(df_referencias)
