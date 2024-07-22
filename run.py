import pdfplumber
import json
import re
import os
import shutil

# Função para extrair texto de um PDF em um intervalo específico de páginas
def extrair_texto_pdf_intervalos(caminho_pdf, pagina_inicial, pagina_final, limite_paginas):
    intervalos_texto = []
    with pdfplumber.open(caminho_pdf) as pdf:
        for inicio_intervalo in range(pagina_inicial - 1, min(pagina_final, len(pdf.pages)), limite_paginas):
            fim_intervalo = min(inicio_intervalo + limite_paginas, pagina_final)
            texto_intervalo = []
            for num_pagina in range(inicio_intervalo, fim_intervalo):
                pagina = pdf.pages[num_pagina]
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto_intervalo.append(texto_pagina)
            if texto_intervalo:
                intervalos_texto.append(" ".join(texto_intervalo))
    return intervalos_texto

# Função para identificar seções com base em expressões regulares
def identificar_secoes(texto, secao_inicial):
    secoes = {}
    secao_atual = secao_inicial
    secoes[secao_atual] = ""

    paragrafos = texto.split('\n')
    for paragrafo in paragrafos:
        match = re.match(r'Parte \d+\.', paragrafo) or re.match(r'Capítulo \d+: .*', paragrafo) or re.match(r'\d+\.\d+ .*', paragrafo)
        if match:
            secao_atual = match.group()
            secoes[secao_atual] = ""
        else:
            secoes[secao_atual] += paragrafo + "\n"

    return secoes

# Função para salvar os dados em um arquivo JSON
def salvar_como_json(dados, caminho_saida):
    with open(caminho_saida, 'w', encoding='utf-8') as file:
        json.dump(dados, file, ensure_ascii=False, indent=4)

# Função para processar e salvar cada intervalo como JSON
def processar_e_salvar(intervalos_texto, secao_inicial, caminho_pasta_base, nome_arquivo):
    for i, texto_intervalo in enumerate(intervalos_texto):
        secoes = identificar_secoes(texto_intervalo, secao_inicial)
        caminho_saida = os.path.join(caminho_pasta_base, f"{nome_arquivo}_{i}.json")
        salvar_como_json(secoes, caminho_saida)


# Função para fazer upload e extração de textos de arquivos JSON ou PDF
def upload_and_extract_references(uploaded_file):
    references = {}
    try:
        if uploaded_file.name.endswith('.json'):
            references = json.load(uploaded_file)
            with open("references.json", 'w') as file:
                json.dump(references, file, indent=4)
            return "references.json"
        elif uploaded_file.name.endswith('.pdf'):
            intervalos_texto = extrair_texto_pdf_intervalos(uploaded_file, 1, 1000, 10)
            dfs = [text_to_dataframe(texto) for texto in intervalos_texto if texto]
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df.to_csv("references.csv", index=False)
                return df
            else:
                st.error("Nenhum texto extraído do PDF.")
                return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar e extrair referências: {e}")
        return pd.DataFrame()


import streamlit as st
import pandas as pd

# Função para converter texto em DataFrame
def text_to_dataframe(text):
    lines = text.split('\n')
    data = [line.split() for line in lines if line.strip()]
    if data:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame()
    return df

# Carrega as opções de Agentes a partir do arquivo JSON
agent_options = load_agent_options()

# Layout da página
st.image('updating.gif', width=300, caption='Consultor de PDFs + IA', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Consultor de PDFs</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Utilize nossa plataforma para consultas detalhadas em PDFs.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Seleção de memória do chat
memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100])

# Caixa de entrada para solicitação do usuário
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

    container_saida = st.container()

    chat_history = load_chat_history()[-memory_selection:]

    if fetch_clicked:
        if references_file:
            df = upload_and_extract_references(references_file)
            if isinstance(df, pd.DataFrame):
                st.write("### Dados Extraídos do PDF")
                st.dataframe(df)
                st.session_state.references_path = "references.csv"

        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, chat_history, interaction_number)
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""
        save_chat_history(user_input, user_prompt, st.session_state.resposta_assistente)

    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, references_file, chat_history, interaction_number)
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
        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")

    st.markdown("### Histórico do Chat")
    for entry in chat_history:
        st.write(f"**Entrada do Usuário:** {entry['user_input']}")
        st.write(f"**Prompt do Usuário:** {entry['user_prompt']}")
        st.write(f"**Resposta do Especialista:** {entry['expert_response']}")
        st.markdown("---")

if refresh_clicked:
    clear_chat_history()
    st.session_state.clear()
    st.rerun()

# Sidebar com manual de uso
st.sidebar.image("logo.png", width=200)
with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do Consultor de PDFs + IA é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

    **Inovações:**
    - Suporte a múltiplos modelos de linguagem: O código permite que o usuário escolha entre diferentes modelos de linguagem, como o LLaMA, para gerar respostas mais precisas e personalizadas.
    - Integração com a API Groq: A integração com a API Groq permite que o aplicativo utilize a capacidade de processamento de linguagem natural de alta performance para gerar respostas precisas.
    - Refinamento de respostas: O código permite que o usuário refine as respostas do modelo de linguagem, tornando-as mais precisas e relevantes para a consulta.
    - Avaliação com o RAG: A avaliação com o RAG (Rational Agent Generator) permite que o aplicativo avalie a qualidade e a precisão das respostas do modelo de linguagem.

    **Pontos positivos:**
    - Personalização: O aplicativo permite que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas de acordo com suas necessidades.
    - Precisão: A integração com a API Groq e o refinamento de respostas garantem que as respostas sejam precisas e relevantes para a consulta.
    - Flexibilidade: O código é flexível o suficiente para permitir que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas.

    **Limitações:**
    - Dificuldade de uso: O aplicativo pode ser difícil de usar para os usuários que não têm experiência com modelos de linguagem ou API.
    - Limitações de token: O código tem limitações em relação ao número de tokens que podem ser processados pelo modelo de linguagem.
    - Necessidade de treinamento adicional: O modelo de linguagem pode precisar de treinamento adicional para lidar com consultas mais complexas ou específicas.

    **Importância de ter colocado instruções em chinês:**
    A linguagem chinesa tem uma densidade de informação mais alta do que muitas outras línguas, o que significa que os modelos de linguagem precisam processar menos tokens para entender o contexto e gerar respostas precisas. Isso torna a linguagem chinesa mais apropriada para a utilização de modelos de linguagem com baixa quantidade de tokens. Portanto, ter colocado instruções em chinês no código é um recurso importante para garantir que o aplicativo possa lidar com consultas em chinês de forma eficaz.

    Em resumo, o código é uma aplicação inovadora que combina modelos de linguagem com a API Groq para proporcionar respostas precisas e personalizadas. No entanto, é importante considerar as limitações do aplicativo e trabalhar para melhorá-lo ainda mais.
    """)

    # Informações de contato
    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Consultor de PDFs + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.consultorpdfs/](https://www.instagram.com/marceloclaro.consultorpdfs/)
    """)

# Carrega o uso da API e plota o histograma
api_usage = load_api_usage()
if api_usage:
    plot_api_usage(api_usage)

# Botão para resetar os gráficos
if st.sidebar.button("Resetar Gráficos"):
    reset_api_usage()


