import os  # Importa funções para interagir com o sistema operacional (como manipulação de arquivos e pastas)
import pdfplumber  # Importa a biblioteca para extrair texto de arquivos PDF
import json  # Importa funções para trabalhar com dados no formato JSON
import re  # Importa funções para trabalhar com expressões regulares (útil para encontrar padrões em texto)
import pandas as pd  # Importa a biblioteca pandas para manipulação e análise de dados
import streamlit as st  # Importa a biblioteca Streamlit para criar aplicações web interativas
from typing import Tuple  # Importa a função Tuple para anotar tipos de dados em funções
import time  # Importa funções para manipulação de tempo (como pausas e medições de duração)
import matplotlib.pyplot as plt  # Importa a biblioteca matplotlib para criação de gráficos
import seaborn as sns  # Importa a biblioteca seaborn para criação de gráficos estatísticos
from groq import Groq  # Importa a biblioteca Groq para interagir com modelos de linguagem


# Configurações da página do Streamlit
# Esta seção define como a página web será configurada e exibida

st.set_page_config(
    page_title="Consultor de PDFs + IA",  # Define o título da página que aparecerá na aba do navegador
    page_icon="logo.png",  # Define o ícone que aparecerá na aba do navegador (neste caso, um arquivo chamado logo.png)
    layout="wide",  # Define o layout da página como "wide", o que significa que a página ocupará toda a largura da janela do navegador
)

# Definição de constantes
# Constantes são valores que não mudam durante a execução do programa

FILEPATH = "agents.json"  # Caminho do arquivo onde estão salvos os agentes (especialistas)
CHAT_HISTORY_FILE = 'chat_history.json'  # Caminho do arquivo onde está salvo o histórico de conversas
API_USAGE_FILE = 'api_usage.json'  # Caminho do arquivo onde está salvo o uso da API

# Dicionário que armazena o número máximo de tokens suportados por cada modelo
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,  # Modelo mixtral com 32768 tokens
    'llama3-70b-8192': 8192,  # Modelo llama3-70b com 8192 tokens
    'llama3-8b-8192': 8192,  # Modelo llama3-8b com 8192 tokens
    'gemma-7b-it': 8192,  # Modelo gemma-7b-it com 8192 tokens
}

# Dicionário que armazena as chaves de API para diferentes ações
API_KEYS = {
    "fetch": ["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", "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"],  # Chaves para ação de busca
    "refine": ["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", "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"],  # Chaves para ação de refinamento
    "evaluate": ["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", "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"]  # Chaves para ação de avaliação
}


# Funções utilitárias
# Estas funções são auxiliares e ajudam a realizar tarefas específicas no código

# Função para obter a próxima chave de API disponível para uma determinada ação
def get_api_key(action: str) -> str:
    keys = API_KEYS.get(action, [])  # Obtém a lista de chaves de API para a ação especificada
    if keys:
        return keys.pop(0)  # Retira e retorna a primeira chave da lista
    else:
        raise ValueError(f"No API keys available for action: {action}")  # Levanta um erro se não houver chaves disponíveis

# Função para carregar as opções de agentes (especialistas) a partir de um arquivo JSON
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']  # Opção padrão na lista de agentes
    if os.path.exists(FILEPATH):  # Verifica se o arquivo de agentes existe
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)  # Carrega os agentes do arquivo
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])  # Adiciona os nomes dos agentes à lista de opções
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de Agentes. Por favor, verifique o formato.")  # Mostra um erro se houver problema ao ler o arquivo
    return agent_options  # Retorna a lista de opções de agentes


# Função para extrair texto de um arquivo PDF
def extrair_texto_pdf(file):
    texto_paginas = []  # Lista para armazenar o texto de cada página
    with pdfplumber.open(file) as pdf:  # Abre o arquivo PDF usando pdfplumber
        for num_pagina in range(len(pdf.pages)):  # Itera sobre todas as páginas do PDF
            pagina = pdf.pages[num_pagina]  # Obtém a página atual
            texto_pagina = pagina.extract_text()  # Extrai o texto da página
            if texto_pagina:  # Se houver texto na página
                texto_paginas.append({'page': num_pagina + 1, 'text': texto_pagina})  # Adiciona o texto da página à lista, junto com o número da página
    return texto_paginas  # Retorna a lista de textos das páginas

# Função para converter o texto das páginas em um DataFrame
def text_to_dataframe(texto_paginas):
    dados = {'Page': [], 'Text': []}  # Dicionário para armazenar os dados (número da página e texto)
    for entrada in texto_paginas:  # Itera sobre cada entrada na lista de textos das páginas
        dados['Page'].append(entrada['page'])  # Adiciona o número da página ao dicionário
        dados['Text'].append(entrada['text'])  # Adiciona o texto da página ao dicionário
    return pd.DataFrame(dados)  # Converte o dicionário em um DataFrame e retorna


# Função para identificar seções no texto
# Esta função separa o texto em diferentes seções com base em padrões específicos (como "Parte", "Capítulo", etc.)

def identificar_secoes(texto, secao_inicial):
    secoes = {}  # Dicionário para armazenar as seções
    secao_atual = secao_inicial  # Inicializa a seção atual com o valor passado como parâmetro
    secoes[secao_atual] = ""  # Adiciona a seção inicial ao dicionário

    paragrafos = texto.split('\n')  # Divide o texto em parágrafos
    for paragrafo in paragrafos:  # Itera sobre cada parágrafo
        # Verifica se o parágrafo corresponde a um padrão de nova seção (ex: "Parte", "Capítulo", etc.)
        match = re.match(r'Parte \d+\.', paragrafo) or re.match(r'Capítulo \d+: .*', paragrafo) or re.match(r'\d+\.\d+ .*', paragrafo)
        if match:  # Se encontrar uma correspondência
            secao_atual = match.group()  # Atualiza a seção atual com o novo título da seção
            secoes[secao_atual] = ""  # Inicializa a nova seção no dicionário
        else:
            secoes[secao_atual] += paragrafo + "\n"  # Adiciona o parágrafo à seção atual

    return secoes  # Retorna o dicionário com as seções e seus respectivos textos


# Função para salvar dados em um arquivo JSON
def salvar_como_json(dados, caminho_saida):
    with open(caminho_saida, 'w', encoding='utf-8') as file:  # Abre o arquivo no modo de escrita com codificação UTF-8
        json.dump(dados, file, ensure_ascii=False, indent=4)  # Salva os dados no arquivo em formato JSON, com indentação de 4 espaços para melhor legibilidade

# Função para processar o texto das páginas e salvar as seções como um arquivo JSON
def processar_e_salvar(texto_paginas, secao_inicial, caminho_pasta_base, nome_arquivo):
    # Concatena o texto de todas as páginas em uma única string e identifica as seções
    secoes = identificar_secoes(" ".join([entrada['text'] for entrada in texto_paginas]), secao_inicial)
    # Cria o caminho completo do arquivo de saída
    caminho_saida = os.path.join(caminho_pasta_base, f"{nome_arquivo}.json")
    # Salva as seções identificadas como um arquivo JSON
    salvar_como_json(secoes, caminho_saida)

# Função para preencher dados faltantes em uma referência
# Esta função retorna um dicionário com informações preenchidas quando faltam dados sobre o título
def preencher_dados_faltantes(titulo):
    return {
        'titulo': titulo,  # Preenche o título fornecido
        'autor': 'Autor Desconhecido',  # Preenche "Autor Desconhecido" quando o autor não é fornecido
        'ano': 'Ano Desconhecido',  # Preenche "Ano Desconhecido" quando o ano não é fornecido
        'paginas': 'Páginas Desconhecidas'  # Preenche "Páginas Desconhecidas" quando o número de páginas não é fornecido
    }

# Função para fazer upload e extrair referências de arquivos JSON ou PDF
def upload_and_extract_references(uploaded_file):
    references = {}  # Inicializa um dicionário vazio para armazenar as referências
    try:
        if uploaded_file.name.endswith('.json'):  # Verifica se o arquivo é um JSON
            references = json.load(uploaded_file)  # Carrega as referências do arquivo JSON
            with open("references.json", 'w') as file:  # Abre um arquivo para salvar as referências
                json.dump(references, file, indent=4)  # Salva as referências no arquivo com indentação de 4 espaços
            return "references.json"  # Retorna o nome do arquivo salvo
        elif uploaded_file.name.endswith('.pdf'):  # Verifica se o arquivo é um PDF
            texto_paginas = extrair_texto_pdf(uploaded_file)  # Extrai o texto das páginas do PDF
            if not texto_paginas:  # Verifica se algum texto foi extraído
                st.error("Nenhum texto extraído do PDF.")  # Exibe uma mensagem de erro se nenhum texto foi extraído
                return pd.DataFrame()  # Retorna um DataFrame vazio
            df = text_to_dataframe(texto_paginas)  # Converte o texto das páginas em um DataFrame
            if not df.empty:  # Verifica se o DataFrame não está vazio
                df.to_csv("references.csv", index=False)  # Salva o DataFrame como um arquivo CSV
                return df  # Retorna o DataFrame
            else:
                st.error("Nenhum texto extraído do PDF.")  # Exibe uma mensagem de erro se o DataFrame estiver vazio
                return pd.DataFrame()  # Retorna um DataFrame vazio
    except Exception as e:
        st.error(f"Erro ao carregar e extrair referências: {e}")  # Exibe uma mensagem de erro se ocorrer uma exceção
        return pd.DataFrame()  # Retorna um DataFrame vazio

# Função para obter o número máximo de tokens suportados por um modelo
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)  # Retorna o número máximo de tokens do modelo especificado, ou 4096 se o modelo não estiver no dicionário

# Função para registrar o uso da API
# Esta função salva detalhes sobre cada interação com a API em um arquivo JSON

def log_api_usage(action: str, interaction_number: int, tokens_used: int, time_taken: float, user_input: str, user_prompt: str, api_response: str, agent_used: str, agent_description: str):
    # Cria um dicionário com os detalhes da interação
    entry = {
        'action': action,  # Ação realizada (ex: "fetch", "refine", "evaluate")
        'interaction_number': interaction_number,  # Número da interação
        'tokens_used': tokens_used,  # Número de tokens usados na interação
        'time_taken': time_taken,  # Tempo gasto na interação
        'user_input': user_input,  # Entrada do usuário
        'user_prompt': user_prompt,  # Prompt fornecido pelo usuário
        'api_response': api_response,  # Resposta da API
        'agent_used': agent_used,  # Agente (especialista) utilizado
        'agent_description': agent_description  # Descrição do agente utilizado
    }
    
    # Verifica se o arquivo de uso da API já existe
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r+') as file:  # Abre o arquivo no modo de leitura e escrita
            api_usage = json.load(file)  # Carrega os dados existentes no arquivo
            api_usage.append(entry)  # Adiciona a nova entrada ao final da lista
            file.seek(0)  # Move o ponteiro do arquivo para o início
            json.dump(api_usage, file, indent=4)  # Salva os dados atualizados no arquivo com indentação de 4 espaços
    else:
        with open(API_USAGE_FILE, 'w') as file:  # Abre o arquivo no modo de escrita
            json.dump([entry], file, indent=4)  # Cria um novo arquivo com a entrada como uma lista de um elemento, com indentação de 4 espaços


# Função para lidar com a limitação de taxa da API
# Esta função verifica se a mensagem de erro indica que o limite de taxa foi excedido e, em caso afirmativo, aguarda o tempo necessário antes de tentar novamente

def handle_rate_limit(error_message: str, action: str):
    if 'rate_limit_exceeded' in error_message:  # Verifica se a mensagem de erro indica que o limite de taxa foi excedido
        # Usa expressão regular para encontrar o tempo de espera indicado na mensagem de erro
        wait_time = re.search(r'try again in (\d+\.?\d*)s', error_message)
        if wait_time:  # Se encontrar um tempo de espera na mensagem de erro
            wait_time = float(wait_time.group(1))  # Converte o tempo de espera para um número float
            st.warning(f"Limite de taxa atingido. Aguardando {wait_time} segundos...")  # Exibe uma mensagem de aviso no Streamlit
            time.sleep(wait_time)  # Aguarda o tempo especificado
        else:  # Se não encontrar um tempo de espera na mensagem de erro
            st.warning("Limite de taxa atingido. Aguardando 60 segundos...")  # Exibe uma mensagem de aviso no Streamlit
            time.sleep(60)  # Aguarda 60 segundos
        # Alterna para a próxima chave de API disponível para a ação especificada
        API_KEYS[action].append(API_KEYS[action].pop(0))
    else:  # Se a mensagem de erro não indicar limitação de taxa
        raise Exception(error_message)  # Levanta uma exceção com a mensagem de erro original

# Função para salvar o histórico de chat
# Esta função adiciona uma nova entrada ao histórico de chat e salva em um arquivo JSON

def save_chat_history(user_input, user_prompt, expert_response, chat_history_file=CHAT_HISTORY_FILE):
    # Cria um dicionário com os detalhes da entrada de chat
    chat_entry = {
        'user_input': user_input,  # Entrada do usuário
        'user_prompt': user_prompt,  # Prompt fornecido pelo usuário
        'expert_response': expert_response  # Resposta do especialista
    }
    
    # Verifica se o arquivo de histórico de chat já existe
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r+') as file:  # Abre o arquivo no modo de leitura e escrita
            chat_history = json.load(file)  # Carrega o histórico de chat existente no arquivo
            chat_history.append(chat_entry)  # Adiciona a nova entrada ao final da lista
            file.seek(0)  # Move o ponteiro do arquivo para o início
            json.dump(chat_history, file, indent=4)  # Salva o histórico de chat atualizado no arquivo com indentação de 4 espaços
    else:
        with open(chat_history_file, 'w') as file:  # Abre o arquivo no modo de escrita
            json.dump([chat_entry], file, indent=4)  # Cria um novo arquivo com a entrada de chat como uma lista de um elemento, com indentação de 4 espaços

# Função para carregar o histórico de chat a partir de um arquivo JSON
def load_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):  # Verifica se o arquivo de histórico de chat existe
        with open(chat_history_file, 'r') as file:  # Abre o arquivo no modo de leitura
            chat_history = json.load(file)  # Carrega o histórico de chat do arquivo
        return chat_history  # Retorna o histórico de chat
    return []  # Se o arquivo não existir, retorna uma lista vazia

# Função para limpar (deletar) o histórico de chat
def clear_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):  # Verifica se o arquivo de histórico de chat existe
        os.remove(chat_history_file)  # Deleta o arquivo de histórico de chat

# Função para carregar o uso da API a partir de um arquivo JSON
def load_api_usage():
    if os.path.exists(API_USAGE_FILE):  # Verifica se o arquivo de uso da API existe
        with open(API_USAGE_FILE, 'r') as file:  # Abre o arquivo no modo de leitura
            api_usage = json.load(file)  # Carrega o uso da API do arquivo
        return api_usage  # Retorna o uso da API
    return []  # Se o arquivo não existir, retorna uma lista vazia

# Função para plotar o uso da API
# Esta função cria gráficos que mostram o uso de tokens e o tempo gasto por diferentes chamadas de API

def plot_api_usage(api_usage):
    df = pd.DataFrame(api_usage)  # Converte o uso da API em um DataFrame

    if 'action' not in df.columns:  # Verifica se a coluna 'action' está presente no DataFrame
        st.error("A coluna 'action' não foi encontrada no dataframe de uso da API.")  # Exibe uma mensagem de erro no Streamlit se a coluna não for encontrada
        return  # Sai da função

    if 'agent_description' in df.columns:  # Verifica se a coluna 'agent_description' está presente no DataFrame
        df['agent_description'] = df['agent_description'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))  # Converte a descrição do agente em string, se for um dicionário

    # Cria uma figura com dois subplots (um em cima do outro)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plota o uso de tokens por diferentes ações da API no primeiro subplot
    sns.histplot(df[df['action'] == 'fetch']['tokens_used'], bins=20, color='blue', label='Fetch', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'refine']['tokens_used'], bins=20, color='green', label='Refine', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['tokens_used'], bins=20, color='red', label='Evaluate', ax=ax1, kde=True)
    ax1.set_title('Uso de Tokens por Chamada de API')  # Define o título do gráfico
    ax1.set_xlabel('Tokens')  # Define o rótulo do eixo x
    ax1.set_ylabel('Frequência')  # Define o rótulo do eixo y
    ax1.legend()  # Adiciona uma legenda ao gráfico

    # Plota o tempo gasto por diferentes ações da API no segundo subplot
    sns.histplot(df[df['action'] == 'fetch']['time_taken'], bins=20, color='blue', label='Fetch', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'refine']['time_taken'], bins=20, color='green', label='Refine', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['time_taken'], bins=20, color='red', label='Evaluate', ax=ax2, kde=True)
    ax2.set_title('Tempo por Chamada de API')  # Define o título do gráfico
    ax2.set_xlabel('Tempo (s)')  # Define o rótulo do eixo x
    ax2.set_ylabel('Frequência')  # Define o rótulo do eixo y
    ax2.legend()  # Adiciona uma legenda ao gráfico

    st.sidebar.pyplot(fig)  # Exibe a figura com os gráficos na barra lateral do Streamlit

    # Exibe o DataFrame com o uso da API na barra lateral do Streamlit
    st.sidebar.markdown("### Uso da API - DataFrame")
    st.sidebar.dataframe(df)


# Função para resetar (limpar) os dados de uso da API
# Esta função remove o arquivo que armazena os dados de uso da API

def reset_api_usage():
    if os.path.exists(API_USAGE_FILE):  # Verifica se o arquivo de uso da API existe
        os.remove(API_USAGE_FILE)  # Remove o arquivo de uso da API
    st.success("Os dados de uso da API foram resetados.")  # Exibe uma mensagem de sucesso no Streamlit informando que os dados foram resetados


# Função para buscar a resposta do assistente
# Esta função usa um modelo de linguagem para gerar uma resposta com base na entrada do usuário e no prompt fornecido

def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, chat_history: list, interaction_number: int, references_df: pd.DataFrame = None) -> Tuple[str, str]:
    phase_two_response = ""  # Inicializa a variável para a resposta da fase dois
    expert_title = ""  # Inicializa o título do especialista
    expert_description = ""  # Inicializa a descrição do especialista
    
    try:
        client = Groq(api_key=get_api_key('fetch'))  # Cria um cliente da API Groq com a chave de API para a ação 'fetch'

        # Função interna para obter a conclusão do modelo de linguagem
        def get_completion(prompt: str) -> str:
            start_time = time.time()  # Registra o tempo de início
            backoff_time = 1  # Tempo inicial de espera em segundos
            while True:  # Loop para tentar novamente em caso de limitação de taxa
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
                    end_time = time.time()  # Registra o tempo de término
                    tokens_used = completion.usage.total_tokens  # Obtém o número de tokens usados
                    time_taken = end_time - start_time  # Calcula o tempo gasto
                    api_response = completion.choices[0].message.content if completion.choices else ""  # Obtém a resposta da API
                    log_api_usage('fetch', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, expert_description)  # Registra o uso da API
                    return api_response  # Retorna a resposta da API
                except Exception as e:
                    if "503" in str(e):  # Verifica se ocorreu um erro 503
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")  # Exibe uma mensagem de erro no Streamlit
                        return ""
                    handle_rate_limit(str(e), 'fetch')  # Lida com a limitação de taxa
                    backoff_time = min(backoff_time * 2, 64)  # Aumenta o tempo de espera para a próxima tentativa, com um máximo de 64 segundos
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")  # Exibe uma mensagem de aviso no Streamlit
                    time.sleep(backoff_time)  # Aguarda o tempo de espera

        # Se o usuário não escolheu um especialista específico
        if agent_selection == "Escolher um especialista...":
            phase_one_prompt = (
                f"Descreva o especialista ideal para responder a seguinte solicitação: {user_input} e {user_prompt}."
            )
            phase_one_response = get_completion(phase_one_prompt)  # Obtém a resposta da fase um
            first_period_index = phase_one_response.find(".")  # Encontra o primeiro ponto na resposta
            if first_period_index != -1:
                expert_title = phase_one_response[:first_period_index].strip()  # Extrai o título do especialista
                expert_description = phase_one_response[first_period_index + 1:].strip()  # Extrai a descrição do especialista
                save_expert(expert_title, expert_description)  # Salva o especialista gerado
            else:
                st.error("Erro ao extrair título e descrição do especialista.")  # Exibe uma mensagem de erro no Streamlit
        else:
            # Se o usuário escolheu um especialista específico
            if os.path.exists(FILEPATH):  # Verifica se o arquivo de especialistas existe
                with open(FILEPATH, 'r') as file:
                    agents = json.load(file)  # Carrega os especialistas do arquivo
                    agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)  # Encontra o especialista selecionado
                    if agent_found:
                        expert_title = agent_found["agente"]  # Obtém o título do especialista
                        expert_description = agent_found["descricao"]  # Obtém a descrição do especialista
                    else:
                        raise ValueError("Especialista selecionado não encontrado no arquivo.")  # Levanta um erro se o especialista não for encontrado
            else:
                raise FileNotFoundError(f"Arquivo {FILEPATH} não encontrado.")  # Levanta um erro se o arquivo não for encontrado

        # Cria o contexto do histórico de chat
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Cria o contexto das referências
        references_context = ""
        if references_df is not None:
            for index, row in references_df.iterrows():
                titulo = row.get('titulo', 'Título Desconhecido')
                autor = row.get('autor', 'Autor Desconhecido')
                ano = row.get('ano', 'Ano Desconhecido')
                paginas = row.get('Page', 'Página Desconhecida')
                references_context += f"Título: {titulo}\nAutor: {autor}\nAno: {ano}\nPáginas: {paginas}\n\n"

        # Cria o prompt da fase dois
        phase_two_prompt = (
            f"{expert_title}, responda a seguinte solicitação de forma completa e detalhada: {user_input} e {user_prompt}."
            f"\n\nHistórico do chat:{history_context}"
            f"\n\nReferências:\n{references_context}"
        )
        phase_two_response = get_completion(phase_two_prompt)  # Obtém a resposta da fase dois

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")  # Exibe uma mensagem de erro no Streamlit
        return "", ""

    return expert_title, phase_two_response  # Retorna o título do especialista e a resposta da fase dois


# Função para refinar a resposta
# Esta função usa um modelo de linguagem para refinar a resposta gerada anteriormente com base na entrada do usuário, no prompt e no contexto fornecido

def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, references_context: str, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_api_key('refine'))  # Cria um cliente da API Groq com a chave de API para a ação 'refine'

        # Função interna para obter a conclusão do modelo de linguagem
        def get_completion(prompt: str) -> str:
            start_time = time.time()  # Registra o tempo de início
            backoff_time = 1  # Tempo inicial de espera em segundos
            while True:  # Loop para tentar novamente em caso de limitação de taxa
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
                    end_time = time.time()  # Registra o tempo de término
                    tokens_used = completion.usage.total_tokens  # Obtém o número de tokens usados
                    time_taken = end_time - start_time  # Calcula o tempo gasto
                    api_response = completion.choices[0].message.content if completion.choices else ""  # Obtém a resposta da API
                    log_api_usage('refine', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, "")  # Registra o uso da API
                    return api_response  # Retorna a resposta da API
                except Exception as e:
                    if "503" in str(e):  # Verifica se ocorreu um erro 503
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")  # Exibe uma mensagem de erro no Streamlit
                        return ""
                    handle_rate_limit(str(e), 'refine')  # Lida com a limitação de taxa
                    backoff_time = min(backoff_time * 2, 64)  # Aumenta o tempo de espera para a próxima tentativa, com um máximo de 64 segundos
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")  # Exibe uma mensagem de aviso no Streamlit
                    time.sleep(backoff_time)  # Aguarda o tempo de espera

        # Cria o contexto do histórico de chat
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Cria o prompt de refinamento
        refine_prompt = (
            f"{expert_title}, refine a seguinte resposta: {phase_two_response}. Solicitação original: {user_input} e {user_prompt}."
            f"\n\nHistórico do chat:{history_context}"
            f"\n\nReferências:\n{references_context}"
        )

        # Adiciona uma mensagem caso não haja referências fornecidas
        if not references_context:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas, certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas."
            )

        refined_response = get_completion(refine_prompt)  # Obtém a resposta refinada
        return refined_response  # Retorna a resposta refinada

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")  # Exibe uma mensagem de erro no Streamlit
        return ""

# Função para avaliar a resposta com RAG (Rational Agent Generator)
# Esta função usa um modelo de linguagem para avaliar a resposta gerada anteriormente, fornecendo sugestões de melhorias detalhadas

def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_title: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_api_key('evaluate'))  # Cria um cliente da API Groq com a chave de API para a ação 'evaluate'

        # Função interna para obter a conclusão do modelo de linguagem
        def get_completion(prompt: str) -> str:
            start_time = time.time()  # Registra o tempo de início
            backoff_time = 1  # Tempo inicial de espera em segundos
            while True:  # Loop para tentar novamente em caso de limitação de taxa
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
                    end_time = time.time()  # Registra o tempo de término
                    tokens_used = completion.usage.total_tokens  # Obtém o número de tokens usados
                    time_taken = end_time - start_time  # Calcula o tempo gasto
                    api_response = completion.choices[0].message.content if completion.choices else ""  # Obtém a resposta da API
                    log_api_usage('evaluate', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, expert_description)  # Registra o uso da API
                    return api_response  # Retorna a resposta da API
                except Exception as e:
                    if "503" in str(e):  # Verifica se ocorreu um erro 503
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")  # Exibe uma mensagem de erro no Streamlit
                        return ""
                    handle_rate_limit(str(e), 'evaluate')  # Lida com a limitação de taxa
                    backoff_time = min(backoff_time * 2, 64)  # Aumenta o tempo de espera para a próxima tentativa, com um máximo de 64 segundos
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")  # Exibe uma mensagem de aviso no Streamlit
                    time.sleep(backoff_time)  # Aguarda o tempo de espera

        # Cria o contexto do histórico de chat
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Cria o prompt para a avaliação com RAG
        rag_prompt = (
            f"{expert_title}, por favor, avalie a seguinte resposta: {assistant_response}. Solicitação original: {user_input} e {user_prompt}."
            f"\n\nHistórico do chat:{history_context}"
            f"\n\nDescreva detalhadamente as melhorias possíveis na resposta fornecida."
        )

        rag_response = get_completion(rag_prompt)  # Obtém a resposta da avaliação com RAG
        return rag_response  # Retorna a resposta da avaliação com RAG

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")  # Exibe uma mensagem de erro no Streamlit
        return ""


# Função para salvar um novo especialista
# Esta função adiciona um novo especialista (agente) ao arquivo JSON de especialistas

def save_expert(expert_title: str, expert_description: str):
    # Cria um dicionário com o título e a descrição do novo especialista
    new_expert = {
        "agente": expert_title,  # Título do especialista
        "descricao": expert_description  # Descrição do especialista
    }
    
    # Verifica se o arquivo de especialistas já existe
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r+') as file:  # Abre o arquivo no modo de leitura e escrita
            agents = json.load(file)  # Carrega os especialistas existentes no arquivo
            agents.append(new_expert)  # Adiciona o novo especialista à lista
            file.seek(0)  # Move o ponteiro do arquivo para o início
            json.dump(agents, file, indent=4)  # Salva os especialistas atualizados no arquivo com indentação de 4 espaços
    else:
        with open(FILEPATH, 'w') as file:  # Abre o arquivo no modo de escrita
            json.dump([new_expert], file, indent=4)  # Cria um novo arquivo com o novo especialista como uma lista de um elemento, com indentação de 4 espaços

# Interface Principal com Streamlit

# Inicializa variáveis de sessão no Streamlit
# Isso garante que as variáveis sejam preservadas entre interações do usuário
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

# Carrega as opções de agentes (especialistas)
agent_options = load_agent_options()

# Configurações de layout da página
st.image('updating.gif', width=300, caption='Consultor de PDFs + IA', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Consultor de PDFs</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Utilize nossa plataforma para consultas detalhadas em PDFs.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Seleção da quantidade de interações para lembrar no histórico
memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100])

# Caixa de texto para entrada do usuário
st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")
col1, col2 = st.columns(2)  # Divide a página em duas colunas

with col1:
    # Campos de entrada para o usuário
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    interaction_number = len(load_api_usage()) + 1

    # Botões para ações
    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar")

    # Upload de arquivo de referências (opcional)
    references_file = st.file_uploader("Upload do arquivo JSON ou PDF com referências (opcional)", type=["json", "pdf"], key="arquivo_referencias")

with col2:
    container_saida = st.container()  # Contêiner para saída de resultados

    # Carrega o histórico de chat com base na seleção de memória
    chat_history = load_chat_history()[-memory_selection:]

    # Lógica para quando o botão "Buscar Resposta" é clicado
    if fetch_clicked:
        if references_file:  # Se um arquivo de referências foi carregado
            df = upload_and_extract_references(references_file)  # Faz upload e extrai referências do arquivo
            if isinstance(df, pd.DataFrame):  # Verifica se o resultado é um DataFrame
                st.write("### Dados Extraídos do PDF")
                st.dataframe(df)
                st.session_state.references_path = "references.csv"
                st.session_state.references_df = df

        # Busca a resposta do assistente
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, chat_history, interaction_number, st.session_state.get('references_df'))
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""
        save_chat_history(user_input, user_prompt, st.session_state.resposta_assistente)  # Salva o histórico de chat

    # Lógica para quando o botão "Refinar Resposta" é clicado
    if refine_clicked:
        if st.session_state.resposta_assistente:  # Verifica se há uma resposta do assistente
            references_context = ""
            if not st.session_state.references_df.empty:  # Verifica se há referências
                for index, row in st.session_state.references_df.iterrows():
                    titulo = row.get('titulo', row['Text'][:50] + '...')
                    autor = row.get('autor', 'Autor Desconhecido')
                    ano = row.get('ano', 'Ano Desconhecido')
                    paginas = row.get('Page', 'Página Desconhecida')
                    references_context += f"Título: {titulo}\nAutor: {autor}\nAno: {ano}\nPágina: {paginas}\n\n"
            # Refina a resposta do assistente
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, references_context, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.resposta_refinada)  # Salva o histórico de chat
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")  # Exibe um aviso no Streamlit

    # Lógica para quando o botão "Avaliar Resposta com RAG" é clicado
    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:  # Verifica se há uma resposta do assistente e uma descrição do especialista
            # Avalia a resposta do assistente com RAG
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.rag_resposta)  # Salva o histórico de chat
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")  # Exibe um aviso no Streamlit

    # Exibe as respostas e análises na interface
    with container_saida:
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")
        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")

    # Exibe o histórico do chat em abas numeradas
    st.markdown("### Histórico do Chat")
    if chat_history:
        tab_titles = [f"Interação {i+1}" for i in range(len(chat_history))]  # Cria títulos de abas numeradas
        tabs = st.tabs(tab_titles)  # Cria abas com os títulos

        for i, entry in enumerate(chat_history):  # Itera sobre cada entrada no histórico de chat
            with tabs[i]:  # Cria uma aba para cada entrada
                st.write(f"**Entrada do Usuário:** {entry['user_input']}")
                st.write(f"**Prompt do Usuário:** {entry['user_prompt']}")
                st.write(f"**Resposta do Especialista:** {entry['expert_response']}")
                st.markdown("---")

# Lógica para quando o botão "Apagar" é clicado
if refresh_clicked:
    clear_chat_history()  # Limpa o histórico de chat
    st.session_state.clear()  # Limpa o estado da sessão
    st.rerun()  # Recarrega a aplicação

# Adiciona a imagem do logotipo na barra lateral
st.sidebar.image("logo.png", width=200)

# Expansível na barra lateral para mostrar os insights do código
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

    # Adiciona uma imagem pessoal e informações de contato na barra lateral
    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Consultor de PDFs + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.consultorpdfs/](https://www.instagram.com/marceloclaro.consultorpdfs/)
    """)

# Carrega o uso da API e plota o histograma se houver dados
api_usage = load_api_usage()
if api_usage:
    plot_api_usage(api_usage)

# Botão na barra lateral para resetar os gráficos de uso da API
if st.sidebar.button("Resetar Gráficos"):
    reset_api_usage()

# Função para carregar referências de um arquivo CSV, se existir
def carregar_referencias():
    if os.path.exists('references.csv'):
        return pd.read_csv('references.csv')
    else:
        return pd.DataFrame()

# Função para transformar referências em histórico de chat
def referencias_para_historico(df_referencias, chat_history_file=CHAT_HISTORY_FILE):
    if not df_referencias.empty:
        for _, row in df_referencias.iterrows():
            titulo = row.get('titulo', row['Text'][:50] + '...')  # Pega o título ou parte do texto como título
            autor = row.get('autor', 'Autor Desconhecido')  # Pega o autor ou coloca "Autor Desconhecido"
            ano = row.get('ano', 'Ano Desconhecido')  # Pega o ano ou coloca "Ano Desconhecido"
            paginas = row.get('Page', 'Página Desconhecida')  # Pega a página ou coloca "Página Desconhecida"
            
            # Cria uma entrada de chat para cada referência
            chat_entry = {
                'user_input': f"Título: {titulo}",
                'user_prompt': f"Autor: {autor}\nAno: {ano}\nPágina: {paginas}\nTexto: {row['Text']}",
                'expert_response': 'Informação adicionada ao histórico de chat como referência.'
            }
            
            # Adiciona a entrada de chat ao arquivo de histórico de chat
            if os.path.exists(chat_history_file):
                with open(chat_history_file, 'r+') as file:
                    chat_history = json.load(file)
                    chat_history.append(chat_entry)
                    file.seek(0)
                    json.dump(chat_history, file, indent=4)
            else:
                with open(chat_history_file, 'w') as file:
                    json.dump([chat_entry], file, indent=4)

# Carrega as referências do CSV e as transforma em histórico de chat
df_referencias = carregar_referencias()
referencias_para_historico(df_referencias)

