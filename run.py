import json
import streamlit as st
import os
from typing import Tuple, List
from groq import Groq
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configura o layout da página Streamlit para ser "wide", ocupando toda a largura disponível.
st.set_page_config(layout="wide")

# Define o caminho para o arquivo JSON que contém os Agentes.
FILEPATH = "agents.json"
CHAT_HISTORY_FILE = 'chat_history.json'
API_USAGE_FILE = 'api_usage.json'  # Arquivo para armazenar o uso da API

# Define um dicionário que mapeia nomes de modelos para o número máximo de tokens que cada modelo suporta.
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# API keys
API_KEYS = {
    "fetch": ["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", "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"],
    "refine": ["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", "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"],
    "evaluate": ["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", "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"]
}

# Função para alternar a chave da API em caso de limite de taxa
def get_api_key(action: str) -> str:
    keys = API_KEYS[action]
    for key in keys:
        return key
    return keys[0]  # fallback para a primeira chave se todas estiverem no limite

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

def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

def handle_rate_limit(error_message: str, action: str):
    if 'rate_limit_exceeded' in error_message:
        wait_time = float(error_message.split("try again in")[1].split("s.")[0].strip())
        st.warning(f"Limite de taxa atingido. Aguardando {wait_time} segundos...")
        time.sleep(wait_time)
        return True
    return False

def refresh_page():
    st.experimental_rerun()

def save_expert(expert_title: str, expert_description: dict):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

def log_api_usage(action: str, interaction_number: int, tokens_used: int, time_taken: float, user_input: str, user_prompt: str, api_response: str):
    entry = {
        'action': action,
        'interaction_number': interaction_number,
        'tokens_used': tokens_used,
        'time_taken': time_taken,
        'user_input': user_input,
        'user_prompt': user_prompt,
        'api_response': api_response
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

def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, chat_history: list, interaction_number: int) -> Tuple[str, str]:
    phase_two_response = ""
    expert_title = ""

    try:
        client = Groq(api_key=get_api_key("fetch"))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
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
                    log_api_usage('fetch', interaction_number, tokens_used, time_taken, user_input, user_prompt, completion.choices[0].message.content)
                    return completion.choices[0].message.content
                except Exception as e:
                    if not handle_rate_limit(str(e), "fetch"):
                        raise

        if agent_selection == "Escolher um especialista...":
            phase_one_prompt = (
                f"假设自己是一位具有高度科学严谨性的高级提示工程专家。"
                f"请以‘markdown’格式呈现Python代码及其相应的库，并在每行中添加详细的教育性注释。"
                f"仔细分析所提出的要求，确定最适合处理该问题的专家特征的标准。"
                f"确定后，详细描述该专家的主要技能和资格，避免偏见。"
                f"介绍能够处理所提出问题的特征和资格：{user_input}和{user_prompt}。"
                f"准确度为10.0，符合最高的专业、科学和学术标准。"
                f"对于涉及代码和计算的情况，请以 'markdown' 格式呈现，并在每行中添加详细的注释。"
                f"回答必须仅用葡萄牙语。"
                f"假设你是高级工程方面的专家，并且具有高度的科学严谨性。"
                f"请以‘markdown’格式提供 Python 代码及其相应的库。"
                f"在该行中添加详细的教育说明。"
                f"仔细分析所提出的要求，以确定最适合处理问题的专家的特征的标准。"
                f"首先，有必要确定最能反映需要提供完整、深入和明确答案的答案。"
                f"需要经验的头衔。"
                f"一旦确定，请详细描述专家的关键技能和资格，以避免偏见。"
                f"就这样，然后从清晰的、有教育意义的、深入的描述开始。"
                f"介绍允许他们处理所提出问题的特征和资格：{user_input} 和 {user_prompt}。"
                f"这种详细的分析对于确保所选专家拥有必要的知识和经验来提供完整且令人满意的答复至关重要。"
                f"准确度为10.0，符合最高的专业、科学和学术标准，每一行都有详细的注释。"
            )

            phase_one_response = get_completion(phase_one_prompt)
            first_period_index = phase_one_response.find(".")
            expert_title = phase_one_response[:first_period_index].strip()
            expert_description = phase_one_response[first_period_index + 1:].strip()
            save_expert(expert_title, expert_description)
        else:
            with open(FILEPATH, 'r') as file:
                agents = json.load(file)
                agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                if agent_found:
                    expert_title = agent_found["agente"]
                    expert_description = agent_found["descricao"]
                else:
                    raise ValueError("Especialista selecionado não encontrado no arquivo.")

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        phase_two_prompt = (
            f"输出和响应必须仅翻译成巴西葡萄牙语。"
            f"扮演{expert_title}的角色，这是一位在其领域内广受认可和尊敬的专家，"
            f"作为该领域的博士和专家，提供一个全面且深入的回答，涵盖问题的各个方面，做到清晰、详细、扩展、"
            f"教育性和简洁：{user_input}和{user_prompt}。"
            f"考虑到我在相关学科的丰富经验和深厚知识，"
            f"有必要以科学严谨的态度关注并探讨每个方面。"
            f"因此，我将概述需要考虑和调查的主要要素，提供基于证据的详细分析，"
            f"避免偏见，并根据需要引用参考文献：{user_prompt}。"
            f"最终目标是提供一个完整且令人满意的回答，符合最高的学术和 profissional标准，"
            f"满足所提出问题的具体需求。"
            f"确保以'markdown'格式呈现回答，并在每行中添加详细注释。"
            f"保持写作标准在10个段落，每个段落4句话，每句话用逗号分隔，"
            f"始终遵循亚里士多德的最佳教育实践。"
            f"\n\nHistórico do chat:{history_context}"
        )
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, references_file: str, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_api_key("refine"))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
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
                    log_api_usage('refine', interaction_number, tokens_used, time_taken, user_input, user_prompt, completion.choices[0].message.content)
                    return completion.choices[0].message.content
                except Exception as e:
                    if not handle_rate_limit(str(e), "refine"):
                        raise

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        refine_prompt = (
            f"输出和响应必须仅翻译成巴西葡萄牙语。"
            f"扮演{expert_title}的角色，这是一位在其领域内广受认可和尊敬的专家，"
            f"作为该领域的博士和专家，提供一个全面且深入的回答，涵盖问题的各个方面，做到清晰、详细、扩展、"
            f"教育性和简洁：{user_input}和{user_prompt}。"
            f"考虑到我在相关学科的丰富经验和深厚知识，"
            f"有必要以科学严谨的态度关注并探讨每个方面。"
            f"因此，我将概述需要考虑和调查的主要要素，提供基于证据的详细分析，"
            f"避免偏见，并根据需要引用参考文献：{phase_two_response}。"
            f"最终目标是提供一个完整且令人满意的回答，符合最高的 acadêmico e profissional padrão，"
            f"满足所提出问题的具体需求。"
            f"确保以'markdown'格式呈现回答，并在每行中添加详细注释。"
            f"保持写作 padrão em 10 parágrafos，每个 parágrafo contendo 4 frases, "
            f"e sempre seguindo as melhores práticas educacionais de Aristóteles。"
            f"\n\nHistórico do chat:{history_context}"
        )

        if not references_file:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas, certifique-se de fornecer uma resposta detalhada e precisa, "
                f"mesmo sem o uso de fontes externas. "
                f"Mantenha um padrão de escrita consistente, com 10 parágrafos, cada parágrafo contendo 4 frases, e cite de acordo com as normas ABNT. "
                f"Utilize sempre um tom profissional e traduza tudo para o português do Brasil."
            )

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_api_key("evaluate"))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
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
                    log_api_usage('evaluate', interaction_number, tokens_used, time_taken, user_input, user_prompt, completion.choices[0].message.content)
                    return completion.choices[0].message.content
                except Exception as e:
                    if not handle_rate_limit(str(e), "evaluate"):
                        raise

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        rag_prompt = (
            f"输出和响应必须仅翻译成巴西葡萄牙语。"
            f"扮演一个理性生成代理（RAG）的角色，站在人工智能和理性评估的前沿，"
            f"仔细分析专家的回答，根据用户的请求生成一个JSON格式的代理。"
            f"该代理应详细描述根据子代理提供的信息采取的行动，以向用户提供回答。"
            f"在变量'描述'中包含9个子代理的描述，每个子代理具有不同的专门功能，共同协作。"
            f"这些子代理共同协作，以改善系统代理向用户提供的最终回答，并在代理的'描述'中记录种子和gen_id。"
            f"此外，系统代理内的子代理一体化操作，通过扩展提示提供高级和专业的回答。"
            f"每个子代理在网络处理过程中扮演特定且互补的角色，以达到更高的精度，提升最终回答的质量。"
            f"例如，子代理'AI_自适应_和_上下文化'使用先进的机器学习算法来理解和适应变化的上下文，"
            f"动态整合相关数据。而子代理'RAG_与_上下文智能'使用增强生成的恢复（RAG）技术，"
            f"动态调整最相关的数据及其功能。 这种协作方法确保回答准确且更新，"
            f"符合最高的科学和学术标准。"
            f"以下是专家的详细描述，突出其资格和专业知识：{expert_description}。"
            f"原始问题提交如下：{user_input}和{user_prompt}。"
            f"专家用葡萄牙语提供的回答如下：{assistant_response}。"
            f"因此，请对专家用葡萄牙语提供的回答的质量和准确性进行全面评估，"
            f"考虑专家的描述和提供的回答。"
            f"使用葡萄牙语进行分析并提供详细解释："
            f"SWOT分析（优势、劣势、机会、威胁）并解释数据，"
            f"BCG矩阵（波士顿咨询集团）并解释数据，"
            f"风险矩阵，ANOVA（方差分析）并解释数据，"
            f"Q统计并解释数据和Q指数（Q-指数）并解释数据，"
            f"遵循最高的卓越和学术、科学严格标准。"
            f"确保每段保持4句话，每句话用逗号分隔，始终遵循亚里士多德和苏格拉底的最佳教育实践。"
            f"回答必须使用巴西葡萄牙语。"
            f"\n\nHistórico do chat:{history_context}"
        )

        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

def save_chat_history(user_input, user_prompt, expert_response, chat_history_file=CHAT_HISTORY_FILE):
    chat_entry = {
        'user_input': user_input,
        'user_prompt': user_prompt,
        'expert_response': expert_response
    }
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r+') as file:
            chat_history = json.load(file)
            chat_history.append(chat_entry)
            file.seek(0)
            json.dump(chat_history, file, indent=4)
    else:
        with open(chat_history_file, 'w') as file:
            json.dump([chat_entry], file, indent=4)

def load_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r') as file:
            chat_history = json.load(file)
        return chat_history
    return []

def clear_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        os.remove(chat_history_file)

def load_api_usage():
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r') as file:
            api_usage = json.load(file)
        return api_usage
    return []

def plot_api_usage(api_usage):
    df = pd.DataFrame(api_usage)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    sns.histplot(df[df['action'] == 'fetch']['tokens_used'], bins=20, color='blue', label='Fetch', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'refine']['tokens_used'], bins=20, color='green', label='Refine', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['tokens_used'], bins=20, color='red', label='Evaluate', ax=ax1, kde=True)
    ax1.set_title('Uso de Tokens por Chamada de API')
    ax1.set_xlabel('Tokens')
    ax1.set_ylabel('Frequência')
    ax1.legend()

    sns.histplot(df[df['action'] == 'fetch']['time_taken'], bins=20, color='blue', label='Fetch', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'refine']['time_taken'], bins=20, color='green', label='Refine', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['time_taken'], bins=20, color='red', label='Evaluate', ax=ax2, kde=True)
    ax2.set_title('Tempo por Chamada de API')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Frequência')
    ax2.legend()

    st.sidebar.pyplot(fig)

def reset_api_usage():
    if os.path.exists(API_USAGE_FILE):
        os.remove(API_USAGE_FILE)
    st.success("Os dados de uso da API foram resetados.")

# Carrega as opções de Agentes a partir do arquivo JSON.
agent_options = load_agent_options()

# Layout da página
st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes Alan Kay</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

with st.expander("Clique para saber mais sobre os Agentes Alan Kay."):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")
    st.image("fluxograma agente 4.png")

memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100])

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    max_tokens = get_max_tokens(model_name)
    st.write(f"Número Máximo de Tokens para o modelo selecionado: {max_tokens}")

    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar")

    references_file = st.file_uploader("Upload do arquivo JSON com referências (opcional)", type="json", key="arquivo_referencias")

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

    interaction_number = len(chat_history) + 1

    if fetch_clicked:
        if references_file is None:
            st.warning("Não foi fornecido um arquivo de referências. Certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas. Saída sempre traduzido para o portugues brasileiro com tom profissional.")
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
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, chat_history, interaction_number)
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
    st.experimental_rerun()

st.sidebar.image("logo.png", width=200)
with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do Agentes Alan Kay é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

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

api_usage = load_api_usage()
if api_usage:
    plot_api_usage(api_usage)

if st.sidebar.button("Resetar Gráficos"):
    reset_api_usage()
