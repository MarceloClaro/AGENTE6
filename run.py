import json  # Importa o módulo json para trabalhar com dados JSON.
import streamlit as st  # Importa o Streamlit para criar aplicativos web interativos.
import os  # Importa o módulo os para interagir com o sistema operacional, como verificar a existência de arquivos.
from typing import Tuple  # Importa Tuple da biblioteca typing para fornecer tipos de dados mais precisos para funções.
from groq import Groq  # Importa a biblioteca Groq, possivelmente para uma função não especificada neste código.
import time  # Importa o módulo time para adicionar atrasos entre as tentativas de solicitação da API

# Configura o layout da página Streamlit para ser "wide", ocupando toda a largura disponível.
st.set_page_config(layout="wide")

# Define o caminho para o arquivo JSON que contém os Agentes.
FILEPATH = "agents.json"
CHAT_HISTORY_FILE = 'chat_history.json'

# Define um dicionário que mapeia nomes de modelos para o número máximo de tokens que cada modelo suporta.
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# API keys
API_KEY_FETCH = "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"
API_KEY_REFINE = "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"
API_KEY_EVALUATE = "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"

# Define uma função para carregar as opções de Agentes a partir do arquivo JSON.
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']  # Inicia a lista de opções com uma opção padrão.
    if os.path.exists(FILEPATH):  # Verifica se o arquivo de Agentes existe.
        with open(FILEPATH, 'r') as file:  # Abre o arquivo para leitura.
            try:
                agents = json.load(file)  # Tenta carregar os dados JSON do arquivo.
                # Adiciona os nomes dos Agentes à lista de opções, se existirem.
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:  # Captura erros de decodificação JSON.
                st.error("Erro ao ler o arquivo de Agentes. Por favor, verifique o formato.")  # Exibe uma mensagem de erro no Streamlit.
    return agent_options  # Retorna a lista de opções de Agentes.

# Define uma função para obter o número máximo de tokens permitido por um modelo específico.
def get_max_tokens(model_name: str) -> int:
    # Retorna o número máximo de tokens para o modelo fornecido, ou 4096 se o modelo não estiver no dicionário.
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Define uma função para recarregar a página do Streamlit.
def refresh_page():
    st.experimental_rerun()  # Recarrega a aplicação Streamlit.

# Define uma função para salvar um novo especialista no arquivo JSON.
def save_expert(expert_title: str, expert_description: dict):
    with open(FILEPATH, 'r+') as file:  # Abre o arquivo para leitura e escrita.
        # Carrega os Agentes existentes se o arquivo não estiver vazio, caso contrário, inicia uma lista vazia.
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        # Adiciona o novo especialista à lista de Agentes.
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)  # Move o ponteiro do arquivo para o início.
        json.dump(agents, file, indent=4)  # Grava a lista de Agentes de volta no arquivo com indentação para melhor legibilidade.
        file.truncate()  # Remove qualquer conteúdo restante do arquivo após a nova escrita para evitar dados obsoletos.

# Função para buscar uma resposta do assistente baseado no modelo Groq, incluindo o histórico.
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, groq_api_key: str, chat_history: list) -> Tuple[str, str]:
    phase_two_response = ""  # Inicializa a variável para armazenar a resposta da segunda fase.
    expert_title = ""  # Inicializa a variável para armazenar o título do especialista.

    try:
        client = Groq(api_key=API_KEY_FETCH)  # Usa a chave API específica para buscar respostas.

        # Define uma função interna para obter a conclusão/completar um prompt usando a API Groq.
        def get_completion(prompt: str) -> str:
            while True:  # Loop para tentar novamente em caso de erro de limite de taxa
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil."},  # Mensagem do sistema definindo o comportamento do assistente.
                            {"role": "user", "content": prompt},  # Mensagem do usuário contendo o prompt.
                        ],
                        model=model_name,  # Nome do modelo a ser usado.
                        temperature=temperature,  # Temperatura para controlar a aleatoriedade das respostas.
                        max_tokens=get_max_tokens(model_name),  # Número máximo de tokens permitido para o modelo.
                        top_p=1,  # Parâmetro para amostragem nuclear.
                        stop=None,  # Sem tokens de parada específicos.
                        stream=False  # Desabilita o streaming de respostas.
                    )
                    return completion.choices[0].message.content  # Retorna o conteúdo da primeira escolha da resposta.
                except Exception as e:
                    error_message = str(e)
                    if 'rate_limit_exceeded' in error_message:
                        wait_time = float(error_message.split("try again in")[1].split("s.")[0].strip())
                        time.sleep(wait_time)  # Espera pelo tempo sugerido antes de tentar novamente
                    else:
                        raise e  # Se o erro não for de limite de taxa, relança a exceção

        if agent_selection == "Escolher um especialista...":
            # Se nenhum especialista específico for selecionado, cria um prompt para determinar o título e descrição do especialista.
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

            phase_one_response = get_completion(phase_one_prompt)  # Obtém a resposta para o prompt da fase um.
            first_period_index = phase_one_response.find(".")  # Encontra o índice do primeiro ponto na resposta.
            expert_title = phase_one_response[:first_period_index].strip()  # Extrai o título do especialista até o primeiro ponto.
            expert_description = phase_one_response[first_period_index + 1:].strip()  # Extrai a descrição do especialista após o primeiro ponto.
            save_expert(expert_title, expert_description)  # Salva o novo especialista no arquivo JSON.
        else:
            # Se um especialista específico for selecionado, carrega os dados do especialista do arquivo JSON.
            with open(FILEPATH, 'r') as file:  # Abre o arquivo JSON para leitura.
                agents = json.load(file)  # Carrega os dados dos Agentes do arquivo JSON.
                # Encontra o agente selecionado na lista de Agentes.
                agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                if agent_found:
                    expert_title = agent_found["agente"]  # Obtém o título do especialista.
                    expert_description = agent_found["descricao"]  # Obtém a descrição do especialista.
                else:
                    raise ValueError("Especialista selecionado não encontrado no arquivo.")  # Lança um erro se o especialista não for encontrado.

        # Formata o histórico do chat para incluir nas mensagens do prompt.
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Cria um prompt para a segunda fase, onde o especialista selecionado fornece uma resposta detalhada.
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
        phase_two_response = get_completion(phase_two_prompt)  # Obtém a resposta para o prompt da segunda fase.

    except Exception as e:  # Captura qualquer exceção que ocorra durante o processo.
        st.error(f"Ocorreu um erro: {e}")  # Exibe uma mensagem de erro no Streamlit.
        return "", ""  # Retorna tuplas vazias se ocorrer um erro.

    return expert_title, phase_two_response  # Retorna o título do especialista e a resposta da segunda fase.

# Função para refinar uma resposta existente com base na análise e melhoria do conteúdo.
def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, groq_api_key: str, references_file: str, chat_history: list) -> str:
    try:
        client = Groq(api_key=API_KEY_REFINE)  # Usa a chave API específica para refinar respostas.

        # Define uma função interna para obter a conclusão/completar um prompt usando a API Groq.
        def get_completion(prompt: str) -> str:
            while True:  # Loop para tentar novamente em caso de erro de limite de taxa
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil."},  # Mensagem do sistema definindo o comportamento do assistente.
                            {"role": "user", "content": prompt},  # Mensagem do usuário contendo o prompt.
                        ],
                        model=model_name,  # Nome do modelo a ser usado.
                        temperature=temperature,  # Temperatura para controlar a aleatoriedade das respostas.
                        max_tokens=get_max_tokens(model_name),  # Número máximo de tokens permitido para o modelo.
                        top_p=1,  # Parâmetro para amostragem nuclear.
                        stop=None,  # Sem tokens de parada específicos.
                        stream=False  # Desabilita o streaming de respostas.
                    )
                    return completion.choices[0].message.content  # Retorna o conteúdo da primeira escolha da resposta.
                except Exception as e:
                    error_message = str(e)
                    if 'rate_limit_exceeded' in error_message:
                        wait_time = float(error_message.split("try again in")[1].split("s.")[0].strip())
                        time.sleep(wait_time)  # Espera pelo tempo sugerido antes de tentar novamente
                    else:
                        raise e  # Se o erro não for de limite de taxa, relança a exceção

        # Formata o histórico do chat para incluir nas mensagens do prompt.
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Cria um prompt detalhado para refinar a resposta.
        refine_prompt = (
            f"输出和响应必须仅翻译成巴西葡萄牙语。"
            f"扮演{expert_title}的角色，这是一位在其领域内广受认可和尊敬的专家，"
            f"作为该领域的博士和专家，提供一个全面且深入的回答，涵盖问题的各个方面，做到清晰、详细、扩展、"
            f"教育性和简洁：{user_input}和{user_prompt}。"
            f"考虑到我在相关学科的丰富经验和深厚知识，"
            f"有必要以科学严谨的态度关注并探讨每个方面。"
            f"因此，我将概述需要考虑和调查的主要要素，提供基于证据的详细分析，"
            f"避免偏见，并根据需要引用参考文献：{phase_two_response}。"
            f"最终目标是提供一个完整且令人满意的回答，符合最高的学术 e profissional标准，"
            f"满足所提出问题的具体需求。"
            f"确保以'markdown'格式呈现回答，并在每行中添加详细注释。"
            f"保持写作标准在10个段落，每个段落4句话，每句话用逗号分隔，"
            f"始终遵循亚里士多德的最佳教育实践。"
            f"\n\nHistórico do chat:{history_context}"
        )

        # Adiciona um prompt mais detalhado se não houver referências fornecidas.
        if not references_file:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas, certifique-se de fornecer uma resposta detalhada e precisa, "
                f"mesmo sem o uso de fontes externas. "
                f"Mantenha um padrão de escrita consistente, com 10 parágrafos, cada parágrafo contendo 4 frases, e cite de acordo com as normas ABNT. "
                f"Utilize sempre um tom profissional e traduza tudo para o português do Brasil."
            )

        refined_response = get_completion(refine_prompt)  # Obtém a resposta refinada a partir do prompt detalhado.
        return refined_response  # Retorna a resposta refinada.

    except Exception as e:  # Captura qualquer exceção que ocorra durante o processo de refinamento.
        st.error(f"Ocorreu um erro durante o refinamento: {e}")  # Exibe uma mensagem de erro no Streamlit.
        return ""  # Retorna uma string vazia se ocorrer um erro.

# Função para avaliar a resposta com base em um agente gerador racional (RAG).
def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, groq_api_key: str, chat_history: list) -> str:
    try:
        client = Groq(api_key=API_KEY_EVALUATE)  # Usa a chave API específica para avaliar respostas.

        # Define uma função interna para obter a conclusão/completar um prompt usando a API Groq.
        def get_completion(prompt: str) -> str:
            while True:  # Loop para tentar novamente em caso de erro de limite de taxa
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil."},  # Mensagem do sistema definindo o comportamento do assistente.
                            {"role": "user", "content": prompt},  # Mensagem do usuário contendo o prompt.
                        ],
                        model=model_name,  # Nome do modelo a ser usado.
                        temperature=temperature,  # Temperatura para controlar a aleatoriedade das respostas.
                        max_tokens=get_max_tokens(model_name),  # Número máximo de tokens permitido para o modelo.
                        top_p=1,  # Parâmetro para amostragem nuclear.
                        stop=None,  # Sem tokens de parada específicos.
                        stream=False  # Desabilita o streaming de respostas.
                    )
                    return completion.choices[0].message.content  # Retorna o conteúdo da primeira escolha da resposta.
                except Exception as e:
                    error_message = str(e)
                    if 'rate_limit_exceeded' in error_message:
                        wait_time = float(error_message.split("try again in")[1].split("s.")[0].strip())
                        time.sleep(wait_time)  # Espera pelo tempo sugerido antes de tentar novamente
                    else:
                        raise e  # Se o erro não for de limite de taxa, relança a exceção

        # Formata o histórico do chat para incluir nas mensagens do prompt.
        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        # Cria um prompt detalhado para avaliar a resposta usando o agente gerador racional (RAG).
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

        rag_response = get_completion(rag_prompt)  # Obtém a resposta avaliada a partir do prompt detalhado.
        return rag_response  # Retorna a resposta avaliada.

    except Exception as e:  # Captura qualquer exceção que ocorra durante o processo de avaliação com RAG.
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")  # Exibe uma mensagem de erro no Streamlit.
        return ""  # Retorna uma string vazia se ocorrer um erro.

# Função para salvar o histórico de chat
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

# Função para carregar o histórico de chat
def load_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r') as file:
            chat_history = json.load(file)
        return chat_history
    return []

# Função para apagar o histórico de chat
def clear_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        os.remove(chat_history_file)

# Carrega as opções de Agentes a partir do arquivo JSON.
agent_options = load_agent_options()

# Layout da página
st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes Alan Kay</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

# Exibe informações sobre os Agentes Alan Kay
with st.expander("Clique para saber mais sobre os Agentes Alan Kay."):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")
    st.image("fluxograma agente 4.png")

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
    groq_api_key = st.text_input("Chave da API Groq:", key="groq_api_key")
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

    chat_history = load_chat_history()[-memory_selection:]  # Carrega as últimas 'memory_selection' interações

    if fetch_clicked:
        if references_file is None:
            st.warning("Não foi fornecido um arquivo de referências. Certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas. Saída sempre traduzido para o portugues brasileiro com tom profissional.")
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key, chat_history)
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""
        save_chat_history(user_input, user_prompt, st.session_state.resposta_assistente)

    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, groq_api_key, references_file, chat_history)
            save_chat_history(user_input, user_prompt, st.session_state.resposta_refinada)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, groq_api_key, chat_history)
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

    # Exibe o histórico do chat
    st.markdown("### Histórico do Chat")
    for entry in chat_history:
        st.write(f"**Entrada do Usuário:** {entry['user_input']}")
        st.write(f"**Prompt do Usuário:** {entry['user_prompt']}")
        st.write(f"**Resposta do Especialista:** {entry['expert_response']}")
        st.markdown("---")

if refresh_clicked:
    clear_chat_history()  # Limpa o arquivo de histórico de chat
    st.session_state.clear()  # Reseta o estado do Streamlit
    st.experimental_rerun()  # Recarrega a aplicação Streamlit

# Sidebar com manual de uso
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

