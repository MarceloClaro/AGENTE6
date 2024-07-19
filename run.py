import json
import streamlit as st
import os
from typing import Tuple, List
from groq import Groq

# Configure the Streamlit page layout
st.set_page_config(layout="wide")

# Define the path for the agents JSON file
FILEPATH = "agents.json"

# Define the maximum tokens for each model
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192, 
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Load agent options from JSON file
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de Agentes. Verifique o formato.")
    return agent_options

# Get the maximum tokens for a given model
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Save a new expert to the JSON file
def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

# Initialize session state for memory and history
if 'history' not in st.session_state:
    st.session_state.history = []

if 'history_limit' not in st.session_state:
    st.session_state.history_limit = 5

# Function to add interaction to history
def add_to_history(user_input: str, assistant_response: str):
    st.session_state.history.append({"user_input": user_input, "assistant_response": assistant_response})
    if len(st.session_state.history) > st.session_state.history_limit:
        st.session_state.history.pop(0)

# Function to fetch the full history
def get_full_history() -> str:
    history = st.session_state.history
    full_history = ""
    for interaction in history:
        full_history += f"Usuário: {interaction['user_input']}\nAssistente: {interaction['assistant_response']}\n\n"
    return full_history

# Fetch assistant response from Groq API
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, groq_api_key: str) -> Tuple[str, str]:
    phase_two_response = ""
    expert_title = ""

    try:
        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
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
            return completion.choices[0].message.content

        if agent_selection == "Escolher um especialista...":
            phase_one_prompt = (
                f"Você é um especialista em {user_input}. "
                f"Descreva suas habilidades e como você ajudaria com {user_prompt}."
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

        full_history = get_full_history()
        phase_two_prompt = (
            f"Como {expert_title}, responda detalhadamente a: {user_input} e {user_prompt}.\n"
            f"Histórico:\n{full_history}"
        )
        phase_two_response = get_completion(phase_two_prompt)
        add_to_history(user_input, phase_two_response)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

# Refine the assistant's response
def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, groq_api_key: str, references_file: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
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
            return completion.choices[0].message.content

        full_history = get_full_history()
        refine_prompt = (
            f"Como {expert_title}, refine sua resposta anterior a: {phase_two_response} com base em {user_input} e {user_prompt}.\n"
            f"Histórico:\n{full_history}"
        )
        if references_file:
            refine_prompt += f"\nReferências: {references_file.read().decode('utf-8')}"

        refined_response = get_completion(refine_prompt)
        add_to_history(user_input, refined_response)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

# Evaluate the response with Rational Agent Generator (RAG)
def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, groq_api_key: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
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
            return completion.choices[0].message.content

        full_history = get_full_history()
        rag_prompt = (
            f"Avalie a resposta fornecida por {expert_description} para {user_input} e {user_prompt}. "
            f"Resposta: {assistant_response}\n"
            f"Histórico:\n{full_history}"
        )
        rag_response = get_completion(rag_prompt)
        add_to_history(user_input, rag_response)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Load agent options
agent_options = load_agent_options()

# Streamlit UI
st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes 4  - Alan Kay</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

with st.expander("Clique para saber mais sobre os Agentes 4  - Alan Kay."):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")
    st.image("fluxograma agente 4.png")
    st.markdown("### Explicando o Fluxograma de Utilização\n"
                "Vamos detalhar o processo de utilização da plataforma, conforme representado no fluxograma, passo a passo. A plataforma é projetada para ajudar usuários a interagir com modelos de linguagem avançados através de uma interface intuitiva. Vamos explorar cada etapa para que você possa aproveitar ao máximo todas as funcionalidades oferecidas.\n"
                "#### Passo a Passo do Fluxograma\n"
                "1. **Início**:\n"
                "   - O fluxo começa com o início da execução do aplicativo.\n"
                "2. **Importar bibliotecas**:\n"
                "   - Importação de todas as bibliotecas necessárias, incluindo `streamlit`, `json`, e outras bibliotecas essenciais para o funcionamento da aplicação.\n"
                "3. **Configurar página**:\n"
                "   - Configuração inicial da página usando `streamlit` para definir o layout e outras propriedades da página.\n"
                "4. **Verificar existência de agents.json**:\n"
                "   - O aplicativo verifica se o arquivo `agents.json` existe no diretório. Este arquivo contém informações sobre os Agentes 4  - disponíveis.\n"
                "5. **agents.json existe?**:\n"
                "   - Decisão condicional:\n"
                "     - **Sim**:\n"
                "       - Carregar Agentes 4  -: O arquivo `agents.json` é carregado.\n"
                "       - **Erro ao carregar JSON?**:\n"
                "         - **Sim**: Exibir mensagem de erro: O aplicativo mostra uma mensagem de erro indicando problemas ao carregar o arquivo JSON.\n"
                "         - **Não**: Mostrar opções de Agentes 4  -: As opções de Agentes 4  - carregadas são exibidas.\n"
                "     - **Não**: Usar opções de Agentes 4  - padrão: Se o arquivo não for encontrado, o aplicativo usa opções de Agentes 4  - padrão.\n"
                "6. **Mostrar opções de Agentes 4  -**:\n"
                "   - O aplicativo exibe as opções de Agentes 4  - para o usuário escolher.\n"
                "7. **Selecionar modelo e agente**:\n"
                "   - O usuário seleciona o modelo de linguagem e o agente desejado.\n"
                "8. **Obter tokens máximos para o modelo**:\n"
                "   - O aplicativo obtém o número máximo de tokens permitidos para o modelo selecionado.\n"
                "9. **Solicitar resposta do assistente**:\n"
                "   - O usuário solicita uma resposta do assistente.\n"
                "10. **Agente selecionado?**:\n"
                "    - Decisão condicional:\n"
                "      - **Não**: Solicitar seleção de agente: O aplicativo pede ao usuário que selecione um agente.\n"
                "      - **Sim**: Criar cliente Groq: O cliente para interação com a API Groq é criado.\n"
                "        - Gerar resposta do assistente: O aplicativo gera a resposta do assistente usando o modelo de linguagem selecionado.\n"
                "        - Retornar resposta do assistente: A resposta gerada é retornada pelo modelo.\n"
                "        - Exibir resposta do assistente: A resposta é exibida ao usuário.\n"
                "11. **Solicitar refinamento da resposta**:\n"
                "    - O usuário pode solicitar um refinamento da resposta fornecida.\n"
                "12. **Refinar resposta do assistente**:\n"
                "    - O assistente refina a resposta com base nas novas informações ou correções fornecidas pelo usuário.\n"
                "13. **Retornar resposta refinada**:\n"
                "    - A resposta refinada é retornada pelo modelo.\n"
                "14. **Exibir resposta refinada**:\n"
                "    - A resposta refinada é exibida ao usuário.\n"
                "15. **Solicitar avaliação RAG**:\n"
                "    - O usuário pode solicitar uma avaliação da resposta usando o Rational Agent Generator (RAG).\n"
                "16. **Avaliar resposta com RAG**:\n"
                "    - O assistente avalia a resposta utilizando RAG.\n"
                "17. **Retornar avaliação RAG**:\n"
                "    - A avaliação do RAG é retornada pelo modelo.\n"
                "18. **Exibir avaliação RAG**:\n"
                "    - A avaliação do RAG é exibida ao usuário.\n"
                "19. **Fim**:\n"
                "    - O fluxo termina.\n"
                "### Conclusão\n"
                "Este fluxograma detalha de maneira clara e organizada todas as etapas que o usuário deve seguir para utilizar a plataforma de maneira eficaz. Com este guia, esperamos que você possa explorar todas as funcionalidades disponíveis e tirar o máximo proveito das ferramentas avançadas de interação com modelos de linguagem.\n"
                "Aproveite para experimentar, fazer perguntas complexas e refinar suas interações para obter as respostas mais precisas e relevantes possíveis.\n")

st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("Clique para saber mais sobre o Rational Agent Generator (RAG)"):
    st.info("""
    O Rational Agent Generator (RAG) é usado para avaliar a resposta fornecida pelo especialista. Aqui está uma explicação mais detalhada de como ele é usado:
    
    1. Quando o usuário busca uma resposta do especialista, a função `fetch_assistant_response()` é chamada. Nessa função, é gerado um prompt para o modelo de linguagem que representa a solicitação do usuário e o prompt específico para o especialista escolhido. A resposta inicial do especialista é então obtida usando o Groq API.
    
    2. Se o usuário optar por refinar a resposta, a função `refine_response()` é chamada. Nessa função, é gerado um novo prompt que inclui a resposta inicial do especialista e solicita uma resposta mais detalhada e aprimorada, levando em consideração as referências fornecidas pelo usuário. A resposta refinada é obtida usando novamente o Groq API.
    
    3. Se o usuário optar por avaliar a resposta com o RAG, a função `evaluate_response_with_rag()` é chamada. Nessa função, é gerado um prompt que inclui a descrição do especialista e as respostas inicial e refinada do especialista. O RAG é então usado para avaliar a qualidade e a precisão da resposta do especialista.
    
    Em resumo, o RAG é usado como uma ferramenta para avaliar e melhorar a qualidade das respostas fornecidas pelos especialistas, garantindo que atendam aos mais altos padrões de excelência e rigor científico.
    """)
    st.image("diagram agente 4.png")

    st.markdown("### Explicando o Fluxograma de Funcionamento\n"
                "Neste post, vamos detalhar o processo de utilização da plataforma, conforme representado no fluxograma, passo a passo. A plataforma é projetada para ajudar usuários a interagir com modelos de linguagem avançados através de uma interface intuitiva. Vamos explorar cada etapa para que você possa aproveitar ao máximo todas as funcionalidades oferecidas.\n"
                "#### Passo a Passo do Fluxograma de funcionamento\n"
                "1. **Início**:\n"
                "   - O fluxo começa com o início da execução do aplicativo.\n"
                "2. **Importar bibliotecas**:\n"
                "   - Importação de todas as bibliotecas necessárias, incluindo `streamlit`, `json`, e outras bibliotecas essenciais para o funcionamento da aplicação.\n"
                "3. **Configurar página**:\n"
                "   - Configuração inicial da página usando `streamlit` para definir o layout e outras propriedades da página.\n"
                "4. **Verificar existência de agents.json**:\n"
                "   - O aplicativo verifica se o arquivo `agents.json` existe no diretório. Este arquivo contém informações sobre os Agentes 4  - disponíveis.\n"
                "5. **agents.json existe?**:\n"
                "   - Decisão condicional:\n"
                "     - **Sim**:\n"
                "       - Carregar Agentes 4  -: O arquivo `agents.json` é carregado.\n"
                "       - **Erro ao carregar JSON?**:\n"
                "         - **Sim**: Exibir mensagem de erro: O aplicativo mostra uma mensagem de erro indicando problemas ao carregar o arquivo JSON.\n"
                "         - **Não**: Mostrar opções de Agentes 4  -: As opções de Agentes 4  - carregadas são exibidas.\n"
                "     - **Não**: Usar opções de Agentes 4  - padrão: Se o arquivo não for encontrado, o aplicativo usa opções de Agentes 4  - padrão.\n"
                "6. **Mostrar opções de Agentes 4  -**:\n"
                "   - O aplicativo exibe as opções de Agentes 4  - para o usuário escolher.\n"
                "7. **Selecionar modelo e agente**:\n"
                "   - O usuário seleciona o modelo de linguagem e o agente desejado.\n"
                "8. **Obter tokens máximos para o modelo**:\n"
                "   - O aplicativo obtém o número máximo de tokens permitidos para o modelo selecionado.\n"
                "9. **Solicitar resposta do assistente**:\n"
                "   - O usuário solicita uma resposta do assistente.\n"
                "10. **Agente selecionado?**:\n"
                "    - Decisão condicional:\n"
                "      - **Não**: Solicitar seleção de agente: O aplicativo pede ao usuário que selecione um agente.\n"
                "      - **Sim**: Criar cliente Groq: O cliente para interação com a API Groq é criado.\n"
                "        - Gerar resposta do assistente: O aplicativo gera a resposta do assistente usando o modelo de linguagem selecionado.\n"
                "        - Retornar resposta do assistente: A resposta gerada é retornada pelo modelo.\n"
                "        - Exibir resposta do assistente: A resposta é exibida ao usuário.\n"
                "11. **Solicitar refinamento da resposta**:\n"
                "    - O usuário pode solicitar um refinamento da resposta fornecida.\n"
                "12. **Refinar resposta do assistente**:\n"
                "    - O assistente refina a resposta com base nas novas informações ou correções fornecidas pelo usuário.\n"
                "13. **Retornar resposta refinada**:\n"
                "    - A resposta refinada é retornada pelo modelo.\n"
                "14. **Exibir resposta refinada**:\n"
                "    - A resposta refinada é exibida ao usuário.\n"
                "15. **Solicitar avaliação RAG**:\n"
                "    - O usuário pode solicitar uma avaliação da resposta usando o Rational Agent Generator (RAG).\n"
                "16. **Avaliar resposta com RAG**:\n"
                "    - O assistente avalia a resposta utilizando RAG.\n"
                "17. **Retornar avaliação RAG**:\n"
                "    - A avaliação do RAG é retornada pelo modelo.\n"
                "18. **Exibir avaliação RAG**:\n"
                "    - A avaliação do RAG é exibida ao usuário.\n"
                "19. **Fim**:\n"
                "    - O fluxo termina.\n"
                "### Conclusão\n"
                "Este fluxograma detalha de maneira clara e organizada todas as etapas que o usuário deve seguir para utilizar a plataforma de maneira eficaz. Com este guia, esperamos que você possa explorar todas as funcionalidades disponíveis e tirar o máximo proveito das ferramentas avançadas de interação com modelos de linguagem.\n"
                "Aproveite para experimentar, fazer perguntas complexas e refinar suas interações para obter as respostas mais precisas e relevantes possíveis.\n")

st.markdown("<hr>", unsafe_allow_html=True)

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    groq_api_key = st.text_input("Chave da API Groq: Você pode usar esse como teste - ... ", key="groq_api_key")
    max_tokens = get_max_tokens(model_name)
    st.write(f"Número Máximo de Tokens para o modelo selecionado: {max_tokens}")
    
    history_limit = st.selectbox("Número de Interações a Armazenar", options=[5, 10, 30, 60], index=0, key="history_limit")
    st.session_state.history_limit = history_limit

    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar Histórico")

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

    if fetch_clicked:
        if references_file is None:
            st.warning("Não foi fornecido um arquivo de referências. Certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas. Saída sempre traduzido para o portugues brasileiro com tom profissional.")
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key)
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""

    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, groq_api_key, references_file)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, groq_api_key)
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")

    with container_saida:
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")
        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")

if refresh_clicked:
    st.session_state.history.clear()
    st.experimental_rerun()

st.sidebar.image("logo.png", width=200)

with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do Agentes 4  - Alan Kay é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

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

with st.sidebar.expander("Análise de Expertise do Código"):
    st.markdown("""
    ### Análise de Expertise do Código

    O código fornecido implementa um sistema de chat interativo com especialistas usando a biblioteca Streamlit para a interface de usuário e um modelo de linguagem baseado em API para gerar respostas. A seguir está uma análise detalhada da expertise refletida no código, considerando diferentes aspectos do desenvolvimento de chats com modelos de linguagem (LLMs).

    #### Pontos Positivos

    1. **Configuração da Interface de Usuário**:
       - Uso adequado do Streamlit para criar uma interface web interativa, permitindo a seleção de especialistas e a entrada de consultas pelo usuário.
       - Boa estruturação visual com o uso de `st.markdown` e `st.expander` para apresentar informações de forma organizada e acessível.

    2. **Gestão de Arquivos e Dados**:
       - Carregamento e armazenamento de dados JSON (`agents.json`) para manter informações sobre os especialistas, utilizando boas práticas de manuseio de arquivos.
       - Tratamento de exceções ao carregar dados JSON, com mensagens de erro amigáveis (`json.JSONDecodeError`).

    3. **Integração com API Externa**:
       - Uso da biblioteca `groq` para interagir com uma API de modelo de linguagem, incluindo a configuração de chaves API e parâmetros de consulta.
       - Funções específicas para obter respostas do modelo (`fetch_assistant_response`, `refine_response`, `evaluate_response_with_rag`), demonstrando uma boa modularidade.

    4. **Flexibilidade na Escolha de Modelos**:
       - Inclusão de um dicionário `MODEL_MAX_TOKENS` para definir limites de tokens para diferentes modelos, permitindo flexibilidade na escolha de modelos com diferentes capacidades.
       - Interface de seleção para escolher entre diferentes modelos de linguagem, ajustando dinamicamente os parâmetros (`max_tokens`, `temperature`).

    5. **Funcionalidades de Refinamento e Avaliação**:
       - Implementação de um mecanismo para refinar respostas, permitindo uma análise mais profunda e a melhoria da precisão das respostas geradas.
       - Uso de um sistema de avaliação com Rational Agent Generator (RAG) para assegurar a qualidade e precisão das respostas, incluindo diversas técnicas de análise (SWOT, ANOVA, Q-Statistics).

    #### Pontos a Melhorar

    1. **Segurança e Validação de Entrada**:
       - Falta de sanitização das entradas do usuário, o que pode levar a vulnerabilidades como injeção de código ou dados maliciosos.
       - As chaves da API são inseridas diretamente no código, o que pode não ser seguro. Sugere-se o uso de variáveis de ambiente ou mecanismos seguros de armazenamento.

    2. **Gestão de Sessões e Estado**:
       - Uso de variáveis de sessão (`st.session_state`) para manter o estado da resposta do assistente e outras informações, mas a implementação poderia ser mais robusta para evitar perda de dados entre interações.

    3. **Documentação e Comentários**:
       - O código se beneficiaria de comentários mais detalhados e documentação para melhorar a legibilidade e a manutenção futura.
       - A inclusão de exemplos de uso e uma descrição mais detalhada das funções principais ajudaria outros desenvolvedores a entender melhor o fluxo do código.

    4. **Eficiência e Desempenho**:
       - Dependendo do tamanho dos arquivos JSON e da quantidade de dados processados, a leitura e escrita de arquivos podem se tornar um gargalo. Considere otimizações como a leitura parcial ou o uso de uma base de dados.

    ### Nota Final de Expertise

    Baseando-se nos pontos destacados, a expertise no desenvolvimento deste código pode ser avaliada como alta, especialmente considerando a integração de diferentes componentes (interface, API de modelo de linguagem, gerenciamento de dados) e a implementação de funcionalidades avançadas de avaliação e refinamento de respostas.

    **Nota: 9.5/10**

    Esta avaliação reflete um bom equilíbrio entre funcionalidade, usabilidade e boas práticas de desenvolvimento, com algumas áreas para melhorias em termos de segurança, documentação e eficiência.
    """)

import base64

def main():
    st.sidebar.title("Controle de Áudio")
    mp3_files = {
        "Agente Alan Kay": "AGENTE-4AlanKay1.mp3",
        "Agente 4": "agente4.mp3",
        "Agente Alan-Kay": "AGENTEAlan-Kay.mp3",
        "Instrumental": "ambienteindia.mp3"
    }
    selected_mp3 = st.sidebar.radio("Escolha uma música", list(mp3_files.keys()))
    loop = st.sidebar.checkbox("Repetir música")

    audio_placeholder = st.sidebar.empty()
    if selected_mp3:
        mp3_path = mp3_files[selected_mp3]
        try:
            with open(mp3_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                loop_attr = "loop" if loop else ""
                audio_html = f"""
                <audio id="audio-player" controls autoplay {loop_attr}>
                  <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                  Seu navegador não suporta o elemento de áudio.
                </audio>
                """
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
        except FileNotFoundError:
            audio_placeholder.error(f"Arquivo {mp3_path} não encontrado.")

    st.sidebar.write("""
        Código principal do Agentes Alan Kay
    """)
    try:
        with open("runBR.py", "r") as file:
            code = file.read()
            st.sidebar.code(code, language='python')
    except FileNotFoundError:
        st.sidebar.error("Arquivo runBR.py não encontrado.")

    st.sidebar.write("""
        Código dos Agentes contidos no arquivo agents.json
    """)
    try:
        with open("agentsBR.json", "r") as file:
            code = file.read()
            st.sidebar.code(code, language='json')
    except FileNotFoundError:
        st.sidebar.error("Arquivo agentsBR.json não encontrado.")

    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)

if __name__ == "__main__":
    main()
