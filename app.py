import json
import streamlit as st
from typing import Tuple
from groq import Groq
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import CSVSearchTool, JSONSearchTool
import os
import subprocess
import sys

# Verificar a versão do SQLite
def check_sqlite_version():
    try:
        sqlite_version = subprocess.check_output(['sqlite3', '--version']).decode('utf-8').strip()
        if '3.35.0' not in sqlite_version:
            raise RuntimeError("Unsupported SQLite version. Chroma requires SQLite >= 3.35.0.")
    except Exception as e:
        print(f"Error checking SQLite version: {e}")
        raise

# Instalar SQLite se necessário
def install_sqlite():
    try:
        subprocess.run("wget https://www.sqlite.org/2024/sqlite-autoconf-3400000.tar.gz", shell=True, check=True)
        subprocess.run("tar xvfz sqlite-autoconf-3400000.tar.gz", shell=True, check=True)
        os.chdir("sqlite-autoconf-3400000")
        subprocess.run("./configure --prefix=/usr/local", shell=True, check=True)
        subprocess.run("make", shell=True, check=True)
        subprocess.run("sudo make install", shell=True, check=True)
        os.chdir("..")
        print("SQLite installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing SQLite: {e}")
        sys.exit(1)

# Checar e instalar SQLite se necessário
try:
    check_sqlite_version()
except RuntimeError:
    install_sqlite()
    check_sqlite_version()

# Adicionar o novo caminho ao PATH
os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]

# Importar dependências após garantir que a versão correta do SQLite está instalada
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import CSVSearchTool, JSONSearchTool

# Configurações de API
API_KEY = st.text_input("Insira sua chave API da Groq:", type="password")

# Configuração do modelo e API
def get_llm(api_key: str, model_name: str):
    return ChatOpenAI(model=model_name, api_key=api_key)

def get_max_tokens(model_name: str):
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Função para buscar a resposta do assistente
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, api_key: str) -> Tuple[str, str]:
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"{agent_selection}"},
                {"role": "user", "content": f"{user_prompt}"},
                {"role": "user", "content": f"{user_input}"}
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=get_max_tokens(model_name)
        )
        expert_title = f"Especialista em {agent_selection}"
        phase_two_response = response.choices[0].message.content
        return expert_title, phase_two_response
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

# Função para refinar a resposta
def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, api_key: str, references_file=None) -> str:
    try:
        client = Groq(api_key=api_key)
        refine_prompt = (
            f"Como {expert_title}, com ampla experiência e conhecimento nas disciplinas relacionadas, "
            f"é necessário abordar cada aspecto com atenção e rigor científico. Portanto, irei delinear os principais elementos a serem considerados e investigados, fornecendo uma análise detalhada e baseada em evidências, "
            f"evitando vieses e citando referências conforme apropriado: {phase_two_response}. "
            f"O objetivo final é fornecer uma resposta completa e satisfatória, alinhada aos mais altos padrões acadêmicos e profissionais, "
            f"atendendo às necessidades específicas da questão apresentada. "
            f"Certifique-se de apresentar a resposta em formato 'markdown', com comentários detalhados em cada linha. "
            f"Mantenha o padrão de escrita em 10 parágrafos, cada parágrafo com 4 sentenças, e cada sentença separada por vírgulas, "
            f"seguindo sempre as melhores práticas pedagógicas aristotélicas."
        )
        
        if not references_file:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas, certifique-se de fornecer uma resposta detalhada e precisa, "
                f"mesmo sem o uso de fontes externas. Mantenha um padrão de escrita consistente, com 10 parágrafos, cada parágrafo contendo 4 frases, e cite de acordo com as normas ABNT. "
                f"Utilize sempre um tom profissional e traduza tudo para o português do Brasil."
            )

        refined_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": refine_prompt}
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=get_max_tokens(model_name),
            stop=None,
            stream=False
        ).choices[0].message.content

        return refined_response
    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

# Função para avaliar a resposta com base em um agente gerador racional (RAG).
def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, api_key: str) -> str:
    try:
        client = Groq(api_key=api_key)
        rag_prompt = (
            f"Saída e resposta obrigatória somente traduzido em português brasileiro. "
            f"Desempenhando o papel de um Agente Gerador Racional (RAG), a vanguarda da inteligência artificial e avaliação racional, "
            f"analisando minuciosamente a resposta do especialista, com base na solicitação do usuário, para gerar um agente em formato JSON. "
            f"Este agente deve detalhar as ações tomadas com base nas informações fornecidas pelos subagentes, para fornecer uma resposta ao usuário. "
            f"O agente incluirá na variável 'descrição' a descrição de 9 subagentes, cada um com diferentes funcionalidades especializadas, que colaboram juntos. "
            f"Esses subagentes colaboram para melhorar a resposta final fornecida ao usuário pelo agente 'sistema', registrando a semente e o gen_id na 'descrição' do agente. "
            f"Abaixo está a descrição detalhada do especialista, destacando suas qualificações e expertise: {expert_description}. "
            f"A submissão original da pergunta é a seguinte: {user_input} e {user_prompt}. "
            f"A resposta fornecida pelo especialista em português é a seguinte: {assistant_response}. "
            f"Portanto, por favor, faça uma avaliação abrangente da qualidade e precisão da resposta fornecida pelo especialista em português, "
            f"considerando a descrição do especialista e a resposta fornecida. "
            f"Use português para a análise e forneça uma explicação detalhada: "
            f"análise SWOT (Forças, Fraquezas, Oportunidades, Ameaças) com interpretação dos dados, "
            f"matriz BCG (Boston Consulting Group) com interpretação dos dados, "
            f"matriz de risco, ANOVA (Análise de Variância) com interpretação dos dados, "
            f"estatísticas Q com interpretação dos dados e índice Q (Q-Exponential) com interpretação dos dados, "
            f"seguindo os mais altos padrões de excelência e rigor acadêmico e científico. "
            f"Certifique-se de manter cada parágrafo com 4 sentenças, cada sentença separada por vírgulas, seguindo sempre as melhores práticas pedagógicas aristotélicas. "
            f"A resposta deve ser em português do Brasil."
        )

        rag_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": rag_prompt}
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=get_max_tokens(model_name),
            stop=None,
            stream=False
        ).choices[0].message.content

        return rag_response
    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Função para buscar no CSV e JSON.
def search_csv_and_json(csv_file, json_file, query):
    csv_results = None
    json_results = None

    if csv_file:
        csv_search_tool = CSVSearchTool(csv_file)
        csv_results = csv_search_tool.search(query)
    
    if json_file:
        json_search_tool = JSONSearchTool(json_file)
        json_results = json_search_tool.search(query)

    return csv_results, json_results

# Função para exibir os resultados da busca.
def display_search_results(csv_results, json_results):
    if csv_results:
        st.write("### Resultados da busca no CSV")
        st.write(csv_results)
    
    if json_results:
        st.write("### Resultados da busca no JSON")
        st.write(json_results)

# Função para criar e executar a equipe de agentes e tarefas.
def execute_team(phase_two_response, refined_response, rag_response, model_name, api_key):
    client = Groq(api_key=api_key)
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    
    # Criação de agentes
    agent_phase_two = Agent(name="Phase Two Agent", description=phase_two_response, model=llm)
    agent_refined = Agent(name="Refined Agent", description=refined_response, model=llm)
    agent_rag = Agent(name="RAG Agent", description=rag_response, model=llm)
    
    # Definição das tarefas
    task_phase_two = Task(description="Executar a resposta da fase dois.", agent=agent_phase_two)
    task_refined = Task(description="Refinar a resposta.", agent=agent_refined)
    task_rag = Task(description="Avaliar a resposta refinada com RAG.", agent=agent_rag)
    
    # Criação da equipe
    crew = Crew(agents=[agent_phase_two, agent_refined, agent_rag], tasks=[task_phase_two, task_refined, task_rag])
    
    # Execução das tarefas
    result = crew.execute()
    
    return result

# Interface do Streamlit
st.markdown("# Assistente Acadêmico Avançado")
st.markdown("## Sua ferramenta para excelência acadêmica")

st.markdown("### Funcionalidades:")
st.write("1. **Assistente avançado:** Obtenha respostas detalhadas e precisas para suas perguntas acadêmicas.")
st.write("2. **Refinamento de respostas:** Melhore a qualidade das respostas fornecidas.")
st.write("3. **Busca em arquivos CSV e JSON:** Encontre informações relevantes em seus arquivos.")
st.write("4. **Tecnologia de ponta:** Utilize as mais recentes inovações em IA para aprimorar sua experiência educacional.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Escolha o modelo e o especialista:</h2>", unsafe_allow_html=True)

user_input = st.text_input("Digite sua pergunta:")
user_prompt = st.text_area("Adicione informações adicionais (opcional):")
model_name = st.selectbox("Escolha o modelo de linguagem:", list(MODEL_MAX_TOKENS.keys()))
temperature = st.slider("Escolha a temperatura do modelo:", min_value=0.0, max_value=1.0, value=0.5)
agent_selection = st.selectbox("Escolha o especialista:", load_agent_options())

uploaded_files = st.file_uploader("Faça upload dos arquivos CSV e JSON para busca (opcional):", accept_multiple_files=True)
csv_file = None
json_file = None

for uploaded_file in uploaded_files:
    if uploaded_file.name.endswith(".csv"):
        csv_file = uploaded_file
    elif uploaded_file.name.endswith(".json"):
        json_file = uploaded_file

if st.button("Obter Resposta"):
    if not user_input or not model_name or not API_KEY:
        st.error("Por favor, preencha todos os campos obrigatórios.")
    else:
        expert_title, phase_two_response = fetch_assistant_response(
            user_input, user_prompt, model_name, temperature, agent_selection, API_KEY
        )
        refined_response = refine_response(
            expert_title, phase_two_response, user_input, user_prompt, model_name, temperature, API_KEY, json_file
        )
        rag_response = evaluate_response_with_rag(
            user_input, user_prompt, expert_title, phase_two_response, model_name, temperature, API_KEY
        )

        result = execute_team(phase_two_response, refined_response, rag_response, model_name, API_KEY)
        st.success("Resposta gerada com sucesso!")
        st.markdown(result)

if st.button("Buscar no CSV e JSON"):
    if csv_file or json_file:
        query = st.text_input("Digite sua consulta:")
        if query:
            csv_results, json_results = search_csv_and_json(csv_file, json_file, query)
            display_search_results(csv_results, json_results)
        else:
            st.error("Por favor, digite uma consulta.")
    else:
        st.error("Por favor, faça o upload de pelo menos um arquivo CSV ou JSON.")
