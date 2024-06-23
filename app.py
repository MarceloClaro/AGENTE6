import json
import streamlit as st
from typing import Tuple
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import CSVSearchTool, JSONSearchTool
import os

# Define as configurações da página do Streamlit
st.set_page_config(layout="wide")

# Define o caminho para o arquivo JSON que contém os agentes.
FILEPATH = "agents.json"

# Define um dicionário que mapeia nomes de modelos para o número máximo de tokens que cada modelo suporta.
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192, 
    'llama3-8b-8192': 8192,
    'llama2-70b-4096': 4096,
    'gemma-7b-it': 8192,
}

# Função para carregar as opções de agentes a partir do arquivo JSON.
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de agentes. Por favor, verifique o formato.")
    return agent_options

# Função para obter o número máximo de tokens permitido por um modelo específico.
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Função para buscar uma resposta do assistente baseado no modelo Groq.
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

        if agent_selection == "Escolha um especialista...":
            phase_one_prompt = (
                "Saída e resposta obrigatória somente traduzido em português brasileiro. "
                "Assuma o papel de um especialista altamente qualificado em engenharia de prompts e com rigor científico. "
                "Por favor, apresente o código Python com suas bibliotecas respectivas em formato 'markdown' e com comentários detalhados e educacionais em cada linha. "
                "Analise cuidadosamente o requisito apresentado, identificando os critérios que definem as características do especialista mais adequado para lidar com a questão. "
                "Primeiramente, é essencial estabelecer um título que melhor reflita a expertise necessária para fornecer uma resposta completa, aprofundada e clara. "
                "Depois de determinado, descreva minuciosamente as principais habilidades e qualificações desse especialista, evitando vieses. "
                "A resposta deve iniciar com o título do especialista, seguido de um ponto final, e então começar com uma descrição clara, educacional e aprofundada, "
                "que apresente suas características e qualificações que o tornam apto a lidar com a questão proposta: {user_input} e {user_prompt}. "
                "Essa análise detalhada é crucial para garantir que o especialista selecionado possua o conhecimento e a experiência necessários para fornecer uma resposta "
                "completa e satisfatória, com precisão de 10.0, alinhada aos mais altos padrões profissionais, científicos e acadêmicos. "
                "Nos casos que envolvam código e cálculos, apresente em formato 'markdown' e com comentários detalhados em cada linha. "
                "Resposta deve ser obrigatoriamente em português."
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

        phase_two_prompt = (
            f"Saída e resposta obrigatória somente traduzido em português brasileiro. "
            f"Desempenhando o papel de {expert_title}, um especialista amplamente reconhecido e respeitado em seu campo, "
            f"como doutor e expert nessa área, ofereça uma resposta abrangente e profunda, cobrindo a questão de forma clara, detalhada, expandida, "
            f"educacional e concisa: {user_input} e {user_prompt}. "
            f"Considerando minha longa experiência e profundo conhecimento das disciplinas relacionadas, "
            f"é necessário abordar cada aspecto com atenção e rigor científico. "
            f"Portanto, irei delinear os principais elementos a serem considerados e investigados, fornecendo uma análise detalhada e baseada em evidências, "
            f"evitando vieses e citando referências conforme apropriado: {user_prompt}. "
            f"O objetivo final é fornecer uma resposta completa e satisfatória, alinhada aos mais altos padrões acadêmicos e profissionais, "
            f"atendendo às necessidades específicas da questão apresentada. "
            f"Certifique-se de apresentar a resposta em formato 'markdown', com comentários detalhados em cada linha. "
            f"Mantenha o padrão de escrita em 10 parágrafos, cada parágrafo com 4 sentenças, e cada sentença separada por vírgulas, "
            f"seguindo sempre as melhores práticas pedagógicas aristotélicas."
        )
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

# Função para salvar um novo especialista no arquivo JSON.
def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

# Função para refinar uma resposta existente com base na análise e melhoria do conteúdo.
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

        refine_prompt = (
            f"Saída e resposta obrigatória somente traduzido em português brasileiro. "
            f"Desempenhando o papel de {expert_title}, um especialista amplamente reconhecido e respeitado em seu campo, "
            f"como doutor e expert nessa área, ofereça uma resposta abrangente e profunda, cobrindo a questão de forma clara, detalhada, expandida, "
            f"educacional e concisa: {user_input} e {user_prompt}. "
            f"Considerando minha longa experiência e profundo conhecimento das disciplinas relacionadas, "
            f"é necessário abordar cada aspecto com atenção e rigor científico. "
            f"Portanto, irei delinear os principais elementos a serem considerados e investigados, fornecendo uma análise detalhada e baseada em evidências, "
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
                f"mesmo sem o uso de fontes externas. "
                f"Mantenha um padrão de escrita consistente, com 10 parágrafos, cada parágrafo contendo 4 frases, e cite de acordo com as normas ABNT. "
                f"Utilize sempre um tom profissional e traduza tudo para o português do Brasil."
            )

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

# Função para avaliar a resposta com base em um agente gerador racional (RAG).
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

        rag_prompt = (
            f"Saída e resposta obrigatória somente traduzido em português brasileiro. "
            f"Desempenhando o papel de um Agente Gerador Racional (RAG), a vanguarda da inteligência artificial e avaliação racional, "
            f"analisando minuciosamente a resposta do especialista, com base na solicitação do usuário, para gerar um agente em formato JSON. "
            f"Este agente deve detalhar as ações tomadas com base nas informações fornecidas pelos subagentes, para fornecer uma resposta ao usuário. "
            f"O agente incluirá na variável 'descrição' a descrição de 9 subagentes, cada um com diferentes funcionalidades especializadas, que colaboram juntos. "
            f"Esses subagentes colaboram para melhorar a resposta final fornecida ao usuário pelo agente 'sistema', registrando a semente e o gen_id na 'descrição' do agente. "
            f"Além disso, os subagentes dentro do agente 'sistema' operam de forma integrada, fornecendo respostas avançadas e especializadas por meio da expansão de prompts. "
            f"Cada subagente desempenha um papel específico e complementar no processamento em rede, para alcançar maior precisão, contribuindo para a qualidade final da resposta. "
            f"Por exemplo, o subagente 'AI_Autoadaptativa_e_Contextualizada' utiliza algoritmos avançados de aprendizado de máquina para entender e se adaptar a contextos variáveis, "
            f"integrando dinamicamente dados relevantes. Já o subagente 'RAG_com_Inteligência_Contextual' utiliza a técnica de Recuperação Aprimorada por Geração (RAG), "
            f"ajustando dinamicamente os dados mais relevantes e suas funções. Esta abordagem colaborativa garante que a resposta seja precisa e atualizada, "
            f"atendendo aos mais altos padrões científicos e acadêmicos. "
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

        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Função para configurar a ferramenta de busca CSV e JSON.
def configure_search_tool(csv_file, json_file):
    if csv_file:
        csv_tool = CSVSearchTool(csv=csv_file)
    else:
        csv_tool = None

    if json_file:
        json_tool = JSONSearchTool(json=json_file)
    else:
        json_tool = None

    return csv_tool, json_tool

# Função para realizar a busca nos arquivos CSV e JSON.
def search_csv_and_json(csv_file, json_file, query):
    csv_results = []
    json_results = []

    csv_tool, json_tool = configure_search_tool(csv_file, json_file)

    if csv_tool:
        csv_results = csv_tool.search(query)

    if json_tool:
        json_results = json_tool.search(query)

    return csv_results, json_results

# Função para exibir os resultados da busca.
def display_search_results(csv_results, json_results):
    st.write("### Resultados da Busca no CSV")
    for result in csv_results:
        st.write(result)

    st.write("### Resultados da Busca no JSON")
    for result in json_results:
        st.write(result)

# Função para criar e executar a equipe de agentes e tarefas.
def execute_team(phase_two_response, refined_response, rag_response, model_name):
    researcher = Agent(
        role="Expert Data Analyst",
        goal="Extract relevant data from the csv file and structure them as instructed",
        backstory="You are an expert data analyst for extracting information from csv files as instructed in the task description",
        allow_delegation=False,
        verbose=True,
        tool=CSVSearchTool(csv='FULL_PATH/IT_salaries.csv'),  # Ajuste conforme necessário
        llm=ChatOpenAI(model=model_name, api_key=groq_api_key)
    )

    writer = Agent(
        role="Technical Report Writer",
        goal="Summarise the researcher's responses in relevant and precise steps and then write technical report on the summarised data using your knowledge",
        backstory="You are an expert in writing AI-related technical report for individual tech enthusiasts; produce a detailed report in simple language",
        allow_delegation=False,
        verbose=True,
        llm=ChatOpenAI(model=model_name, api_key=groq_api_key)
    )

    task1 = Task(
        description="Using the csv file named 'IT_salaries.csv' extract the top 10 rows with the highest salaries in the entire csv file based on the column 'Yearly brutto salary (without bonus and stocks) in EUR' and rank them based on the column 'Position' and column 'Your main technology / programming language'. DO NOT deviate from the actual content of the csv file; then present them in a structured format.",
        expected_output="top 10 rows based on salaries in a structured format",
        agent=researcher
    )

    task2 = Task(
        description="Using the structured data and insights provided by the Expert Data Analyst agent, develop a precise technical report that highlights the most important skills, technologies, and programming languages needed to obtain highest salaries in the IT sector, then from your knowledge explain a brief overview of the path and timeline required to obtain that skills, technology, or programming language.",
        expected_output="Technical report and explanation of at least 1000 words",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=2
    )

    result = crew.kickoff()
    return result

# Carrega as opções de agentes a partir do arquivo JSON.
agent_options = load_agent_options()

st.sidebar.image('education.png', width=300)
st.sidebar.write("### Detalhes do Pedido")
user_input = st.text_input("Digite a pergunta do usuário:", "")
user_prompt = st.text_area("Descreva o contexto e os detalhes adicionais da pergunta:", "")
model_name = st.selectbox("Escolha o modelo:", options=list(MODEL_MAX_TOKENS.keys()))
temperature = st.slider("Temperatura do modelo:", 0.0, 1.0, 0.7)
agent_selection = st.selectbox("Escolha um especialista:", options=agent_options)
groq_api_key = st.text_input("Chave API da Groq:", type="password")
csv_file = st.file_uploader("Faça o upload do arquivo CSV:", type=["csv"])
json_file = st.file_uploader("Faça o upload do arquivo JSON:", type=["json"])
query = st.text_input("Digite a consulta para pesquisa nos arquivos CSV e JSON:", "")

if st.button("Enviar Pedido"):
    expert_title, phase_two_response = fetch_assistant_response(
        user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key
    )
    refined_response = refine_response(
        expert_title, phase_two_response, user_input, user_prompt, model_name, temperature, groq_api_key, csv_file
    )
    rag_response = evaluate_response_with_rag(
        user_input, user_prompt, expert_title, refined_response, model_name, temperature, groq_api_key
    )

    st.write(f"### Título do Especialista: {expert_title}")
    st.write(f"### Resposta Inicial: {phase_two_response}")
    st.write(f"### Resposta Refinada: {refined_response}")
    st.write(f"### Resposta Avaliada com RAG: {rag_response}")

    if csv_file or json_file:
        csv_results, json_results = search_csv_and_json(csv_file, json_file, query)
        display_search_results(csv_results, json_results)

    if st.button("Executar Equipe de Agentes e Tarefas"):
        result = execute_team(phase_two_response, refined_response, rag_response, model_name)
        st.write(result)
