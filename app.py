import json
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import os
from typing import Tuple
from groq import Groq

# Configura o layout da página Streamlit para ser "wide"
st.set_page_config(layout="wide")

# Definir variáveis globais
phase_two_response = None
refined_response = None

# Define o caminho para o arquivo JSON que contém os agentes
FILEPATH = "agents.json"

# Define um dicionário que mapeia nomes de modelos para o número máximo de tokens que cada modelo suporta
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'llama2-70b-4096': 4096,
    'gemma-7b-it': 8192,
}

# Define uma função para carregar as opções de agentes a partir do arquivo JSON
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

# Define uma função para obter o número máximo de tokens permitido por um modelo específico
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Define uma função para recarregar a página do Streamlit
def refresh_page():
    st.experimental_rerun()

# Define uma função para salvar um novo especialista no arquivo JSON
def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

# Função principal para buscar uma resposta do assistente baseado no modelo Groq
def fetch_assistant_response(user_input: str, model_name: str, temperature: float, agent_selection: str, groq_api_key: str) -> Tuple[str, str]:
    global phase_two_response
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
                "que apresente suas características e qualificações que o tornam apto a lidar com a questão proposta: {user_input}. "
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
            f"educacional e concisa: {user_input}. "
            f"Considerando minha longa experiência e profundo conhecimento das disciplinas relacionadas, "
            f"é necessário abordar cada aspecto com atenção e rigor científico. "
            f"Portanto, irei delinear os principais elementos a serem considerados e investigados, fornecendo uma análise detalhada e baseada em evidências, "
            f"evitando vieses e citando referências conforme apropriado. "
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

# Função para refinar uma resposta existente com base na análise e melhoria do conteúdo
def refine_response(expert_title: str, phase_two_response: str, user_input: str, model_name: str, temperature: float, groq_api_key: str, json_file) -> str:
    global refined_response
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
            f"educacional e concisa: {user_input}. "
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

        if not json_file:
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

# Função para avaliar a resposta com base em um agente gerador racional (RAG)
def evaluate_response_with_rag(user_input: str, expert_description: str, refined_response: str, model_name: str, temperature: float, groq_api_key: str) -> str:
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
            f"A submissão original da pergunta é a seguinte: {user_input}. "
            f"A resposta fornecida pelo especialista em português é a seguinte: {refined_response}. "
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

def search_csv_and_json(csv_file, json_file, query):
    csv_data = pd.read_csv(csv_file)
    json_data = json.load(json_file)

    csv_results = csv_data[csv_data.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    json_results = [item for item in json_data if query.lower() in json.dumps(item).lower()]
    
    return csv_results, json_results

def display_search_results(csv_results, json_results):
    st.write("### Resultados da Pesquisa no CSV")
    st.write(csv_results)
    st.write("### Resultados da Pesquisa no JSON")
    st.write(json.dumps(json_results, indent=4))

# Carrega as opções de agentes a partir do arquivo JSON
agent_options = load_agent_options()

st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes Alan Kay</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

with st.expander("Clique para saber mais sobre os Agentes Experts Geomaker."):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")
    st.image("fluxograma agente 4.png")

user_input = st.text_area("Insira sua pergunta ou solicitação", height=100)
model_selection = st.selectbox("Escolha um modelo de linguagem", list(MODEL_MAX_TOKENS.keys()))
agent_selection = st.selectbox("Escolha um especialista", agent_options)
groq_api_key = st.text_input("Insira sua chave API do Groq")
csv_file = st.file_uploader("Envie um arquivo CSV para busca", type=["csv"])
json_file = st.file_uploader("Envie um arquivo JSON para busca", type=["json"])

if st.button("Buscar Resposta do Assistente"):
    if groq_api_key:
        expert_title, phase_two_response = fetch_assistant_response(user_input, model_selection, 0.7, agent_selection, groq_api_key)
        if expert_title and phase_two_response:
            st.success("Resposta obtida com sucesso!")
            st.write(f"**Título do Especialista:** {expert_title}")
            st.write(f"**Resposta:**\n{phase_two_response}")
        else:
            st.error("Não foi possível obter uma resposta. Verifique os dados fornecidos.")
    else:
        st.error("Por favor, insira sua chave API do Groq.")

if st.button("Refinar Resposta"):
    if groq_api_key and phase_two_response:
        refined_response = refine_response(expert_title, phase_two_response, user_input, model_selection, 0.7, groq_api_key, json_file)
        if refined_response:
            st.success("Resposta refinada com sucesso!")
            st.write(f"**Resposta Refinada:**\n{refined_response}")
        else:
            st.error("Não foi possível refinar a resposta.")
    else:
        st.error("Por favor, obtenha uma resposta inicial antes de refiná-la.")

if st.button("Avaliar Resposta com RAG"):
    if groq_api_key and refined_response:
        rag_response = evaluate_response_with_rag(user_input, expert_description, refined_response, model_selection, 0.7, groq_api_key)
        if rag_response:
            st.success("Resposta avaliada com sucesso!")
            st.write(f"**Avaliação RAG:**\n{rag_response}")
        else:
            st.error("Não foi possível avaliar a resposta com RAG.")
    else:
        st.error("Por favor, refine a resposta antes de avaliá-la com RAG.")

if st.button("Pesquisar nos Arquivos"):
    if csv_file and json_file:
        query = st.text_input("Insira a consulta de pesquisa")
        if query:
            csv_results, json_results = search_csv_and_json(csv_file, json_file, query)
            display_search_results(csv_results, json_results)
        else:
            st.error("Por favor, insira uma consulta de pesquisa.")
    else:
        st.error("Por favor, envie os arquivos CSV e JSON para pesquisa.")

if st.button("Recarregar Página"):
    refresh_page()

st.markdown(
    """
    <div style='text-align: center; font-size: 14px;'>
    <p><strong>Explore as infinitas possibilidades com os Agentes Experts Geomaker e transforme sua maneira de aprender e ensinar.</strong></p>
    <p>&copy; 2023 Geomaker. Todos os direitos reservados.</p>
    </div>
    """,
    unsafe_allow_html=True
)
