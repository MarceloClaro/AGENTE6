import json  # Importa o módulo json para trabalhar com dados JSON.
import streamlit as st  # Importa o Streamlit para criar aplicativos web interativos.
from streamlit.delta_generator import DeltaGenerator  # Importa DeltaGenerator, que é usado para gerar alterações na interface do Streamlit.
import os  # Importa o módulo os para interagir com o sistema operacional, como verificar a existência de arquivos.
from typing import Tuple  # Importa Tuple da biblioteca typing para fornecer tipos de dados mais precisos para funções.
from groq import Groq  # Importa a biblioteca Groq, possivelmente para uma função não especificada neste código.
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import CSVSearchTool

# Configura o layout da página Streamlit para ser "wide", ocupando toda a largura disponível.
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

# Define uma função para carregar as opções de agentes a partir do arquivo JSON.
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']  # Inicia a lista de opções com uma opção padrão.
    if os.path.exists(FILEPATH):  # Verifica se o arquivo de agentes existe.
        with open(FILEPATH, 'r') as file:  # Abre o arquivo para leitura.
            try:
                agents = json.load(file)  # Tenta carregar os dados JSON do arquivo.
                # Adiciona os nomes dos agentes à lista de opções, se existirem.
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:  # Captura erros de decodificação JSON.
                st.error("Erro ao ler o arquivo de agentes. Por favor, verifique o formato.")  # Exibe uma mensagem de erro no Streamlit.
    return agent_options  # Retorna a lista de opções de agentes.

# Define uma função para obter o número máximo de tokens permitido por um modelo específico.
def get_max_tokens(model_name: str) -> int:
    # Retorna o número máximo de tokens para o modelo fornecido, ou 4096 se o modelo não estiver no dicionário.
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Define uma função para recarregar a página do Streamlit.
def refresh_page():
    st.rerun()  # Recarrega a aplicação Streamlit.

# Define uma função para salvar um novo especialista no arquivo JSON.
def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:  # Abre o arquivo para leitura e escrita.
        # Carrega os agentes existentes se o arquivo não estiver vazio, caso contrário, inicia uma lista vazia.
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        # Adiciona o novo especialista à lista de agentes.
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)  # Move o ponteiro do arquivo para o início.
        json.dump(agents, file, indent=4)  # Grava a lista de agentes de volta no arquivo com indentação para melhor legibilidade.
        file.truncate()  # Remove qualquer conteúdo restante do arquivo após a nova escrita para evitar dados obsoletos.

# Função principal para buscar uma resposta do assistente baseado no modelo Groq.
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, groq_api_key: str) -> Tuple[str, str]:
    phase_two_response = ""  # Inicializa a variável para armazenar a resposta da segunda fase.
    expert_title = ""  # Inicializa a variável para armazenar o título do especialista.

    try:
        client = Groq(api_key=groq_api_key)  # Cria um cliente Groq usando a chave API fornecida.

        # Define uma função interna para obter a conclusão/completar um prompt usando a API Groq.
        def get_completion(prompt: str) -> str:
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

        if agent_selection == "Escolha um especialista...":
            # Se nenhum especialista específico for selecionado, cria um prompt para determinar o título e descrição do especialista.
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
            phase_one_response = get_completion(phase_one_prompt)  # Obtém a resposta para o prompt da fase um.
            first_period_index = phase_one_response.find(".")  # Encontra o índice do primeiro ponto na resposta.
            expert_title = phase_one_response[:first_period_index].strip()  # Extrai o título do especialista até o primeiro ponto.
            expert_description = phase_one_response[first_period_index + 1:].strip()  # Extrai a descrição do especialista após o primeiro ponto.
            save_expert(expert_title, expert_description)  # Salva o novo especialista no arquivo JSON.
        else:
            # Se um especialista específico for selecionado, carrega os dados do especialista do arquivo JSON.
            with open(FILEPATH, 'r') as file:  # Abre o arquivo JSON para leitura.
                agents = json.load(file)  # Carrega os dados dos agentes do arquivo JSON.
                # Encontra o agente selecionado na lista de agentes.
                agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                if agent_found:
                    expert_title = agent_found["agente"]  # Obtém o título do especialista.
                    expert_description = agent_found["descricao"]  # Obtém a descrição do especialista.
                else:
                    raise ValueError("Especialista selecionado não encontrado no arquivo.")  # Lança um erro se o especialista não for encontrado.

        # Cria um prompt para a segunda fase, onde o especialista selecionado fornece uma resposta detalhada.
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
        phase_two_response = get_completion(phase_two_prompt)  # Obtém a resposta para o prompt da segunda fase.

    except Exception as e:  # Captura qualquer exceção que ocorra durante o processo.
        st.error(f"Ocorreu um erro: {e}")  # Exibe uma mensagem de erro no Streamlit.
        return "", ""  # Retorna tuplas vazias se ocorrer um erro.

    return expert_title, phase_two_response  # Retorna o título do especialista e a resposta da segunda fase.

# Função para refinar uma resposta existente com base na análise e melhoria do conteúdo.
def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, groq_api_key: str, references_file: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)  # Cria um cliente Groq usando a chave API fornecida.

        # Define uma função interna para obter a conclusão/completar um prompt usando a API Groq.
        def get_completion(prompt: str) -> str:
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

        # Cria um prompt detalhado para refinar a resposta.
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
def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, groq_api_key: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)  # Cria um cliente Groq usando a chave API fornecida.

        # Define uma função interna para obter a conclusão/completar um prompt usando a API Groq.
        def get_completion(prompt: str) -> str:
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

        # Cria um prompt detalhado para avaliar a resposta usando o agente gerador racional (RAG).
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

        rag_response = get_completion(rag_prompt)  # Obtém a resposta avaliada a partir do prompt detalhado.
        return rag_response  # Retorna a resposta avaliada.

    except Exception as e:  # Captura qualquer exceção que ocorra durante o processo de avaliação com RAG.
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")  # Exibe uma mensagem de erro no Streamlit.
        return ""  # Retorna uma string vazia se ocorrer um erro.

# Função para configurar a ferramenta de busca no CSV
def configure_csv_search_tool():
    # Implementar a configuração da ferramenta de busca no CSV aqui
    pass

# Função para criar um expander estilizado
def expander(title: str, content: str, icon: str):
    with st.expander(title): 
        st.markdown(f'<img src="{icon}" style="vertical-align:middle"> {content}', unsafe_allow_html=True)

# Carrega as opções de agentes a partir do arquivo JSON.
agent_options = load_agent_options()

st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes Alan Kay</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
# Título da caixa de informação

st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

# Conteúdo da caixa de informação
with st.expander("Clique para saber mais sobre os Agentes Experts Geomaker."):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")
    st.image("fluxograma agente 4.png")

# Função para criar e executar a equipe de agentes e tarefas
def execute_team(expert_title: str, phase_two_response: str, refined_response: str, rag_response: str):
    # Criação de agentes
    agent_phase_two = Agent(name="Phase Two Agent", description=phase_two_response, model="gpt-3.5-turbo")
    agent_refined = Agent(name="Refined Agent", description=refined_response, model="gpt-3.5-turbo")
    agent_rag = Agent(name="RAG Agent", description=rag_response, model="gpt-3.5-turbo")

    # Criação de tarefas
    task_phase_two = Task(name="Generate Phase Two Response", agent=agent_phase_two, prompt=phase_two_response)
    task_refined = Task(name="Generate Refined Response", agent=agent_refined, prompt=refined_response)
    task_rag = Task(name="Evaluate with RAG", agent=agent_rag, prompt=rag_response)

    # Criação e execução da equipe
    crew = Crew(name="Geomaker Crew", tasks=[task_phase_two, task_refined, task_rag])
    crew.execute()

# Barra lateral com manual de uso
st.sidebar.image("logo.png", width=200)

with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do Agente Expert Geomaker é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

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

# Exibição do manual de uso com expander estilizado
expander("Passo 1: Criação da Chave API no Groq Playground", passo_1_content, "https://img.icons8.com/office/30/000000/api-settings.png")
expander("Passo 2: Acesso ao Streamlit Chat Application", passo_2_content, "https://img.icons8.com/office/30/000000/chat.png")
expander("Passo 3: Refinamento da Resposta", passo_3_content, "https://img.icons8.com/office/30/000000/edit-property.png")
expander("Passo 4: Avaliação da Resposta com o RAG", passo_4_content, "https://img.icons8.com/office/30/000000/like--v1.png")
expander("Passo 5: Conclusão da Consulta", passo_5_content, "https://img.icons8.com/office/30/000000/faq.png")
expander("Passo 6: Construindo o Prompt", passo_6_content, "https://img.icons8.com/dusk/30/000000/code-file.png")
st.markdown("<hr>", unsafe_allow_html=True)

# Interface principal
st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Selecione um especialista:", agent_options)
    groq_api_key = st.text_input("Insira sua chave API do Groq:", type="password")
    model_name = st.selectbox("Escolha o modelo:", list(MODEL_MAX_TOKENS.keys()))
    temperature = st.slider("Temperatura (ajuste a aleatoriedade da resposta):", min_value=0.0, max_value=1.0, value=0.5)

# Configuração e busca de CSV e JSON
configure_csv_search_tool()

# Botão para iniciar o processamento
if st.button("Processar"):
    expert_title, phase_two_response = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key)
    refined_response = refine_response(expert_title, phase_two_response, user_input, user_prompt, model_name, temperature, groq_api_key, references_file="")
    rag_response = evaluate_response_with_rag(user_input, user_prompt, expert_description="", assistant_response=refined_response, model_name=model_name, temperature=temperature, groq_api_key=groq_api_key)

    st.success("Resposta processada com sucesso!")  # Mensagem de sucesso quando o processamento é concluído.

    # Integração das respostas finais em uma nova janela
    st.markdown(f"### {expert_title}")
    st.markdown(phase_two_response)
    st.markdown(refined_response)
    st.markdown(rag_response)

    # Chama a função para executar a equipe de agentes
    execute_team(expert_title, phase_two_response, refined_response, rag_response)
