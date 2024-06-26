import streamlit as st
import os
import json
import PyPDF2
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from typing import Tuple
from groq import Groq
import chainlit as cl

# Configuração do layout da página Streamlit para ser "wide"
st.set_page_config(layout="wide")

# Carregar variáveis de ambiente
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY não foi encontrado nas variáveis de ambiente")

# Dicionário de modelos e seus tokens máximos
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192, 
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Função para carregar opções de Agentes
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

# Função para obter o número máximo de tokens permitido por um modelo específico
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Função para recarregar a página do Streamlit
def refresh_page():
    st.rerun()

# Função para salvar um novo especialista no arquivo JSON
def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

# Função para buscar uma resposta do assistente baseado no modelo Groq
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
                "Assumir o papel de um especialista reconhecido e respeitado em seu campo..."
                "Para fornecer uma resposta detalhada e precisa à seguinte pergunta..."
                "User Input: {user_input}, User Prompt: {user_prompt}."
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
            f"Assumir o papel de {expert_title}, um especialista reconhecido..."
            f"Para fornecer uma resposta detalhada e precisa..."
            f"User Input: {user_input}, User Prompt: {user_prompt}."
        )
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

# Função para refinar uma resposta existente
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
            f"Assumir o papel de {expert_title}, um especialista reconhecido..."
            f"Para fornecer uma resposta detalhada e precisa..."
            f"User Input: {user_input}, User Prompt: {user_prompt}."
        )

        if not references_file:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas..."
                f"Certifique-se de fornecer uma resposta detalhada e precisa..."
            )

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

# Função para avaliar a resposta com base em um agente gerador racional (RAG)
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
            f"Assumir o papel de um agente gerador racional (RAG)..."
            f"Para avaliar a qualidade e a precisão da resposta..."
            f"User Input: {user_input}, User Prompt: {user_prompt}."
            f"Expert Description: {expert_description}, Assistant Response: {assistant_response}."
        )

        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Função para inicializar e configurar o modelo de chat
def initialize_chat_model(model_name: str):
    llm_groq = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=0.2
    )
    return llm_groq

# Função para processar arquivos PDF
def process_pdf_files(files):
    texts = []
    metadatas = []
    for file in files:
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)
    return texts, metadatas

# Função para processar arquivos CSV
def process_csv_files(files):
    texts = []
    metadatas = []
    for file in files:
        df = pd.read_csv(file)
        csv_text = df.to_string()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(csv_text)
        texts.extend(file_texts)
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)
    return texts, metadatas

# Função para processar arquivos JSON
def process_json_files(files):
    texts = []
    metadatas = []
    for file in files:
        data = json.load(file)
        json_text = json.dumps(data, indent=2)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(json_text)
        texts.extend(file_texts)
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)
    return texts, metadatas

# Função para processar diferentes tipos de arquivos
def process_files(file_type, files):
    if file_type == "PDF":
        return process_pdf_files(files)
    elif file_type == "CSV":
        return process_csv_files(files)
    elif file_type == "JSON":
        return process_json_files(files)
    else:
        raise ValueError("Tipo de arquivo não suportado")

# Inicialização do chat
@cl.on_chat_start
async def on_chat_start():
    file_type = await cl.AskSelectMessage(
        content="Selecione o tipo de arquivo que deseja enviar:",
        options=["PDF", "CSV", "JSON"],
        timeout=180
    ).send()
    
    files = await cl.AskFileMessage(
        content=f"Por favor, envie um ou mais arquivos {file_type} para começar!",
        accept=["application/pdf" if file_type == "PDF" else "text/csv" if file_type == "CSV" else "application/json"],
        max_size_mb=100,
        max_files=10,
        timeout=180
    ).send()
    
    texts, metadatas = process_files(file_type, files)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(texts, embeddings, metadatas=metadatas)
    
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=initialize_chat_model(model_name),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    msg = cl.Message(content=f"Processamento de {len(files)} arquivos concluído. Você já pode fazer perguntas!", elements=[])
    await msg.send()
    cl.user_session.set("chain", chain)

# Função para tratar mensagens do usuário
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\nFontes: {', '.join(source_names)}"
        else:
            answer += "\nNenhuma fonte encontrada"
    await cl.Message(content=answer, elements=text_elements).send()

# Configurar página
st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes 4 - Alan Kay</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Informação detalhada sobre os Agentes 4
with st.expander("Clique para saber mais sobre os Agentes 4 - Alan Kay."):
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
                "   - O aplicativo verifica se o arquivo `agents.json` existe no diretório. Este arquivo contém informações sobre os Agentes 4 - disponíveis.\n"
                "5. **agents.json existe?**:\n"
                "   - Decisão condicional:\n"
                "     - **Sim**:\n"
                "       - Carregar Agentes 4 -: O arquivo `agents.json` é carregado.\n"
                "       - **Erro ao carregar JSON?**:\n"
                "         - **Sim**: Exibir mensagem de erro: O aplicativo mostra uma mensagem de erro indicando problemas ao carregar o arquivo JSON.\n"
                "         - **Não**: Mostrar opções de Agentes 4 -: As opções de Agentes 4 - carregadas são exibidas.\n"
                "     - **Não**: Usar opções de Agentes 4 - padrão: Se o arquivo não for encontrado, o aplicativo usa opções de Agentes 4 - padrão.\n"
                "6. **Mostrar opções de Agentes 4 -**:\n"
                "   - O aplicativo exibe as opções de Agentes 4 - para o usuário escolher.\n"
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

# Informação sobre o Rational Agent Generator (RAG)
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
                "   - O aplicativo verifica se o arquivo `agents.json` existe no diretório. Este arquivo contém informações sobre os Agentes 4 - disponíveis.\n"
                "5. **agents.json existe?**:\n"
                "   - Decisão condicional:\n"
                "     - **Sim**:\n"
                "       - Carregar Agentes 4 -: O arquivo `agents.json` é carregado.\n"
                "       - **Erro ao carregar JSON?**:\n"
                "         - **Sim**: Exibir mensagem de erro: O aplicativo mostra uma mensagem de erro indicando problemas ao carregar o arquivo JSON.\n"
                "         - **Não**: Mostrar opções de Agentes 4 -: As opções de Agentes 4 - carregadas são exibidas.\n"
                "     - **Não**: Usar opções de Agentes 4 - padrão: Se o arquivo não for encontrado, o aplicativo usa opções de Agentes 4 - padrão.\n"
                "6. **Mostrar opções de Agentes 4 -**:\n"
                "   - O aplicativo exibe as opções de Agentes 4 - para o usuário escolher.\n"
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

# Informação sobre o Rational Agent Generator (RAG)
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
                "   - O aplicativo verifica se o arquivo `agents.json` existe no diretório. Este arquivo contém informações sobre os Agentes 4 - disponíveis.\n"
                "5. **agents.json existe?**:\n"
                "   - Decisão condicional:\n"
                "     - **Sim**:\n"
                "       - Carregar Agentes 4 -: O arquivo `agents.json` é carregado.\n"
                "       - **Erro ao carregar JSON?**:\n"
                "         - **Sim**: Exibir mensagem de erro: O aplicativo mostra uma mensagem de erro indicando problemas ao carregar o arquivo JSON.\n"
                "         - **Não**: Mostrar opções de Agentes 4 -: As opções de Agentes 4 - carregadas são exibidas.\n"
                "     - **Não**: Usar opções de Agentes 4 - padrão: Se o arquivo não for encontrado, o aplicativo usa opções de Agentes 4 - padrão.\n"
                "6. **Mostrar opções de Agentes 4 -**:\n"
                "   - O aplicativo exibe as opções de Agentes 4 - para o usuário escolher.\n"
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

# Função para criar um expander estilizado
def expander(title: str, content: str, icon: str):
    with st.expander(title):
        st.markdown(f'<img src="{icon}" style="vertical-align:middle"> {content}', unsafe_allow_html=True)

# Conteúdo do manual de uso
passo_1_content = """
1. Acesse o Groq Playground em https://console.groq.com/playground.

2. Faça login na sua conta ou crie uma nova conta.

3. No menu lateral, selecione "API Keys".

4. Clique em "Create API Key" e siga as instruções para criar uma chave API. Copie a chave gerada, pois será necessária para autenticar suas consultas.

5. Se quiser usar esta API Key provisória: [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]. Lembre-se de que ela pode não funcionar mais devido ao uso excessivo pelos usuários. Portanto, é aconselhável que cada usuário tenha sua própria chave API.
"""

passo_2_content = """
1. Acesse o Streamlit Chat Application em https://agente4.streamlit.app/#87cc9dff (Agentes 4 - Alan Kay).

2. Na interface do aplicativo, você verá um campo para inserir a sua chave API do Groq. Cole a chave que você copiou no Passo 1.

3. Escolha um Agente Especializado e um dos modelos de agente disponíveis para interagir. Você pode selecionar entre 'mixtral-8x7b-32768' com 32768 tokens, 'llama3-70b-8192'com 8192 tokens, 'llama3-8b-8192' com 8192 tokens, 'llama2-70b-4096'com 4096 tokens ou 'gemma-7b-it' com 8192 tokens.

4. Digite sua pergunta ou solicitação na caixa de texto e clique em "Enviar".

5. O aplicativo consultará o Groq API e apresentará a resposta do especialista. Você terá a opção de refinar a resposta ou avaliá-la com o RAG.
"""

passo_3_content = """
1. Se desejar refinar a resposta do especialista, clique em "Refinar Resposta". Digite mais detalhes ou correções na caixa de texto e clique em "Enviar".

2. O aplicativo consultará novamente o Groq API e apresentará a resposta refinada.
"""

passo_4_content = """
1. Se preferir avaliar a resposta com o RAG, clique em "Avaliar Resposta com o RAG". O RAG analisará a qualidade e a precisão da resposta do especialista e apresentará uma avaliação.

2. Você terá a opção de concordar ou discordar com a avaliação do RAG e fornecer feedback adicional, se desejar.
"""

passo_5_content = """
1. Após refinar a resposta ou avaliá-la com o RAG, você poderá encerrar a consulta ou fazer uma nova pergunta.
"""

passo_6_content = """
Para melhorar a eficiência e qualidade das respostas geradas pelos modelos de linguagem, o conteúdo inserido no campo "Escreva um prompt ou coloque o texto para consulta para o especialista (opcional)" deve ser detalhado, claro e específico. Aqui estão algumas diretrizes e possibilidades sobre o que incluir nesse campo:
        
#### Diretrizes para um Prompt Eficiente
        
1. **Contexto**: Forneça o contexto necessário para entender o problema ou a pergunta. Inclua informações relevantes sobre o cenário ou o objetivo da solicitação.
2. **Detalhamento**: Seja detalhado em sua pergunta ou solicitação. Quanto mais informações você fornecer, melhor o modelo poderá entender e responder.
3. **Objetivos**: Especifique claramente o que você espera obter com a resposta. Isso ajuda o modelo a focar nos aspectos mais importantes.
4. **Formato de Resposta**: Indique o formato desejado para a resposta (por exemplo, uma explicação passo a passo, código em Python com comentários, etc.).
5. **Referências**: Se aplicável, inclua referências ou fontes de informação que podem ser úteis para a resposta.
        
#### Exemplos de Prompts
        
1. **Análise de Dados**
   - Contexto: "Eu tenho um conjunto de dados sobre vendas de produtos ao longo de um ano."
   - Detalhamento: "Os dados incluem colunas para data, produto, quantidade vendida e receita."
   - Objetivos: "Gostaria de saber quais produtos têm o maior crescimento de vendas mensal e identificar padrões sazonais."
   - Formato de Resposta: "Por favor, forneça uma análise em Python, incluindo gráficos e comentários explicativos."
        
2. **Desenvolvimento de Modelo de Machine Learning**
   - Contexto: "Estou trabalhando em um projeto de previsão de preços de imóveis."
   - Detalhamento: "Os dados incluem características dos imóveis, como número de quartos, localização, tamanho e preço."
   - Objetivos: "Preciso desenvolver um modelo de machine learning que preveja os preços dos imóveis com base nessas características."
   - Formato de Resposta: "Gostaria de um exemplo de código em Python usando scikit-learn, com explicações sobre a escolha do modelo e a avaliação de desempenho."
        
3. **Revisão de Código**
   - Contexto: "Estou desenvolvendo um script para automatizar a coleta de dados da web."
   - Detalhamento: "O script é escrito em Python e utiliza bibliotecas como BeautifulSoup e requests."
   - Objetivos: "Gostaria de uma revisão do código para identificar possíveis melhorias em termos de eficiência e boas práticas de programação."
   - Formato de Resposta: "Por favor, forneça sugestões de melhorias e justifique-as com exemplos de código."
        
4. **Pesquisa Acadêmica**
   - Contexto: "Estou escrevendo um artigo sobre os impactos das mudanças climáticas na biodiversidade."
   - Detalhamento: "Estou focando nos efeitos em ecossistemas marinhos e terrestres."
   - Objetivos: "Preciso de uma revisão bibliográfica detalhada, incluindo as principais pesquisas recentes e suas conclusões."
   - Formato de Resposta: "Por favor, forneça um resumo estruturado com citações em formato ABNT."
        
#### Exemplo de Prompt Detalhado
        
        
Contexto: Eu tenho um conjunto de dados sobre vendas de produtos ao longo de um ano. Os dados incluem colunas para data, produto, quantidade vendida e receita.
Objetivos: Gostaria de saber quais produtos têm o maior crescimento de vendas mensal e identificar padrões sazonais.
Formato de Resposta: Por favor, forneça uma análise em Python, incluindo gráficos e comentários explicativos.
        
#### Conclusão
        
A qualidade do prompt é fundamental para obter respostas úteis e precisas de modelos de linguagem. Seguindo essas diretrizes e incluindo detalhes específicos no campo de prompt, você maximizará a eficiência e a qualidade das respostas geradas.
"""

# Exibição do manual de uso com expander estilizado
expander("Passo 1: Criação da Chave API no Groq Playground", passo_1_content, "https://img.icons8.com/office/30/000000/api-settings.png")
expander("Passo 2: Acesso ao Streamlit Chat Application", passo_2_content, "https://img.icons8.com/office/30/000000/chat.png")
expander("Passo 3: Refinamento da Resposta", passo_3_content, "https://img.icons8.com/office/30/000000/edit-property.png")
expander("Passo 4: Avaliação da Resposta com o RAG", passo_4_content, "https://img.icons8.com/office/30/000000/like--v1.png")
expander("Passo 5: Conclusão da Consulta", passo_5_content, "https://img.icons8.com/office/30/000000/faq.png")
expander("Passo 6: Construindo o Prompt", passo_6_content, "https://img.icons8.com/dusk/30/000000/code-file.png")
st.markdown("<hr>", unsafe_allow_html=True)

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    groq_api_key = st.text_input("Chave da API Groq: Você pode usar esse como teste - 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 ", key="groq_api_key")
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
    st.session_state.clear()
    st.experimental_rerun()

# Sidebar com manual de uso
st.sidebar.image("logo.png", width=200)

with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do Agentes 4 - Alan Kay é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

    **Inovações:**
    - Suporte a múltiplos modelos de linguagem: O código permite que o usuário escolha entre diferentes modelos de linguagem, como o LLaMA, para gerar respostas mais precisas e personalizadas.
    - Integração com a API Groq: A integração com a API Groq permite que o aplicativo utilize a capacidade de processamento de linguagem natural de alta performance para gerar respostas precisas.
    - Refinamento de respostas: O código permite que o usuário refine as respostas do modelo de linguagem, tornando-as mais precisas e relevantes para a consulta.
    - Avaliação com o RAG: A avaliação com o RAG (Rational Agent Generator) permite que o aplicativo avalie a qualidade e a precisão das respostas geradas pelos especialistas.

    **Pontos Positivos:**
    - Interface de usuário intuitiva: A interface de usuário baseada em Streamlit é fácil de usar e permite uma interação fluida com o aplicativo.
    - Flexibilidade na escolha de modelos: A possibilidade de escolher entre diferentes modelos de linguagem permite que o usuário selecione o mais adequado para suas necessidades.
    - Opções de refinamento e avaliação: As funcionalidades de refinamento e avaliação de respostas garantem que o usuário obtenha respostas de alta qualidade e relevância.

    **Limitações:**
    - Dependência de chave API: O aplicativo depende de uma chave API do Groq para funcionar, o que pode limitar o acesso de alguns usuários.
    - Complexidade do código: O código pode ser complexo para usuários sem experiência em programação, especialmente na configuração e integração com a API Groq.
    - Limitação de tokens: Cada modelo de linguagem tem uma limitação no número de tokens que pode processar, o que pode afetar a qualidade das respostas para consultas mais longas.

    **Sugestões de Melhoria:**
    - Automação da obtenção de chave API: Incluir um guia passo a passo para a obtenção da chave API diretamente na interface do aplicativo pode facilitar o acesso dos usuários.
    - Documentação detalhada: Fornecer uma documentação detalhada sobre a configuração e uso do aplicativo pode ajudar usuários iniciantes a entender e utilizar todas as funcionalidades.
    - Expansão do suporte a arquivos: Adicionar suporte para mais tipos de arquivos, além de PDF e JSON, pode tornar o aplicativo mais versátil e útil para diferentes tipos de consultas.

    Em resumo, o Agentes 4 - Alan Kay é um aplicativo poderoso e inovador que permite a interação com modelos de linguagem avançados através de uma interface intuitiva. Com algumas melhorias, ele pode se tornar uma ferramenta ainda mais acessível e versátil para usuários de diferentes níveis de experiência.
    """)

with st.sidebar.expander("Informações Importantes"):
    st.markdown("""
    1. A chave API do Groq é necessária para utilizar o aplicativo. 
    2. Você pode criar sua própria chave API no site do Groq.
    3. O aplicativo oferece suporte a múltiplos modelos de linguagem, incluindo LLaMA.
    4. As funcionalidades de refinamento e avaliação de respostas garantem alta qualidade nas respostas fornecidas pelos especialistas.
    5. A interface de usuário baseada em Streamlit é intuitiva e fácil de usar.
    """)
