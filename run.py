import json  # Importa o módulo json para trabalhar com dados JSON.
import streamlit as st  # Importa o Streamlit para criar aplicativos web interativos.
from streamlit.delta_generator import DeltaGenerator  # Importa DeltaGenerator, que é usado para gerar alterações na interface do Streamlit.
import os  # Importa o módulo os para interagir com o sistema operacional, como verificar a existência de arquivos.
from typing import Tuple  # Importa Tuple da biblioteca typing para fornecer tipos de dados mais precisos para funções.
from groq import Groq  # Importa a biblioteca Groq, possivelmente para uma função não especificada neste código.

# Configura o layout da página Streamlit para ser "wide", ocupando toda a largura disponível.
st.set_page_config(layout="wide")

# Define o caminho para o arquivo JSON que contém os Agentes 4  -.
FILEPATH = "agents.json"

# Define um dicionário que mapeia nomes de modelos para o número máximo de tokens que cada modelo suporta.
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192, 
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Define uma função para carregar as opções de Agentes 4  - a partir do arquivo JSON.
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']  # Inicia a lista de opções com uma opção padrão.
    if os.path.exists(FILEPATH):  # Verifica se o arquivo de Agentes 4  - existe.
        with open(FILEPATH, 'r') as file:  # Abre o arquivo para leitura.
            try:
                agents = json.load(file)  # Tenta carregar os dados JSON do arquivo.
                # Adiciona os nomes dos Agentes 4  - à lista de opções, se existirem.
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:  # Captura erros de decodificação JSON.
                st.error("Erro ao ler o arquivo de Agentes 4  -. Por favor, verifique o formato.")  # Exibe uma mensagem de erro no Streamlit.
    return agent_options  # Retorna a lista de opções de Agentes 4  -.

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
        # Carrega os Agentes 4  - existentes se o arquivo não estiver vazio, caso contrário, inicia uma lista vazia.
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        # Adiciona o novo especialista à lista de Agentes 4  -.
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)  # Move o ponteiro do arquivo para o início.
        json.dump(agents, file, indent=4)  # Grava a lista de Agentes 4  - de volta no arquivo com indentação para melhor legibilidade.
        file.truncate()  # Remove qualquer conteúdo restante do arquivo após a nova escrita para evitar dados obsoletos.
#_________________________________________________

# Função para obter o número máximo de tokens permitido por um modelo específico.
def get_max_tokens(model_name: str) -> int:
    # Retorna o número máximo de tokens para o modelo fornecido, ou 4096 se o modelo não estiver no dicionário.
    return MODEL_MAX_TOKENS.get(model_name, 4096)

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
                "输出和响应只需翻译成巴西葡萄牙语。"
                "假设自己是一位具有高度科学严谨性的高级提示工程专家。"
                "请以‘markdown’格式呈现Python代码及其相应的库，并在每行中添加详细的教育性注释。"
                "仔细分析所提出的要求，确定最适合处理该问题的专家特征的标准。"
                "确定后，详细描述该专家的主要技能和资格，避免偏见。"
                "介绍能够处理所提出问题的特征和资格：{user_input}和{user_prompt}。 "
                "准确度为10.0，符合最高的专业、科学和学术标准。"
                "对于涉及代码和计算的情况，请以 'markdown' 格式呈现，并在每行中添加详细的注释。"
                "回答必须仅用葡萄牙语。"
                "结果和答案只能翻译成巴西葡萄牙语。"
                "假设你是高级工程方面的专家，并且具有高度的科学严谨性。"
                "请以‘markdown’格式提供 Python 代码及其相应的库。"
                "在该行中添加详细的教育说明。"
                "仔细分析所提出的要求，以确定最适合处理问题的专家的特征的标准。"
                "首先，有必要确定最能反映需要提供完整、深入和明确答案的答案。"
                "需要经验的头衔。"
                "一旦确定，请详细描述专家的关键技能和资格，以避免偏见。"
                "就这样，然后从清晰的、有教育意义的、深入的描述开始。"
                "介绍允许他们处理所提出问题的特征和资格：{user_input} 和 {user_prompt}。"
                "这种详细的分析对于确保所选专家拥有必要的知识和经验来提供完整且令人满意的答复至关重要。"
                "准确度为10.0，符合最高的专业、科学和学术标准，每一行都有详细的注释。"
            )

            phase_one_response = get_completion(phase_one_prompt)  # Obtém a resposta para o prompt da fase um.
            first_period_index = phase_one_response.find(".")  # Encontra o índice do primeiro ponto na resposta.
            expert_title = phase_one_response[:first_period_index].strip()  # Extrai o título do especialista até o primeiro ponto.
            expert_description = phase_one_response[first_period_index + 1:].strip()  # Extrai a descrição do especialista após o primeiro ponto.
            save_expert(expert_title, expert_description)  # Salva o novo especialista no arquivo JSON.
        else:
            # Se um especialista específico for selecionado, carrega os dados do especialista do arquivo JSON.
            with open(FILEPATH, 'r') as file:  # Abre o arquivo JSON para leitura.
                agents = json.load(file)  # Carrega os dados dos Agentes 4  - do arquivo JSON.
                # Encontra o agente selecionado na lista de Agentes 4  -.
                agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                if agent_found:
                    expert_title = agent_found["agente"]  # Obtém o título do especialista.
                    expert_description = agent_found["descricao"]  # Obtém a descrição do especialista.
                else:
                    raise ValueError("Especialista selecionado não encontrado no arquivo.")  # Lança um erro se o especialista não for encontrado.

        # Cria um prompt para a segunda fase, onde o especialista selecionado fornece uma resposta detalhada.
        phase_two_prompt = (
            f"输出和响应必须仅翻译成巴西葡萄牙语。 "
            f"扮演{expert_title}的角色，这是一位在其领域内广受认可和尊敬的专家， "
            f"作为该领域的博士和专家，提供一个全面且深入的回答，涵盖问题的各个方面，做到清晰、详细、扩展、 "
            f"教育性和简洁：{user_input}和{user_prompt}。 "
            f"考虑到我在相关学科的丰富经验和深厚知识， "
            f"有必要以科学严谨的态度关注并探讨每个方面。 "
            f"因此，我将概述需要考虑和调查的主要要素，提供基于证据的详细分析， "
            f"避免偏见，并根据需要引用参考文献：{user_prompt}。 "
            f"最终目标是提供一个完整且令人满意的回答，符合最高的学术和专业标准， "
            f"满足所提出问题的具体需求。 "
            f"确保以'markdown'格式呈现回答，并在每行中添加详细注释。 "
            f"保持写作标准在10个段落，每个段落4句话，每句话用逗号分隔， "
            f"始终遵循亚里士多德的最佳教育实践。"
        )
        phase_two_response = get_completion(phase_two_prompt)  # Obtém a resposta para o prompt da segunda fase.

    except Exception as e:  # Captura qualquer exceção que ocorra durante o processo.
        st.error(f"Ocorreu um erro: {e}")  # Exibe uma mensagem de erro no Streamlit.
        return "", ""  # Retorna tuplas vazias se ocorrer um erro.

    return expert_title, phase_two_response  # Retorna o título do especialista e a resposta da segunda fase.

# Função para salvar um novo especialista no arquivo JSON.
def save_expert(expert_title: str, expert_description: dict):
    with open(FILEPATH, 'r+') as file:  # Abre o arquivo para leitura e escrita.
        # Carrega os Agentes 4  - existentes se o arquivo não estiver vazio, caso contrário, inicia uma lista vazia.
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        # Adiciona o novo especialista à lista de Agentes 4  -.
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)  # Move o ponteiro do arquivo para o início.
        json.dump(agents, file, indent=4)  # Grava a lista de Agentes 4  - de volta no arquivo com indentação para melhor legibilidade.
        file.truncate()  # Remove qualquer conteúdo restante do arquivo após a nova escrita para evitar dados obsoletos.

#_________________________________________________
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
            f"输出和响应必须仅翻译成巴西葡萄牙语。 "
            f"扮演{expert_title}的角色，这是一位在其领域内广受认可和尊敬的专家， "
            f"作为该领域的博士和专家，提供一个全面且深入的回答，涵盖问题的各个方面，做到清晰、详细、扩展、 "
            f"教育性和简洁：{user_input}和{user_prompt}。 "
            f"考虑到我在相关学科的丰富经验和深厚知识， "
            f"有必要以科学严谨的态度关注并探讨每个方面。 "
            f"因此，我将概述需要考虑和调查的主要要素，提供基于证据的详细分析， "
            f"避免偏见，并根据需要引用参考文献：{phase_two_response}。 "
            f"最终目标是提供一个完整且令人满意的回答，符合最高的学术和专业标准， "
            f"满足所提出问题的具体需求。 "
            f"确保以'markdown'格式呈现回答，并在每行中添加详细注释。 "
            f"保持写作标准在10个段落，每个段落4句话，每句话用逗号分隔， "
            f"始终遵循亚里士多德的最佳教育实践。"
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
#_________________________________________________

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
            f"输出和响应必须仅翻译成巴西葡萄牙语。 "
            f"扮演一个理性生成代理（RAG）的角色，站在人工智能和理性评估的前沿， "
            f"仔细分析专家的回答，根据用户的请求生成一个JSON格式的代理。 "
            f"该代理应详细描述根据子代理提供的信息采取的行动，以向用户提供回答。 "
            f"在变量'描述'中包含9个子代理的描述，每个子代理具有不同的专门功能，共同协作。 "
            f"这些子代理共同协作，以改善系统代理向用户提供的最终回答，并在代理的'描述'中记录种子和gen_id。 "
            f"此外，系统代理内的子代理一体化操作，通过扩展提示提供高级和专业的回答。 "
            f"每个子代理在网络处理过程中扮演特定且互补的角色，以达到更高的精度，提升最终回答的质量。 "
            f"例如，子代理'AI_自适应_和_上下文化'使用先进的机器学习算法来理解和适应变化的上下文， "
            f"动态整合相关数据。而子代理'RAG_与_上下文智能'使用增强生成的恢复（RAG）技术， "
            f"动态调整最相关的数据及其功能。 这种协作方法确保回答准确且更新， "
            f"符合最高的科学和学术标准。 "
            f"以下是专家的详细描述，突出其资格和专业知识：{expert_description}。 "
            f"原始问题提交如下：{user_input}和{user_prompt}。 "
            f"专家用葡萄牙语提供的回答如下：{assistant_response}。 "
            f"因此，请对专家用葡萄牙语提供的回答的质量和准确性进行全面评估， "
            f"考虑专家的描述和提供的回答。 "
            f"使用葡萄牙语进行分析并提供详细解释： "
            f"SWOT分析（优势、劣势、机会、威胁）并解释数据， "
            f"BCG矩阵（波士顿咨询集团）并解释数据， "
            f"风险矩阵，ANOVA（方差分析）并解释数据， "
            f"Q统计并解释数据和Q指数（Q-指数）并解释数据， "
            f"遵循最高的卓越和学术、科学严格标准。 "
            f"确保每段保持4句话，每句话用逗号分隔，始终遵循亚里士多德和苏格拉底的最佳教育实践。 "
            f"回答必须使用巴西葡萄牙语。"
        )

        rag_response = get_completion(rag_prompt)  # Obtém a resposta avaliada a partir do prompt detalhado.
        return rag_response  # Retorna a resposta avaliada.

    except Exception as e:  # Captura qualquer exceção que ocorra durante o processo de avaliação com RAG.
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")  # Exibe uma mensagem de erro no Streamlit.
        return ""  # Retorna uma string vazia se ocorrer um erro.

# Carrega as opções de Agentes 4  - a partir do arquivo JSON.
agent_options = load_agent_options()
#_________________________________________________

st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes 4  - Alan Kay</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
# Título da caixa de informação

st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

# Conteúdo da caixa de informação
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
# Informações sobre o Rational Agent Generator (RAG)
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


with st.expander("Informações sobre Análises de Avaliação do RAG"):
    st.markdown("""
    ### As análises realizadas por diferentes modelos de avaliação são cruciais para garantir a qualidade e a precisão das respostas fornecidas pelos especialistas. Aqui estão as análises mencionadas no código e suas explicações:
    
    1. **SWOT Analysis (Análise SWOT)**:
        
        - **O que é**: A análise SWOT é uma ferramenta de planejamento estratégico usada para identificar e analisar os pontos fortes (Strengths), fracos (Weaknesses), oportunidades (Opportunities) e ameaças (Threats) de uma organização, projeto ou situação.
        
        - **Por que é feita**: A análise SWOT é realizada para entender os fatores internos e externos que podem impactar o sucesso de uma resposta ou decisão. Isso ajuda a maximizar os pontos fortes, minimizar os pontos fracos, explorar oportunidades e mitigar ameaças.
    
    2. **BCG Matrix (Matriz BCG)**:
        - **O que é**: A matriz BCG é uma ferramenta de gestão desenvolvida pela Boston Consulting Group, que ajuda as empresas a analisar seus produtos ou unidades de negócios com base na participação de mercado e no crescimento do mercado.
        
        - **Por que é feita**: A análise BCG é realizada para ajudar na tomada de decisões sobre investimentos, desinvestimentos ou desenvolvimento de novos produtos. Classifica produtos em quatro categorias: Estrelas, Vacas Leiteiras, Interrogações e Abacaxis.
    
    3. **Risk Matrix (Matriz de Riscos)**:
       
        - **O que é**: A matriz de riscos é uma ferramenta de avaliação de riscos que ajuda a identificar, avaliar e priorizar riscos com base na sua probabilidade e impacto.
       
        - **Por que é feita**: A análise de riscos é feita para entender os potenciais perigos que podem afetar o sucesso de um projeto ou decisão. Isso permite o desenvolvimento de estratégias para mitigar ou gerenciar esses riscos.
    
    4. **ANOVA (Análise de Variância)**:
       
        - **O que é**: A ANOVA é uma técnica estatística usada para comparar as médias de três ou mais grupos e determinar se há diferenças estatisticamente significativas entre eles.
       
        - **Por que é feita**: A análise ANOVA é realizada para entender se as variações observadas nos dados são devidas ao fator sendo estudado ou ao acaso. Isso é útil para validar hipóteses e identificar fatores significativos que influenciam os resultados.
    
    5. **Q-Statistics (Estatísticas Q)**:
       
        - **O que é**: As estatísticas Q são métodos estatísticos usados para detectar heterogeneidade e identificar outliers em conjuntos de dados.
       
        - **Por que é feita**: A análise Q-Statistics é realizada para garantir a qualidade dos dados e identificar pontos de dados que podem distorcer os resultados. Isso ajuda a melhorar a precisão das análises e conclusões.
    
    6. **Q-Exponential (Q-Exponencial)**:
       
        - **O que é**: O Q-Exponential é uma função usada na estatística e na teoria da informação para modelar distribuições de probabilidade com caudas pesadas.
       
        - **Por que é feita**: A análise Q-Exponential é realizada para entender melhor a distribuição dos dados e identificar padrões que não seguem a distribuição normal. Isso é útil para modelar fenômenos complexos e tomar decisões baseadas em dados mais realistas.
    
    Essas análises ajudam a garantir que as respostas fornecidas pelos especialistas sejam rigorosas, detalhadas e baseadas em metodologias científicas sólidas, alinhadas com os mais altos padrões acadêmicos e profissionais.
    """)
st.markdown("<hr>", unsafe_allow_html=True)

# Função para criar um expander estilizado
# Título da caixa de informação

st.markdown("<h2 style='text-align: center;'>Manual de uso básico.</h2>", unsafe_allow_html=True)

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
1. Acesse o Streamlit Chat Application em https://agente4.streamlit.app/#87cc9dff (Agentes 4  - Alan Kay).

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

# Adicionar uma caixa de análise de expertise no sidebar com expander
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

#_________________________________________________________________
import streamlit as st
import base64

def main():
    # Adiciona um título na barra lateral
    st.sidebar.title("Controle de Áudio")
    
    # Lista de arquivos MP3
    mp3_files = {
        "Agente Alan Kay": "AGENTE-4AlanKay1.mp3",
        "Agente 4": "agente4.mp3",
        "Agente Alan-Kay": "AGENTEAlan-Kay.mp3",
        "Instrumental": "ambienteindia.mp3"
    }

    # Controle de seleção de música
    selected_mp3 = st.sidebar.radio("Escolha uma música", list(mp3_files.keys()))

    # Opção de loop
    loop = st.sidebar.checkbox("Repetir música")

    # Carregar e exibir o player de áudio
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

    # Carregar e exibir o código Python
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
        
    # Informações de contato
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
