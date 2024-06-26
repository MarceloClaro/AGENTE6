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

# Função para obter o número máximo de tokens permitido por um modelo específico
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Função para recarregar a página do Streamlit
def refresh_page():
    st.rerun()

# Função para processar arquivos PDF
def process_pdf_files(files):
    texts = []
    metadatas = []
    for file in files:
        pdf = PyPDF2.PdfFileReader(file)
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
        with open(file, 'r') as f:
            data = json.load(f)
            json_text = json.dumps(data, indent=4)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
            file_texts = text_splitter.split_text(json_text)
            texts.extend(file_texts)
            file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
            metadatas.extend(file_metadatas)
    return texts, metadatas

# Inicializar e configurar o modelo de chat
def initialize_chat_model(model_name: str):
    llm_groq = Groq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=0.2
    )
    return llm_groq

# Função para buscar uma resposta do assistente baseado no modelo Groq
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, groq_api_key: str) -> str:
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

        prompt = (
            f"User Input: {user_input}, User Prompt: {user_prompt}."
        )
        response = get_completion(prompt)
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return ""
    return response

# Função para tratar mensagens do usuário
def handle_message(message: str, model_name: str, temperature: float, groq_api_key: str):
    response = fetch_assistant_response(message, message, model_name, temperature, groq_api_key)
    return response

# Processamento de arquivos
def process_files(files):
    texts, metadatas = [], []
    pdf_files = [file for file in files if file.type == "application/pdf"]
    csv_files = [file for file in files if file.type == "text/csv"]
    json_files = [file for file in files if file.type == "application/json"]

    if pdf_files:
        pdf_texts, pdf_metadatas = process_pdf_files(pdf_files)
        texts.extend(pdf_texts)
        metadatas.extend(pdf_metadatas)

    if csv_files:
        csv_texts, csv_metadatas = process_csv_files(csv_files)
        texts.extend(csv_texts)
        metadatas.extend(csv_metadatas)

    if json_files:
        json_texts, json_metadatas = process_json_files(json_files)
        texts.extend(json_texts)
        metadatas.extend(json_metadatas)

    return texts, metadatas

# Inicialização do aplicativo
def main():
    st.title("Chatbot Avançado com Groq API")
    st.markdown("Faça upload de arquivos PDF, CSV ou JSON para iniciar.")

    files = st.file_uploader("Envie arquivos", accept_multiple_files=True, type=["pdf", "csv", "json"])

    if files:
        with st.spinner("Processando arquivos..."):
            texts, metadatas = process_files(files)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=initialize_chat_model("llama3-70b-8192"),
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )

        st.success("Processamento de arquivos concluído. Você já pode fazer perguntas!")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Digite sua pergunta:")

        if st.button("Enviar"):
            response = handle_message(user_input, "llama3-70b-8192", 0.2, groq_api_key)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.markdown("### Histórico de Conversas")
        for msg in st.session_state.chat_history[-50:]:
            st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

        if st.button("Limpar Histórico"):
            st.session_state.chat_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()
