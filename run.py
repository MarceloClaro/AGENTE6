import streamlit as st

# Configurações da página do Streamlit
st.set_page_config(
    page_title="Exemplo de Layout no Streamlit",
    page_icon="📊",
    layout="wide",
)

st.title("Exemplo de Layout no Streamlit")

# Criando três colunas
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Coluna 1")
    st.write("Esta é a primeira coluna.")
    st.image("https://via.placeholder.com/150")

with col2:
    st.header("Coluna 2")
    st.write("Esta é a segunda coluna.")
    st.image("https://via.placeholder.com/150")

with col3:
    st.header("Coluna 3")
    st.write("Esta é a terceira coluna.")
    st.image("https://via.placeholder.com/150")

# Expansores para conteúdo adicional
with st.expander("Veja mais informações"):
    st.write("Aqui você pode adicionar informações adicionais que o usuário pode expandir para ver.")

# Exemplo de contêiner
with st.container():
    st.write("Este é um contêiner que agrupa vários elementos.")
    st.line_chart({"data": [1, 2, 3, 4, 5]})

# Exemplo de abas
tab1, tab2, tab3 = st.tabs(["Aba 1", "Aba 2", "Aba 3"])

with tab1:
    st.write("Conteúdo da Aba 1")

with tab2:
    st.write("Conteúdo da Aba 2")

with tab3:
    st.write("Conteúdo da Aba 3")

# Sidebar
st.sidebar.title("Barra Lateral")
st.sidebar.write("Esta é a barra lateral.")
st.sidebar.button("Clique aqui")
