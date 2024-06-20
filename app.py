# Primeiro, certifique-se de instalar as bibliotecas necessárias
!pip install streamlit colorthief

# Crie um arquivo Python para o aplicativo Streamlit
app_code = """
import streamlit as st
from colorthief import ColorThief
from PIL import Image

def main():
    st.title("Extrator de Cor Dominante")
    st.write("Faça upload de uma imagem para extrair a cor dominante")

    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem Carregada', use_column_width=True)
        st.write("")
        st.write("Analisando...")

        # Salve a imagem temporariamente para usar com ColorThief
        image_path = "/tmp/temp_image.png"
        image.save(image_path)
        
        # Use ColorThief para pegar a cor dominante
        color_thief = ColorThief(image_path)
        dominant_color = color_thief.get_color(quality=1)
        
        st.write(f"A cor dominante é: {dominant_color}")
        st.write(f"Cor em hexadecimal: #{dominant_color[0]:02x}{dominant_color[1]:02x}{dominant_color[2]:02x}")
        
        # Mostrar a cor dominante
        st.markdown(
            f'''
            <div style="width:100px; height:100px; background-color: rgb({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]});"></div>
            ''',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
"""

with open("app.py", "w") as file:
    file.write(app_code)

# Execute o Streamlit no Colab
!streamlit run app.py & npx localtunnel --port 8501
