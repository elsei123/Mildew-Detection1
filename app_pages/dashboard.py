import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Função para carregar o modelo treinado
def load_trained_model(model_path="../src/model.h5"):
    model = load_model(model_path)
    return model

# Função para pré-processar a imagem e realizar a predição
def predict_image(model, image, target_size=(128, 128)):
    # Redimensionar a imagem para o tamanho esperado pelo modelo
    image = image.resize(target_size)
    img_array = np.array(image)
    
    # Se a imagem estiver em escala de cinza, converte para RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    else:
        if img_array.shape[2] == 4:  # Se houver canal alfa, remove-o
            img_array = img_array[:,:,:3]
    
    # Normalização dos pixels
    img_array = img_array / 255.0
    # Adiciona uma dimensão para representar o batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Realiza a predição
    prediction = model.predict(img_array)
    return prediction[0][0]

# Configuração da página do Streamlit
st.set_page_config(page_title="Mildew Detection Dashboard", layout="wide")
st.title("Mildew Detection in Cherry Leaves")

# Menu lateral para navegação entre páginas do dashboard
page = st.sidebar.radio("Navegação", ["Home", "Predição", "Análises"])

if page == "Home":
    st.header("Visão Geral do Projeto")
    st.write("""
    Este dashboard permite a detecção automática de míldio em folhas de cereja utilizando um modelo de Machine Learning.
    Selecione a página 'Predição' para testar o modelo com uma imagem de sua escolha ou 'Análises' para visualizar insights.
    """)
    
elif page == "Predição":
    st.header("Realize a Predição")
    st.write("Faça o upload de uma imagem de folha de cereja para saber se ela está saudável ou apresenta míldio.")
    
    # Componente para upload de arquivo
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem carregada', use_column_width=True)
        
        # Carrega o modelo treinado
        with st.spinner('Carregando o modelo...'):
            model = load_trained_model()
        
        st.write("Realizando a predição...")
        prediction = predict_image(model, image)
        threshold = 0.5  # Limiar para classificação; ajuste conforme necessário
        
        # Exibe o resultado da predição
        if prediction < threshold:
            st.success("A folha está saudável!")
        else:
            st.error("A folha apresenta míldio!")
            
elif page == "Análises":
    st.header("Análises e Visualizações")
    st.write("Nesta seção, exiba os gráficos e insights obtidos durante a análise exploratória dos dados.")
    
    # Exemplo simples de gráfico interativo com Streamlit (pode ser substituído por gráficos mais elaborados)
    import pandas as pd
    import numpy as np
    chart_data = pd.DataFrame(
         np.random.randn(20, 3),
         columns=["Saudável", "Míldio", "Outros"])
    st.line_chart(chart_data)
