from tensorflow.keras.models import load_model

def load_trained_model(model_path="../src/model.h5"):
    """
    Carrega o modelo treinado a partir do caminho especificado.
    """
    return load_model(model_path)

def predict_image(model, image):
    """
    Realiza a predição em uma única imagem.
    """
    # Supondo que 'image' já esteja pré-processada
    import numpy as np
    image = image.reshape((1, *image.shape))
    prediction = model.predict(image)
    return prediction
