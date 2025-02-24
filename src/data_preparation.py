import cv2
import os
import numpy as np

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Carrega uma imagem, redimensiona e normaliza.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

def remove_non_image_files(directory):
    """
    Remove arquivos que não são imagens do diretório especificado.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png')
    files = os.listdir(directory)
    return [f for f in files if f.lower().endswith(valid_extensions)]
