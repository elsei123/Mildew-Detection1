import pandas as pd
import numpy as np

def get_sample_data():
    """
    Função para retornar dados de exemplo para visualizações.
    """
    # Exemplo: dados aleatórios para um gráfico de linha
    data = pd.DataFrame({
        'Mês': range(1, 13),
        'Valor': np.random.randint(50, 150, 12)
    })
    data = data.set_index('Mês')
    return data
