import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Carregar os dados
dados = pd.read_csv('treinamento_produtividade.csv')  # Substitua pelo caminho do seu dataset

# Imprimir os nomes das colunas
print(dados.columns)

# Verificar as primeiras linhas
print(dados.head())

# Separar as variáveis
X = dados[['tempo_treinamento']]  # Manter como DataFrame
y = dados['tempo_produtividade (%)'].values  # Usar .values para obter um array

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Salvar o modelo
with open('modelo_regressao.pkl', 'wb') as f:
    pickle.dump(modelo, f)
