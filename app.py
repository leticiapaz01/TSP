from flask import Flask, request, render_template
import numpy as np
import pickle

# Criação da aplicação Flask
app = Flask(__name__)

# Carregar o modelo treinado
with open('modelo_regressao.pkl', 'rb') as f:
    modelo = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obter o valor de tempo de treinamento do formulário
    tempo_treinamento = float(request.form['tempo_treinamento'])
    
    # Transformar a entrada em um array adequado para o modelo
    tempo_treinamento_array = np.array([[tempo_treinamento]])
    
    # Fazer a previsão usando o modelo
    produtividade_prevista = modelo.predict(tempo_treinamento_array)
    
    # Retornar o resultado em HTML
    return render_template('resultado.html', horas=tempo_treinamento, produtividade=produtividade_prevista[0])

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Altere a porta conforme necessário
