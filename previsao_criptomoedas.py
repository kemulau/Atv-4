import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Obtendo dados da CoinGecko API
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30"
response = requests.get(url)
data = response.json()

# Convertendo os dados para um DataFrame
precos = [item[1] for item in data["prices"]]
df = pd.DataFrame({"Preço": precos})

# Criando a variável alvo (1 = preço subiu, 0 = preço caiu)
df["Variação"] = (df["Preço"].diff() > 0).astype(int)
df.dropna(inplace=True)

# Criando variável de entrada
df["Preço Anterior"] = df["Preço"].shift(1)
df.dropna(inplace=True)

# Separando dados para treinamento e teste
X = df[["Preço Anterior"]]
y = df["Variação"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Avaliando o modelo
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2%}")
