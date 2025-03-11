import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Criando dados fictícios
data = {
    "Combustível": ["Gasolina", "Diesel", "Etanol", "Gasolina", "Diesel", "Etanol"],
    "Idade": [3, 5, 2, 8, 6, 1],
    "Quilometragem": [40000, 60000, 20000, 100000, 80000, 15000],
    "Preço": [35000, 30000, 40000, 20000, 25000, 45000]
}

df = pd.DataFrame(data)

# Separando variáveis independentes e dependente
X = df.drop(columns=["Preço"])
y = df["Preço"]

# Criando transformações para cada tipo de dado
column_transformer = ColumnTransformer([
    ("onehot", OneHotEncoder(), ["Combustível"]),
    ("scaler", StandardScaler(), ["Idade", "Quilometragem"])
])

# Criando o pipeline
pipeline = Pipeline([
    ("transformer", column_transformer),
    ("regressor", LinearRegression())
])

# Treinando o modelo
pipeline.fit(X, y)

# Fazendo previsões
y_pred = pipeline.predict(X)

# Calculando o erro quadrático médio (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Erro Quadrático Médio: {mse:.2f}")
