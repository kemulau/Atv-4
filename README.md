# Pipeline para previsão de preço de automóveis

## Objetivo
Criar um pipeline utilizando a biblioteca Scikit-Learn para prever o preço de automóveis com base em variáveis categóricas e numéricas.

## Etapas do pipeline

1. **Codificação da variável categórica**: Converter o tipo de combustível em valores numéricos usando OneHotEncoder.
2. **Padronização das variáveis numéricas**: Aplicar StandardScaler na idade do veículo e na quilometragem.
3. **Uso de ColumnTransformer**: Aplicar transformações distintas para cada tipo de dado.
4. **Treinamento de um modelo de regressão linear**: Criar e ajustar um modelo de regressão linear para prever os preços dos automóveis.
5. **Avaliação do desempenho**: Utilizar o erro quadrático médio (MSE) como métrica de avaliação.

---

# Coleta e previsão de preços de criptomoedas

## Objetivo
Desenvolver um programa que utilize uma API de mercado financeiro para coletar dados históricos de preços de uma criptomoeda e treinar um modelo de aprendizado de máquina para prever se o preço do próximo dia será maior ou menor.

## Etapas do programa

1. **Coleta de dados históricos**: Acessar uma API pública (CoinGecko, Binance, Yahoo Finance etc.) para coletar preços diários de uma criptomoeda.
2. **Armazenamento dos dados**: Salvar os preços coletados em um arquivo CSV.
3. **Criação de variáveis preditoras**: Transformar os dados brutos em variáveis úteis para o modelo.
4. **Treinamento de um modelo de classificação**: Criar um modelo de machine learning para prever se o preço do próximo dia será maior ou menor.
5. **Avaliação do modelo**: Exibir a acurácia do modelo e visualizar a importância de cada variável.
