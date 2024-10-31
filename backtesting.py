import pandas as pd
import numpy as np
import datetime as dt
import riskfolio as rp

LIMITE_SUPERIOR = 77
LIMITE_INFERIOR = 23

np.random.seed(42)

def main():
    indicadores, prices = carregar_dados()

    sinal = montar_sinal(LIMITE_SUPERIOR, LIMITE_INFERIOR, indicadores)
    indicadores_limite = reindexar(sinal, prices)




def carregar_dados():
    indicadores = pd.read_csv('./data/indicadores/indicadores_final.csv', index_col='Data', parse_dates=True)['indicador']
    prices = pd.read_csv('./data/ativos_ibov/prices_adj_close.csv', index_col='Date', parse_dates=True)
    prices = prices[prices.index <= dt.datetime(2023, 12, 30)]

    return indicadores, prices

def montar_sinal(limite_superior, limite_inferior, indicadores):
    limite_superior = 77
    limite_inferior = 23

    reotimizar = True

    sinal = pd.Series(index=indicadores.index)
    for data, indicador in indicadores.items():
        if (indicador > limite_superior or indicador < limite_inferior):
            if reotimizar:
                sinal.loc[data] = 1
            else:
                sinal.loc[data] = 0
            reotimizar = False
        else:
            sinal.loc[data] = 0
            reotimizar = True

    return sinal

def reindexar(indicadores, prices):
    prices_limite = prices[prices.index >= indicadores.index.min()]
    business_days = prices_limite.index

    indicadores = indicadores.reindex(business_days).ffill()
    return indicadores

def optimization(prices, data, **kwargs):
    if 'method_cov' not in kwargs:
        method_cov = 'ledoit'
    if 'method_mu' not in kwargs:
        method_mu = 'hist'
    if 'model' not in kwargs:
        model = 'Classic'
    if 'rm' not in kwargs:
        rm = 'MV'
    if 'obj' not in kwargs:
        obj = 'MinRisk'
    if 'rf' not in kwargs:
        rf = 0
    if 'l' not in kwargs:
        l = 0

    # Selecionando o período de 2 anos de dados
    grupo = prices[prices.index <= data].iloc[-252*2:]

    # Calculando os retornos
    Y = grupo.pct_change().dropna()

    # Criando o portfólio
    port = rp.Portfolio(returns=Y)

    method_mu = kwargs['method_mu']
    method_cov = kwargs['method_cov']  # Usando Ledoit-Wolf shrinkage para tornar a matriz de cov positiva definida

    # Calculando as estatísticas dos ativos
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    # Parâmetros da otimização
    model = kwargs['model']
    rm = kwargs['rm']
    obj = kwargs['obj']
    hist = True
    rf = kwargs['rf']
    l = kwargs['l']

    # Otimizando o portfólio
    return port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)