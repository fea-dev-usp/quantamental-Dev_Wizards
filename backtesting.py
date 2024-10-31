import pandas as pd
import numpy as np
import datetime as dt
import riskfolio as rp
import yfinance as yf
import bt
from tqdm import tqdm

# Definindo o intervalo de limites e objetivos
LIMITES_SUPERIOR = float(input("Digite o limite superior: "))
LIMITES_INFERIOR = float(input("Digite o limite inferior: "))
OBJETIVO = input("Digite o objetivo: ")

np.random.seed(42)

def main():
    indicadores, prices = carregar_dados()
    
    sinal = montar_sinal(LIMITES_SUPERIOR, LIMITES_INFERIOR, indicadores)
    indicadores_limite = reindexar(sinal, prices)
    pesos = calcular_precos(prices, indicadores_limite, obj=OBJETIVO)
    
    results = backtesting(prices, indicadores_limite, pesos)
    print(results['Indicador'].stats)
    print(results['Indicador'].display())

def backtesting(prices, indicadores, pesos):
    prices = prices[prices.index >= indicadores.index.min()]

    strat_indicador = bt.Strategy('Indicador', [
        bt.algos.WeighTarget(pesos),
        bt.algos.Rebalance()
    ])

    backtests = [
        bt.Backtest(strat_indicador, prices, initial_capital=1000000000)
    ]

    results = bt.run(*backtests)
    return results

def carregar_dados():
    indicadores = pd.read_csv('./data/indicadores/indicadores_final.csv', index_col='Data', parse_dates=True)['indicador']
    prices = pd.read_csv('./data/ativos_ibov/prices_adj_close.csv', index_col='Date', parse_dates=True)
    prices = prices[prices.index <= dt.datetime(2023, 12, 30)]
    return indicadores, prices

def montar_sinal(limite_superior, limite_inferior, indicadores):
    sinal = pd.Series(index=indicadores.index)
    reotimizar = True

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
    kwargs.setdefault('method_cov', 'ledoit')
    kwargs.setdefault('method_mu', 'hist')
    kwargs.setdefault('model', 'Classic')
    kwargs.setdefault('rm', 'MV')
    kwargs.setdefault('obj', 'MinRisk')
    kwargs.setdefault('rf', 0)
    kwargs.setdefault('l', 0)

    grupo = prices[prices.index <= data].iloc[-252*2:]
    Y = grupo.pct_change().dropna()
    port = rp.Portfolio(returns=Y)

    port.assets_stats(method_mu=kwargs['method_mu'], method_cov=kwargs['method_cov'])

    return port.optimization(model=kwargs['model'], rm=kwargs['rm'], obj=kwargs['obj'], rf=kwargs['rf'], l=kwargs['l'], hist=True)

def calcular_precos(prices, indicadores_sinal, **kwargs):
    kwargs.setdefault('method_cov', 'ledoit')
    kwargs.setdefault('method_mu', 'hist')
    kwargs.setdefault('model', 'Classic')
    kwargs.setdefault('rm', 'MV')
    kwargs.setdefault('obj', 'MinRisk')
    kwargs.setdefault('rf', 0)
    kwargs.setdefault('l', 0)

    indicadores_limite = indicadores_sinal[indicadores_sinal == 1].sort_index()
    pesos = pd.DataFrame()

    for data, entry in indicadores_limite.items():
        if entry:
            w = optimization(prices, data, obj=kwargs['obj']).T
            w.index = [data]
            pesos = pd.concat([pesos, w], axis=0)

    return pesos

if __name__ == '__main__':
    main()