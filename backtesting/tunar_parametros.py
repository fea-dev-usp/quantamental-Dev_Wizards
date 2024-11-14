import pandas as pd
import numpy as np
import datetime as dt
import riskfolio as rp
import yfinance as yf
import bt
from tqdm import tqdm

# Definindo o intervalo de limites e objetivos
LIMITES_SUPERIOR = range(70, 81)
LIMITES_INFERIOR = range(20, 31)
OBJETIVOS = ['MaxRet', 'MinRisk', 'Utility', 'Sharpe']

np.random.seed(42)

def main():
    indicadores, prices = carregar_dados()
    
    resultados_otimization = pd.DataFrame(columns=['limite_inferior', 'limite_superior', 'obj', 'start', 'end', 'rf', 
                                                   'total_return', 'cagr', 'max_drawdown', 'calmar', 'mtd', 'three_month', 
                                                   'six_month', 'ytd', 'one_year', 'three_year', 'five_year', 'ten_year', 
                                                   'incep', 'daily_sharpe', 'daily_sortino', 'daily_mean', 'daily_vol', 
                                                   'daily_skew', 'daily_kurt', 'best_day', 'worst_day', 'monthly_sharpe', 
                                                   'monthly_sortino', 'monthly_mean', 'monthly_vol', 'monthly_skew', 
                                                   'monthly_kurt', 'best_month', 'worst_month', 'yearly_sharpe', 
                                                   'yearly_sortino', 'yearly_mean', 'yearly_vol', 'yearly_skew', 
                                                   'yearly_kurt', 'best_year', 'worst_year', 'avg_drawdown', 
                                                   'avg_drawdown_days', 'avg_up_month', 'avg_down_month', 'win_year_perc', 
                                                   'twelve_month_win_perc'])
    
    total_combinations = len(LIMITES_SUPERIOR) * len(LIMITES_INFERIOR) * len(OBJETIVOS)
    with tqdm(total=total_combinations, desc="Testando parâmetros") as pbar:
        for limite_superior in LIMITES_SUPERIOR:
            for limite_inferior in LIMITES_INFERIOR:
                for obj in OBJETIVOS:
                    sinal = montar_sinal(limite_superior, limite_inferior, indicadores)
                    indicadores_limite = reindexar(sinal, prices)
                    pesos = calcular_precos(prices, indicadores_limite, obj)
                    
                    results = backtesting(prices, indicadores_limite, pesos).to_frame().T
                    results['limite_inferior'] = limite_inferior
                    results['limite_superior'] = limite_superior
                    results['obj'] = obj
                    
                    resultados_otimization = pd.concat([resultados_otimization, results], axis=0)
                    pbar.update(1)

    resultados_otimization.to_csv('./data/resultados_otimization2.csv')

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
    return results['Indicador'].stats

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

def optimization(prices, data, obj, **kwargs):
    kwargs.setdefault('method_cov', 'ledoit')
    kwargs.setdefault('method_mu', 'hist')
    kwargs.setdefault('model', 'Classic')
    kwargs.setdefault('rm', 'MV')
    kwargs.setdefault('rf', 0)
    kwargs.setdefault('l', 0)

    grupo = prices[prices.index <= data].iloc[-252*2:]
    Y = grupo.pct_change().dropna()
    port = rp.Portfolio(returns=Y)

    port.assets_stats(method_mu=kwargs['method_mu'], method_cov=kwargs['method_cov'])

    return port.optimization(model=kwargs['model'], rm=kwargs['rm'], obj=obj, rf=kwargs['rf'], l=kwargs['l'], hist=True)

def calcular_precos(prices, indicadores_sinal, obj):
    indicadores_limite = indicadores_sinal[indicadores_sinal == 1].sort_index()
    pesos = pd.DataFrame()

    for data, entry in indicadores_limite.items():
        if entry:
            w = optimization(prices, data, obj=obj).T
            w.index = [data]
            pesos = pd.concat([pesos, w], axis=0)

    return pesos

if __name__ == '__main__':
    main()