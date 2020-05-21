import os
from os.path import join
import pandas as pd
import numpy as np
import statsmodels.api as sm

path = os.getcwd()
src = join(path , 'source')
out = join(path , 'output')

TWII = pd.read_csv(join(src, 'TWII.csv') , engine = 'python' , index_col = 0)
TWII.index = pd.to_datetime(TWII.index)

TW150 = pd.read_csv(join(src, 'TW150_2006_S4.csv') , engine = 'python' , index_col = 0).dropna(axis=1)
TW150.index = pd.to_datetime(TW150.index)

N = int(len(TW150)/13)

log_stock = np.log(TW150/TW150.shift(N)).dropna(how = 'all')[1:]


'''
====
CAPM
====
'''
log_TWII = np.log(TWII.shift(1)/TWII).dropna()
stock = log_stock[1:]
stock_p = TW150[N+2:]

n = int(len(stock)/5)

def OLSmodel(y , x):
    X = sm.add_constant(x)
    est = sm.OLS(y , X)
    est = est.fit()
    alpha = est.params[0]
#    beta = est.params[1]
#    pval = est.pvalues[0]
    return alpha

def df_reg(df_Assets , df_Market):
    df = {'alpha':[]}# , 'pval':[] , 'beta':[]}
    for col in df_Assets:
        c = OLSmodel(df_Assets[col] , df_Market)
        df['alpha'].append(c)
#        df['pval'].append(p)
#        df['beta'].append(b)
    df = pd.DataFrame(df)
    df.index = df_Assets.columns
    return df

'''alpha動能'''
df_CAPM = {'date':[stock_p.index[5]] , 'alpha momentum':[100] , 'alpha reverse':[100] , 'TWII':[100]}
for i in range(n-1):
    if i != n-2:
        data = stock[5*i:5*(i+1)]
        market = log_TWII[5*i:5*(i+1)]
        
        alpha_CAPM = df_reg(data , market).dropna().sort_values(by='alpha' , ascending = False)
        
        target1 = list(alpha_CAPM[:5].index)
        target2 = list(alpha_CAPM[-5:].index)
        
        s = stock_p.index[5*(i+1)]
        e = stock_p.index[5*(i+1)+5]
        
        buy1 = TW150.loc[s,target1]
        sell1 = TW150.loc[e,target1]
        
        buy2 = TW150.loc[s,target2]
        sell2 = TW150.loc[e,target2]
        
        units1 = df_CAPM['alpha momentum'][i] * .2 / buy1
        units2 = df_CAPM['alpha reverse'][i] * .2 / buy2
        
        df_CAPM['date'].append(e)
        df_CAPM['alpha momentum'].append(sum(units1 * sell1))
        df_CAPM['alpha reverse'].append(sum(units2 * sell2))
        
        # 計算大盤報酬
        TWII_buy = TWII.loc[s]
        TWII_sell = TWII.loc[e]
        
        U_TWII = df_CAPM['TWII'][i] / TWII_buy
        
        df_CAPM['TWII'].append(U_TWII * TWII_sell)

    else:
        data = stock[5*i:5*(i+1)]
        market = log_TWII[5*i:5*(i+1)]
        
        alpha_CAPM = df_reg(data , market).dropna().sort_values(by='alpha' , ascending = False)
        
        target1 = list(alpha_CAPM[:5].index)
        target2 = list(alpha_CAPM[-5:].index)
        
        s = stock_p.index[5*(i+1)]
        e = stock_p.index[-1]
        
        buy1 = TW150.loc[s,target1]
        sell1 = TW150.loc[e,target1]
        
        buy2 = TW150.loc[s,target2]
        sell2 = TW150.loc[e,target2]
        
        units1 = df_CAPM['alpha momentum'][i] * .2 / buy1
        units2 = df_CAPM['alpha reverse'][i] * .2 / buy2
        
        df_CAPM['date'].append(e)
        df_CAPM['alpha momentum'].append(sum(units1 * sell1))
        df_CAPM['alpha reverse'].append(sum(units2 * sell2))
        
        # 計算大盤報酬
        TWII_buy = TWII.loc[s]
        TWII_sell = TWII.loc[e]
        
        U_TWII = df_CAPM['TWII'][i] / TWII_buy
        
        df_CAPM['TWII'].append(U_TWII * TWII_sell)


df_CAPM = pd.DataFrame(df_CAPM)
df_CAPM.to_excel(join(out , 'CAPM.xlsx') , index = False)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot( 'date', 'alpha momentum', data=df_CAPM, color='#F5B041', linewidth=2)
plt.plot( 'date', 'alpha reverse', data=df_CAPM, color='#5DADE2', linewidth=2)
plt.plot( 'date', 'TWII', data=df_CAPM, color='#E74C3C', linewidth=2)
plt.legend()
plt.grid(True)
plt.title('CAPM - alpha momentum vs alpha reverse')
plt.xlabel('Date')
plt.ylabel('Value')
plt.savefig(join(out ,'CAPM.png'))

