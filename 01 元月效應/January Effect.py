import os
from os.path import join
import pandas as pd
import numpy as np
import pingouin as pg
from sklearn.linear_model import LinearRegression as LR
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error as MSE

path = os.getcwd()
src = join(path, 'source')
out = join(path, 'output')

TWII = pd.read_csv(join(src, 'TWII.csv') , engine = 'python' , index_col = 0)
TWII.index = pd.to_datetime(TWII.index)

TW150 = pd.read_csv(join(src, 'TW150_2006_S4.csv') , engine = 'python' , index_col = 0)
TW150.index = pd.to_datetime(TW150.index)

stock = TW150.dropna(axis = 1)

N = int(len(stock)/13)

log_stock = np.log(stock/stock.shift(N)).dropna(how = 'all')[1:]
#log_stock = ((stock.shift(N) - stock)/stock).dropna(how = 'all')[1:]
log_stock['month'] = log_stock.index.month
log_stock['year'] = log_stock.index.year
all_log_stock = log_stock.groupby(['month' , 'year']).mean()
all_log_stock = all_log_stock.mean(axis = 1)

'''
==============
January Effect
==============
'''

log_stock_jan = log_stock.loc[log_stock.index.month == 1 , :]
log_stock_jan['year'] = log_stock_jan.index.year
mean_jan = log_stock_jan.groupby(log_stock_jan['year']).mean()
mean_jan_2 = mean_jan.mean(axis = 1)

log_stock_njan = log_stock.loc[log_stock.index.month != 1 , :]
log_stock_njan['month'] = log_stock_njan.index.month
log_stock_njan['year'] = log_stock_njan.index.year
mean_njan = log_stock_njan.groupby(['month' , 'year']).mean()
mean_njan_2 = mean_njan.mean(axis = 1)

mean_sub = pd.Series()
for i in range(11):
    mean_sub = mean_sub.append(mean_njan_2.loc[i+2,:])
    

for i in range(12):
    year = i+2007
    df = pd.DataFrame()
    for j in range(11):
        mon = j+2
        df_t = pd.DataFrame(mean_njan_2.loc[mon,:] - mean_jan_2).reset_index()
        df_t['sub'] = df_t.loc[:,0] - mean_jan_2.iloc[0]
        df = df.append(df_t)
        
        df_y = pd.DataFrame(all_log_stock[all_log_stock.index.get_level_values('year') == year])\
        .reset_index().drop('year' , axis=1)
        df_y.index = df_y['month']
        df_y = df_y.drop('month' , axis=1)
        df_y.columns = [str(year)]
        
        p = df_y.plot(kind='line').get_figure()
        p.savefig(join(out , str(year) + '.jpg'))
    
aov = pg.anova(data=df, dv='sub', between='month', detailed=True)
aov.to_csv(join(out , 'Anova.csv') , index = False)

'''
==========
Regression
==========
'''
log_stock = np.log(stock/stock.shift(N)).dropna(how = 'all')[1:]
def reg_src(df_return , window):
    R = df_return.rolling(window).mean()[window:]
    std = df_return.rolling(window).std().shift(1).dropna(how = 'all')
    skew = df_return.rolling(window).skew().shift(1).dropna(how = 'all')
    kurt = df_return.rolling(window).kurt().shift(1).dropna(how = 'all')
    
    return R,std,skew,kurt

D = [60 , 120 , 180 , 240]

for win in D:
    df_reg = {}
    R,std,skew,kurt = reg_src(log_stock , win)
    for comp in log_stock:
        X = pd.DataFrame([std[comp] , skew[comp] , kurt[comp]]).T
        X.columns = ['std' , 'skew' , 'kurt']
        y = R[comp]
        model = LR()
        model.fit(X,y)
        a = [model.intercept_]
        a.extend(model.coef_.tolist())
        y_pred = model.predict(X)
        a.append(MSE(y , y_pred))
        p_val = ['']
        pvalue = f_regression(X , y)[1].round(6)
        p_val.extend(pvalue)
        p_val.append('')
        df_reg[comp] = a
        df_reg[comp+'_p-value'] = p_val
    
    df_reg = pd.DataFrame(df_reg , index = ['a0' , 'a1' , 'a2' , 'a3' , 'MSE'])\
    .to_csv(join(out , 'regression_windows'+str(win)+'.csv') , encoding = 'UTF-8-SIG')

