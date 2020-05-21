import os
from os.path import join
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as LR
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error as MSE

path = os.getcwd()
src = join(path , 'source')
out = join(path , 'output')

'''
==========
Regression
==========
'''
TWII = pd.read_csv(join(src, 'TWII.csv') , engine = 'python' , index_col = 0)
TWII.index = pd.to_datetime(TWII.index)

TW150 = pd.read_csv(join(src, 'TW150_2006_S4.csv') , engine = 'python' , index_col = 0)
TW150.index = pd.to_datetime(TW150.index)
#TW150 = TW150.drop('6004 元京證' , axis = 1)

N = int(len(TW150)/13)

log_stock = np.log(TW150/TW150.shift(N)).dropna(how = 'all')[1:]
'''
df_return = log_stock
window = 60
'''
def reg_src(df_return , window):
    R = df_return.rolling(window).mean()[window+2:]
    
    std = df_return.rolling(window).std().shift(1).dropna(how = 'all')
    dstd = (std.shift(1) - std).shift(1).dropna(how = 'all')
    
    skew = df_return.rolling(window).skew().shift(1).dropna(how = 'all')
    dskew = (skew.shift(1) - skew).shift(1).dropna(how = 'all')
    
    kurt = df_return.rolling(window).kurt().shift(1).dropna(how = 'all')
    dkurt = (kurt.shift(1) - kurt).shift(1).dropna(how = 'all')
    
    return R,dstd,dskew,dkurt

D = [60 , 120 , 180 , 240]

for win in D:
    
        df_reg = {}
        R,dstd,dskew,dkurt = reg_src(log_stock , win)
        
        for comp in log_stock:
            
            try:
                X = pd.DataFrame([dstd[comp] , dskew[comp] , dkurt[comp]]).T
                X.columns = ['std' , 'skew' , 'kurt']
                y = R[comp]
                if y.isnull().values.any():
                    X = X.dropna(how = 'all')[:-2]
                    y = y.dropna(how = 'all')
       
                model = LR()
                model.fit(X,y)
                
            except:
                 print('Error:' , comp , '[D=',win,']')
                 continue
            
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
        .to_excel(join(out , 'regression_windows('+str(win)+').xlsx'))

'''
====
CAPM
====
'''

log_TWII = np.log(TWII.shift(1)/TWII).dropna()
stock = log_stock[1:]

def OLSmodel(y , x):
    X = sm.add_constant(x)
    est = sm.OLS(y , X)
    est = est.fit()
    const = est.params[0]
    beta = est.params[1]
    pval = est.pvalues[0]
    return const , beta , pval

def df_reg(df_Assets , df_Market):
    df = {'alpha':[] , 'pval':[] , 'beta':[]}
    for col in df_Assets:
        c , b , p = OLSmodel(df_Assets[col] , df_Market)
        df['alpha'].append(c)
        df['pval'].append(p)
        df['beta'].append(b)
    df = pd.DataFrame(df).T
    df.columns = df_Assets.columns
    return df

df = df_reg(stock , log_TWII)
df.to_excel(join(out , 'CAPM.xlsx'))

'''
===============================
Fama-French Three-Factor Model
===============================
- 2006/12/29的資料為標準
'''
data = pd.read_excel(join(src , 'DATA.xlsx')).dropna()
data['BtoM'] = data['每股淨值(B)']/data['收盤價(元)']
data = data.sort_values(by = 'BtoM').reset_index(drop=True)
n = int(len(data)*0.3)

data['Value'] = np.nan
for i in range(len(data)):
    if i <= n:
        data['Value'][i] = 'L'
    elif i >= len(data)-n:
        data['Value'][i] = 'H'
    else:
        data['Value'][i] = 'M'

data = data.sort_values(by = '市值(百萬元)').reset_index(drop=True)

data['Size'] = np.nan
for i in range(len(data)):
    if i <= 75:
        data['Size'][i] = 'S'
    else:
        data['Size'][i] = 'B'

data['Type'] = data['Value'] + data['Size']
data = data[['公司' , 'Size' , 'Value' , 'Type']]
data.to_excel(join(out , 'Company Categories.xlsx'))
data = data.set_index('公司')

def cal_smb_hml(df_return , df_class):
    
    R_SL = df_return.loc[:,df_class[(df_class['Size'] == 'S') & (df_class['Value'] == 'L')].index].values.mean()
    R_SM = df_return.loc[:,df_class[(df_class['Size'] == 'S') & (df_class['Value'] == 'M')].index].values.mean()
    R_SH = df_return.loc[:,df_class[(df_class['Size'] == 'S') & (df_class['Value'] == 'H')].index].values.mean()
    R_BL = df_return.loc[:,df_class[(df_class['Size'] == 'B') & (df_class['Value'] == 'L')].index].values.mean()
    R_BM = df_return.loc[:,df_class[(df_class['Size'] == 'B') & (df_class['Value'] == 'M')].index].values.mean()
    R_BH = df_return.loc[:,df_class[(df_class['Size'] == 'B') & (df_class['Value'] == 'H')].index].values.mean()
    
    SMB = (R_SL + R_SM + R_SH)/3 - (R_BL + R_BM + R_BH)/3
    HML = (R_SH + R_BH - R_SL - R_BL)
    
    return SMB , HML

year = list(np.unique(log_stock.index.year))

FF = {'year':year , 'R':[] , 'SMB':[] , 'HML':[]}
for y in year:
    R = log_stock[log_stock.index.year == y].dropna(axis = 1)
    cmps = data.loc[R.columns,:]
    SMB , HML = cal_smb_hml(R , cmps)
    FF['R'].append(R.values.mean())
    FF['SMB'].append(SMB)
    FF['HML'].append(HML)

FF = pd.DataFrame(FF).set_index('year')

x = FF.drop('R' , axis = 1)
X = sm.add_constant(x)
y = FF['R']

est = sm.OLS(y , X)
est = est.fit()
parms = pd.DataFrame(est.params).reset_index().T
parms.columns = FF.columns
FF = FF.append(parms)

year.extend(['','P-value'])
FF.index = year
FF.to_excel(join(out,'FFModel.xlsx') , index_label = 'year')
