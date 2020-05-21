# -*- coding: utf-8 -*-
import os
from os.path import join
import pandas as pd
import numpy as np
import talib
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as LR
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error as MSE

path = os.getcwd()
src = join(path , 'source')
out = join(path , 'output')

# data
TW150 = pd.read_csv(join(src, 'TW150_2006_S4.csv') , engine = 'python' , index_col = 0)
TW150.index = pd.to_datetime(TW150.index)
TW150 = TW150.dropna(axis = 1)

D = [10,20,30,40,60]

# loop 1
#d = 10
for d in D:
    TW150_2 = TW150[len(TW150[TW150.index.year == 2006])-d+1:]
    #df_p = TW150_2[d-1:]
    df_MA = TW150_2.apply(lambda x : talib.MA(x , d)).dropna(how="all")
    df_std = TW150_2.rolling(d).std().dropna(how="all")
        
    writer = pd.ExcelWriter(join(out,'BBand Strategy'+' D'+str(d)+'.xlsx'),datetime_format='YYYY-MM-DD')
    
    # loop 2
    for comp in TW150_2:
        result = {}
        benchmark = {}# = {'Mean':[],'IRR':[],'sigma':[],'Sharpe_IRR':[]}
        for up in [1,2]:
            for down in [2,1]:
                
                #comp = TW150_2.columns[0]
                upper, middle, lower = talib.BBANDS(TW150_2[comp],timeperiod=d,nbdevup=up,nbdevdn=down,matype=0)
                df = pd.DataFrame({"P":list(TW150_2[comp]),"UPBBand":list(upper),"DBBand":list(lower)},index=TW150_2.index).dropna()
                
                df["signals"] = np.nan
                df.loc[(df["P"]>df["UPBBand"]),"signals"] = 1
                df.loc[df["P"]<df["DBBand"],"signals"] = -1
                
                df["signals"] = df["signals"].fillna(method='ffill')
                df["signals"] = df["signals"].replace(np.nan,0)
                
                #changes = df.drop_duplicates(["signals"])
                #changes = list((df["signals"] != np.nan).drop_duplicate().index)
        
#                i = 0
                value = [100]
                for i in range(len(df.index)):
                    date = df.index[i]
                    
                    if i == 0:
                        if df["signals"][i] == 1:
                            u = value[-1]/df["P"][date]
                            
                        else:
                            pass
                    
                    elif df["signals"][i] == 0:
                        value.append(value[-1])
                        
                    elif (df["signals"][i] == 1) & (df["signals"][i-1] == 1):
                        value.append(u * df["P"][i])
                        
                    elif (df["signals"][i] == -1) & (df["signals"][i-1] == -1):
                        value.append(value[-1])
                        
                    elif (df["signals"][i] == 1) & (df["signals"][i-1] != 1):
                        u = value[-1]/df["P"][date]
                        value.append(value[-1])
                                                        
                    elif (df["signals"][i] == -1) & (df["signals"][i-1] != -1):
                        
                        if df["signals"][i-1] == 0:
                            value.append(value[-1])
                            
                        else:
                            value.append(u * df["P"][date])
                            
                result['bband_up'+str(up)+'_down'+str(down)] = value
                                
        result = pd.DataFrame(result,index=df.index)
        
        N = len(df)/12
        R = result.pct_change()[1:]
        
        IRR = round(np.power(result.iloc[-1,:]/result.iloc[0,:] , 1/12) - 1,6)
        STD = R.std()*np.sqrt(N)
        
        benchmark['Mean'] = R.mean() * N
        benchmark['IRR'] = IRR.values
        benchmark['sigma'] = STD
        benchmark['Sharpe_IRR'] = IRR/STD
        
        benchmark = pd.DataFrame(benchmark)
        
        result = pd.concat([df,result],axis=1)     
        result.to_excel(writer,sheet_name=comp,float_format="%.4f")
        
        benchmark.to_excel(writer,sheet_name=comp,float_format="%.4f",startcol = len(result.columns)+3)
        
    writer.close()
    
# twii
TWII = pd.read_csv(join(src, 'TWII.csv') , engine = 'python' , index_col = 0)
TWII.index = pd.to_datetime(TWII.index)
TWII = TWII.dropna(axis = 1)

writer = pd.ExcelWriter(join(out,'BBand Strategy(TWII).xlsx'),datetime_format='YYYY-MM-DD')

for d in D:
    TWII_2 = TWII[len(TWII[TWII.index.year == 2006])-d+1:]
    #df_p = TW150_2[d-1:]
    df_MA = TWII_2.apply(lambda x : talib.MA(x , d)).dropna(how="all")
    df_std = TWII_2.rolling(d).std().dropna(how="all")
    
    result = {}
    benchmark = {}# = {'Mean':[],'IRR':[],'sigma':[],'Sharpe_IRR':[]}
    for up in [1,2]:
        for down in [2,1]:
            
            #comp = TW150_2.columns[0]
            upper, middle, lower = talib.BBANDS(TWII_2.iloc[:,0],timeperiod=d,nbdevup=up,nbdevdn=down,matype=0)
            df = pd.DataFrame({"P":list(TWII_2.iloc[:,0]),"UPBBand":list(upper),"DBBand":list(lower)},index=TWII_2.index).dropna()
            
            df["signals"] = np.nan
            df.loc[(df["P"]>df["UPBBand"]),"signals"] = 1
            df.loc[df["P"]<df["DBBand"],"signals"] = -1
            
            df["signals"] = df["signals"].fillna(method='ffill')
            df["signals"] = df["signals"].replace(np.nan,0)
            
            #changes = df.drop_duplicates(["signals"])
            #changes = list((df["signals"] != np.nan).drop_duplicate().index)
    
#                i = 0
            value = [100]
            for i in range(len(df.index)):
                date = df.index[i]
                
                if i == 0:
                    if df["signals"][i] == 1:
                        u = value[-1]/df["P"][date]
                        
                    else:
                        pass
                
                elif df["signals"][i] == 0:
                    value.append(value[-1])
                    
                elif (df["signals"][i] == 1) & (df["signals"][i-1] == 1):
                    value.append(u * df["P"][i])
                    
                elif (df["signals"][i] == -1) & (df["signals"][i-1] == -1):
                    value.append(value[-1])
                    
                elif (df["signals"][i] == 1) & (df["signals"][i-1] != 1):
                    u = value[-1]/df["P"][date]
                    value.append(value[-1])
                
                elif (df["signals"][i] == -1) & (df["signals"][i-1] != -1):
                    
                    if df["signals"][i-1] == 0:
                        value.append(value[-1])
                        
                    else:
                        value.append(u * df["P"][date])
                        
            result['bband_up'+str(up)+'_down'+str(down)] = value
                            
    result = pd.DataFrame(result,index=df.index)
    
    N = len(df)/12
    R = result.pct_change()[1:]
    
    IRR = round(np.power(result.iloc[-1,:]/result.iloc[0,:] , 1/12) - 1,6)
    STD = R.std()*np.sqrt(N)
    
    benchmark['Mean'] = R.mean() * N
    benchmark['IRR'] = IRR.values
    benchmark['sigma'] = STD
    benchmark['Sharpe_IRR'] = IRR/STD
    
    benchmark = pd.DataFrame(benchmark)
    
    result = pd.concat([df,result],axis=1)     
    result.to_excel(writer,sheet_name='D='+str(d),float_format="%.4f")
    
    benchmark.to_excel(writer,sheet_name='D='+str(d),float_format="%.4f",startcol = len(result.columns)+3)
    
writer.close()

