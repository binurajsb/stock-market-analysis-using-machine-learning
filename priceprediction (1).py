import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import itertools
import warnings
import pickle

warnings.filterwarnings('ignore')

def orderfind(train):
    orde=[]
    sorde=[]
    aic=[]
    p=d=q=range(0,2)
    pdq=list(itertools.product(p,d,q))
    seasonal_pdq=[(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod=sm.tsa.statespace.SARIMAX(train,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                results=mod.fit()
                orde.append(param)
                aic.append(results.aic)
                sorde.append(param_seasonal)
            
          
                print('ARIMA{}X{}12=AIC:{}'.format(param,param_seasonal,results.aic))
            except:
                continue
    
    print(min(aic))
    print(aic.index(min(aic)))
    ind=aic.index(min(aic))
    print(orde[ind])    
    print(sorde[ind])
    
    return orde[ind],sorde[ind]




data=pd.read_csv('SBUX1.csv',index_col=0,parse_dates=[0])
y=data['close'].resample('MS').mean()
y.dropna(inplace=True)

decomposition=sm.tsa.seasonal_decompose(y,model='additive')

decomposition.plot()

p=d=q=range(0,2)



pdq=list(itertools.product(p,d,q))
seasonal_pdq=[(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]
            
orders,sorder=orderfind(y)            
            
mod=sm.tsa.SARIMAX(y,order=orders,seasonal_order=sorder,enforce_stationarity=False,enforce_invertibility=False)
result=mod.fit()
with open('prediction.pickle','wb') as f:
    pickle.dump(result,f)
    
pkl = open('prediction.pickle', 'rb')
clf = pickle.load(pkl) 
print(result.summary().tables[1])

result.plot_diagnostics(figsize=(16,8))
plt.show()



pred=clf.get_prediction(start=pd.to_datetime('2012-10-01 00:00:00'),dynamic=False)


pred_c1=pred.conf_int()

ax=y['2012':].plot(label='original')
pred.predicted_mean.plot(ax=ax,label='one step ahead',figsize=(16,8))
print(pred.predicted_mean)

ax.set_xlabel('date')
ax.set_ylabel('date')
plt.legend()
plt.show()
            

