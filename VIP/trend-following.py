import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from HJB_Solver_v2 import get_thresholds
import warnings
warnings.filterwarnings('ignore')


def calculate_params_v2(data, rho_df):

    chng_thresh = .2

    # data = yf.download("^GSPC", start=pre_start_date, end=end_date).loc[:,['Adj Close']]
    # data.index = pd.to_datetime(data.index)

    # data['ma'] = data['Adj Close'].rolling(ma_size).mean()
    # # data = data.iloc[ma_size:]
    # data = data[start_date:end_date]

    # data['change'] = np.sign(data['Adj Close']-data['ma'])
    # data['ret'] = data['Adj Close'].pct_change()

    # print(data)

    uptrend_lengths = []
    downtrend_lengths = []

    uptrend_ann_rets = []
    downtrend_ann_rets = []

    trend = 0

    local_peak = (data.iloc[0,0],0)
    local_trough = (data.iloc[0,0],0)

    for i in range(1,len(data)):

        if data.iloc[i,2]*data.iloc[i-1,2]==-1:

            if data.iloc[i,2]==1:

                if trend==-1:

                    lt = np.min(data.iloc[local_peak[1]:i,0])
                    length = list(data.iloc[local_peak[1]:i,0]).index(lt)

                    local_trough = (lt,length + local_peak[1])

                    chng = local_trough[0] / local_peak[0]

                    if chng<=(1+chng_thresh):
                        downtrend_lengths.append(252/length)
                        downtrend_ann_rets.append((chng**(252/length))-1)

                trend = 1


            else:

                if trend==1:

                    lp = np.max(data.iloc[local_trough[1]:i,0])
                    length = list(data.iloc[local_trough[1]:i,0]).index(lp)

                    local_peak = (lp,length + local_trough[1])

                    chng = local_peak[0] / local_trough[0]

                    if chng>=(1+chng_thresh):
                        uptrend_lengths.append(252/length)
                        uptrend_ann_rets.append((chng**(252/length))-1)

                trend = -1
    
    lambda1 = np.median(uptrend_lengths)
    lambda2 = np.median(downtrend_lengths)

    # print(uptrend_lengths)
    # print(uptrend_ann_rets)

    mu1 = np.median(uptrend_ann_rets)
    mu2 = np.median(downtrend_ann_rets)

    sigma  = np.std(data.iloc[:,3]) * np.sqrt(252)

    rho = np.median(rho_df)

    params = {'lambda1':lambda1,'lambda2':lambda2,'mu1':mu1,'mu2':mu2,'sigma':sigma,'K':0.001,'rho':rho/100}

    for k in params.keys():
        if np.isnan(params[k]):
            params[k] = 0

    return params 


def p_t1(pt,params,S0,S1):

    dt = 1/252

    g_pt = -pt*(params['lambda1']+params['lambda2']) + params['lambda2'] - (pt*(params['mu1']-params['mu2'])*(1-pt)*(pt*(params['mu1']-params['mu2']) +params['mu2'] - 0.5*(params['sigma']**2)))/(params['sigma']**2)

    return min(max(pt + g_pt*dt + (pt * (1-pt) * (params['mu1']-params['mu2']) * np.log(S1/S0))/(params['sigma']**2),0),1)



def backtest_v2(data,params,p_i,position,last,last_buy,bal,bh_bal):

    # data = yf.download("^GSPC", start=start_date, end=end_date).loc[:,['Adj Close']]
    # data.index = pd.to_datetime(data.index)

    old_bal = bal
    old_bh_bal = bh_bal
    

    # p_s = 0.8
    # p_b = 0.95
    p_b, p_s = get_thresholds(params)

    if p_i != None:
        data['P(t)'] = pd.Series([(p_b + p_s)/2]*len(data),index=data.index)
    else:
        data['P(t)'] = pd.Series([p_i]*len(data),index=data.index)

    data['position'] = pd.Series([position]*len(data),index=data.index)

    for i in range(1,len(data)):

        pt1 = p_t1(data.iloc[i-1,1],params,data.iloc[i-1,0],data.iloc[i,0])

        data.iloc[i,1] = pt1

        if data.iloc[i-1,2]==0 and pt1>=p_b:
            data.iloc[i,2] = 1
        elif data.iloc[i-1,2]==1 and pt1<=p_s:
            data.iloc[i,2] = 0
        else:
            data.iloc[i,2] = data.iloc[i-1,2]

    # bal = 1
    # last_buy = 0
    rets = []

    for i in range(1,len(data)):

        if data.iloc[i-1,2]==0 and data.iloc[i,2]==1:
            last_buy = data.iloc[i,0]
        elif data.iloc[i-1,2]==1 and data.iloc[i,2]==0 and last_buy!=0:
            bal *= data.iloc[i,0]/last_buy
            # print(last_buy)
            rets.append(data.iloc[i,0]/last_buy - 1)
            last_buy = 0

    if last and last_buy!=0:
        bal *= data.iloc[-1,0]/last_buy
        last_buy = 0

    # print(rets)

    strat_ret = bal/old_bal - 1

    bh_bal*=data.iloc[-1,0]/data.iloc[0,0]
    # try:
    bh_ret = bh_bal/old_bh_bal - 1
    # except:
    #     print(start_date)
    #     print(end_date)
    #     print(data)

    # print(pt1)

    return strat_ret, bh_ret, p_b, p_s, pt1, data.iloc[-1,2], last_buy, bal, bh_bal
    
# import warnings
# warnings.filterwarnings('ignore')



def incr_date(date,n):
    return '-'.join([str(int(date.split('-')[0])+n)]+date.split('-')[1:])


df = yf.download("^GSPC", start="1960-01-01", end="2023-01-01").loc[:,['Adj Close']]
df.index = pd.to_datetime(df.index)

rho_df = yf.download("^IRX", start="1960-01-01", end="2023-01-01").loc[:,['Adj Close']]
rho_df.index = pd.to_datetime(rho_df.index)

pre_start_date = "1960-01-01"
in_start_date = incr_date(pre_start_date,3)
in_end_date = incr_date(in_start_date,10)

ma_sizes = (126,189,252,278,504,630,756)

ma_size_names = {126:'6-mo',189:'9-mo',252:'1-yr',278:'1.5-yr',504:'2-yr',630:'2.5-yr',756:'3-yr'}


for m_size in ma_sizes:

    results2 = []

    dataset = df.copy()

    dataset['ma'] = dataset['Adj Close'].rolling(m_size).mean()
    dataset['change'] = np.sign(dataset['Adj Close']-dataset['ma'])
    dataset['ret'] = dataset['Adj Close'].pct_change()

    params_old = {}

    position = 0
    last_buy = 0
    last = False
    bal = 1
    bh_bal = 1

    p_i = None

    for i in range(50):

        if i==49:
            last = True

        
        start_date = incr_date(in_start_date,i)
        end_date = incr_date(start_date,10)
        
        result = {'start-date':end_date,'end-date':incr_date(end_date,1),'ma_size':ma_size_names[m_size]}

        params = calculate_params_v2(dataset[start_date:end_date],rho_df[start_date:end_date])

        # # break
        # print(params)
        # print(start_date,end_date)

        if params_old != {}:
            for k in list(params.keys())[:5]:

                # if np.isnan(params[k]):
                #     params[k] = params_old[k]
                #     continue

                if params[k]==0:
                    params[k] = params_old[k]
                    continue

                params[k] = (1-2/6) * params_old[k] + (2/6) * params[k]

        params_old = params

        # print(params)
        

        strat_ret, bh_ret, p_b, p_s, p_i, position, last_buy, bal, bh_bal = backtest_v2(dataset[end_date:incr_date(end_date,1)],params,p_i,position,last,last_buy,bal,bh_bal)

        # break

        result['strat_ret'] = strat_ret
        result['bh_ret'] = bh_ret
        result['p_b'] = p_b
        result['p_s'] = p_s

        for key in params.keys():
            result[key] = params[key]

        results2.append(result)

    results_df2 = pd.DataFrame(results2,columns=results2[0].keys()).round(3)

    # with pd.ExcelWriter('Results_v3.xlsx', engine='openpyxl', mode='a') as writer: 
    #     results_df2.to_excel(writer,sheet_name=ma_size_names[m_size]) 

    # results_df2.to_excel('Results_v2.xlsx',)

    
        

        

