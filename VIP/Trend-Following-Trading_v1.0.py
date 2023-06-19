"""
Author: Manav Shah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from HJB_Solver import get_thresholds


def get_sp_stats_df(df):
    # Get regime stats
    # df - adj close price data
    df['ret'] = df['Adj Close'].pct_change()

    df['ret'][0] = 0

    df['cumret'] = (1 + df['ret']).cumprod()

    periods = []

    regime = 0

    peak = (1,0)
    trough = (1,0)

    for i in range(len(df)):

        if regime==0:

            cum_ret = df.iloc[i,-1]
            
            if cum_ret<0.81:
                regime = -1
                trough = (cum_ret,i)

            elif cum_ret>1.24:
                regime = 1
                peak = (cum_ret,i)

        else:

            if regime == -1:
                if df.iloc[i,-1]/trough[0]>1.24:
                    regime = 1
                    periods.append({'regime':'bear','top/bottom':df.index[trough[1]],'index':df.iloc[trough[1],0],'% move':(trough[0]/peak[0])-1,'mean':np.mean(df.iloc[peak[1]:trough[1],1]),'stddev':np.std(df.iloc[peak[1]:trough[1],1]),'duration':trough[1]-peak[1]})
                    peak = (df.iloc[i,-1],i)

                elif df.iloc[i,-1] < trough[0]:
                    trough = (df.iloc[i,-1],i)

            elif regime == 1:       
                if df.iloc[i,-1]/peak[0]<0.81:
                    regime = -1
                    periods.append({'regime':'bull','top/bottom':df.index[peak[1]],'index':df.iloc[peak[1],0],'% move':(peak[0]/trough[0])-1,'mean':np.mean(df.iloc[trough[1]:peak[1],1]),'stddev':np.std(df.iloc[trough[1]:peak[1],1]),'duration':peak[1]-trough[1]})
                    trough = (df.iloc[i,-1],i)

                elif df.iloc[i,-1] > peak[0]:
                    peak = (df.iloc[i,-1],i)

    if regime == -1:
        periods.append({'regime':'bear','top/bottom':df.index[trough[1]],'index':df.iloc[trough[1],0],'% move':(trough[0]/peak[0])-1,'mean':np.mean(df.iloc[peak[1]:trough[1],1]),'stddev':np.std(df.iloc[peak[1]:trough[1],1]),'duration':len(df)-peak[1]})

    elif regime == 1:       
        periods.append({'regime':'bull','top/bottom':df.index[peak[1]],'index':df.iloc[peak[1],0],'% move':(peak[0]/trough[0])-1,'mean':np.mean(df.iloc[trough[1]:peak[1],1]),'stddev':np.std(df.iloc[trough[1]:peak[1],1]),'duration':len(df)-trough[1]})


    stats_df = pd.DataFrame.from_dict(periods)
    stats_df.index = pd.to_datetime(stats_df['top/bottom'])
    stats_df.drop(['top/bottom'],axis=1,inplace=True)
    # stats_df.to_excel('params.xlsx')

    return stats_df

def calculate_params_v4(data, rho_df, df):
    # compute params based on regime
    # df - stats_df

    uptrend_lengths = []
    downtrend_lengths = []

    uptrend_ann_rets = []
    downtrend_ann_rets = []

    sigma1 = []
    sigma2 = []

    for i in range(len(df)):
        
        if df['regime'][i] == 'bull':
            uptrend_lengths.append(252/df['duration'][i])
            # uptrend_ann_rets.append((1 + df['% move'][i])**(252/df['duration'][i]) - 1)
            uptrend_ann_rets.append(df['mean'][i] * 252)
            sigma1.append(df['stddev'][i] * (252**0.5))
        
        elif df['regime'][i] == 'bear':
            downtrend_lengths.append(252/df['duration'][i])
            # downtrend_ann_rets.append((1 + df['% move'][i])**(252/df['duration'][i]) - 1)  
            downtrend_ann_rets.append(df['mean'][i] * 252)      
            sigma2.append(df['stddev'][i] * (252**0.5))

    lambda1 = np.mean(uptrend_lengths)
    lambda2 = np.mean(downtrend_lengths)

    mu1 = np.mean(uptrend_ann_rets)
    mu2 = np.mean(downtrend_ann_rets)

    # print(data)

    sigma = np.mean(sigma1)/2 + np.mean(sigma2)/2

    # sigma  = np.std(data['ret']) * np.sqrt(252)

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



def backtest_v4(data,params,p_i,position,last,last_buy,bal,bh_bal,cum_ret):
    # this is for one year

    # data = yf.download("^GSPC", start=start_date, end=end_date).loc[:,['Adj Close']]
    # data.index = pd.to_datetime(data.index)

    old_bal = bal
    old_bh_bal = bh_bal
    
    data = data.loc[:,['Adj Close','ret']]

    if position == 0:
        last_sell = 0

    # print(data)

    # p_s = 0.8
    # p_b = 0.95
    p_b, p_s = get_thresholds(params)

    if p_i == None:
        data['P(t)'] = pd.Series([(p_b + p_s)/2]*len(data),index=data.index)
    else:
        data['P(t)'] = pd.Series([p_i]*len(data),index=data.index)

    data['position'] = pd.Series([position]*len(data),index=data.index)

    # data['ret'] = data['Adj Close'].pct_change()

    data = data.loc[:,['Adj Close','P(t)','position','ret']]

    # data['ret'][0] = 0

    # for i in range(1,len(data)):

    #     # pt1 = p_t1(data.iloc[i-1,1],params,data.iloc[i-1,0],data.iloc[i,0])
    #     pt1 = p_t1(data['P(t)'][i-1],params,data['Adj Close'][i-1],data['Adj Close'][i])

    #     # data.iloc[i,1] = pt1
    #     data['P(t)'][i] = pt1

    #     # if data.iloc[i-1,2]==0 and pt1>=p_b:
    #     if data['position'][i-1]==0 and pt1>=p_b:
    #         # data.iloc[i,2] = 1
    #         data['position'][i] = 1
    #     # elif data.iloc[i-1,2]==1 and pt1<=p_s:
    #     elif data['position'][i-1]==1 and pt1<=p_s:
    #         # data.iloc[i,2] = 0
    #         data['position'][i] = 0
    #     else:
    #         # data.iloc[i,2] = data.iloc[i-1,2]
    #         data['position'][i] = data['position'][i-1]

    for i in range(1,len(data)):

        pt1 = p_t1(data.iloc[i-1,1],params,data.iloc[i-1,0],data.iloc[i,0])

        data.iloc[i,1] = pt1

        if data.iloc[i-1,2]==0 and pt1>=p_b:
            data.iloc[i,2] = 1
        elif data.iloc[i-1,2]==1 and pt1<=p_s:
            data.iloc[i,2] = 0
        else:
            data.iloc[i,2] = data.iloc[i-1,2]

    data['strat_ret'] = data['position'].shift(1) * data['ret']

    rho = (1+params['rho'])**(1/252) - 1

    data['strat_ret'][data['position'].shift(1)==0] = rho

    if position==0:
        data['strat_ret'][0] = rho
    else:
        data['strat_ret'][0] = data['ret'][0]

    # print(data)

    # bal = 1
    # last_buy = 0
    rets = []

    count = 0

    for i in range(1,len(data)):

        # if data.iloc[i-1,2]==0 and data.iloc[i,2]==1:
        if data['position'][i-1]==0 and data['position'][i]==1:

            bal*= (1+params['rho'])**((i - last_sell)/252)

            # last_buy = data.iloc[i,0]
            last_buy = data['Adj Close'][i]


        elif data['position'][i-1]==1 and data['position'][i]==0 and last_buy!=0:

            # bal *= data.iloc[i,0]/last_buy
            bal *= data['Adj Close'][i]/last_buy

            last_sell = i
            # print(last_buy)
            # rets.append(data.iloc[i,0]/last_buy - 1)
            rets.append(data['Adj Close'][i]/last_buy - 1)
            count+=1
            last_buy = 0

    if last and last_buy!=0:
        # bal *= data.iloc[-1,0]/last_buy
        bal *= data['Adj Close'][-1]/last_buy
        count+=1
        last_buy = 0

    if data['position'][-1]==0:
        bal*= (1+params['rho'])**((len(data) - last_sell)/252)

    strat_ret = bal/old_bal - 1

    bh_bal*=data['Adj Close'][-1]/data['Adj Close'][0]
    # try:
    bh_ret = bh_bal/old_bh_bal - 1
    # except:
    #     print(start_date)
    #     print(end_date)
    #     print(data)

    data['cumret'] = (1+ data['strat_ret']).cumprod() * cum_ret

    
    plot_df = data.iloc[:,1:2]
    plot_df['p_b'] = pd.Series([p_b]*len(data),index=data.index)
    plot_df['p_s'] = pd.Series([p_s]*len(data),index=data.index)

    plot_df = plot_df[['p_b','p_s','P(t)']]

    return strat_ret, bh_ret, p_b, p_s, pt1, data.iloc[-1,2], last_buy, bal, bh_bal, count, plot_df, data.iloc[-1,-1], list(data['cumret'])


# MAIN

def incr_date(date,n):
    return '-'.join([str(int(date.split('-')[0])+n)]+date.split('-')[1:])


df = yf.download("^GSPC", start="1962-01-03", end="2009-03-09").loc[:,['Adj Close']]
df.index = pd.to_datetime(df.index)

stats_df = get_sp_stats_df(df)

rho_df = yf.download("^TNX", start="1962-01-03", end="2009-03-09").loc[:,['Adj Close']]
rho_df.index = pd.to_datetime(rho_df.index)

pre_start_date = "1960-01-01"
in_start_date = incr_date(pre_start_date,3)
in_end_date = incr_date(in_start_date,10)



dataset = df.copy()

dataset['ret'] = dataset['Adj Close'].pct_change()

params_old = {}

position = 0
last_buy = 0
last = False
bal = 1
bh_bal = 1
plot_df = pd.DataFrame([],columns=['p_b','p_s','P(t)'])

p_i = None

count = 0

cum_ret = 1

cumrets_list = []

for i in range(50):

    if i==49:
        last = True

    
    start_date = incr_date(in_start_date,i)
    end_date = incr_date(start_date,10)
    
    result = {'start-date':end_date,'end-date':incr_date(end_date,1)}

    # stats_df = get_sp_stats_df(dataset[start_date:end_date])

    # params = calculate_params_v3(dataset[start_date:end_date],rho_df[start_date:end_date])
    params = calculate_params_v4(dataset[start_date:end_date],rho_df[start_date:end_date],stats_df[start_date:end_date])

    # break
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
    

    strat_ret, bh_ret, p_b, p_s, p_i, position, last_buy, bal, bh_bal, c, p_df, cum_ret, cumrets = backtest_v4(dataset[end_date:incr_date(end_date,1)],params,p_i,position,last,last_buy,bal,bh_bal,cum_ret)
    # strat_ret, bh_ret, p_b, p_s, p_i, position, last_buy, bal, bh_bal, c, p_df= backtest_v3(dataset[end_date:incr_date(end_date,1)],params,p_i,position,last,last_buy,bal,bh_bal)

    print(cum_ret)

    print(bal)

    # print(cum_ret/bal)

    print('\n\n')

    cumrets_list += cumrets

    plot_df = pd.concat([plot_df,p_df])

    # break

    count+=c

    result['strat_ret'] = strat_ret
    result['bh_ret'] = bh_ret
    result['p_b'] = p_b
    result['p_s'] = p_s

    for key in params.keys():
        result[key] = params[key]

    # results_df2 = pd.DataFrame(results2,columns=results2[0].keys()).round(3)

    # results_df2.to_excel('Results_v11.xlsx')

    print(bal)

    print(bh_bal)

    print(count)

    print(count/50)

    
        

        


