import numpy as np
import pandas as pd

import datetime


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    _data = pd.read_csv(file_name, encoding='gb2312')
    _data = _data.loc[:, ['date', 'PriceMin']]
    _data.rename(columns={'date': 'datadate', 'PriceMin': 'close'}, inplace=True)

    print(_data['datadate'][0])


    _data['datadate'] = _data['datadate'].apply(
        lambda date: int(datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')))
    return _data

def preprocess_data(origin_data_path, moving_window=240):
    """data preprocessing pipeline"""
    df = load_dataset(file_name=origin_data_path)
    df['spread'] = df.close.diff()
    df['profitR'] = df['close'].pct_change(1)

    df.drop(index=[0], inplace=True)
    df.index = df.datadate.factorize()[0]
    calculate_history_spread(df=df, hist_window=20)
    calculate_MACD(df=df)
    calculate_RSI(df=df, window=20)



    normalize_moving(df=df, moving_window=moving_window)

    df.fillna(method='bfill', inplace=True)
    return df


def normalize_moving(df, epsilon=1e-8, clip=10, moving_window=240):
    '''
    均值方差标准化  用每个点前面一年的数据的均值标准差归一化
    '''
    length_df = len(df)
    n = moving_window
    mean_spread = []
    var_spread = []
    mean_MACD = []
    var_MACD = []
    mean_RSI = []
    var_RSI = []
    for i in range(length_df):
        if i < n-1:
            i = n-1
        tmp_spread = df['spread'][i-n+1:i]
        mean_spread.append(tmp_spread.mean())
        var_spread.append(tmp_spread.var())

        tmp_MACD = df['MACD'][i-n+1:i]
        mean_MACD.append(tmp_MACD.mean())
        var_MACD.append(tmp_MACD.var())

        tmp_RSI = df['RSI'][i-n+1:i]
        mean_RSI.append(tmp_RSI.mean())
        var_RSI.append(tmp_RSI.var())














    df['historySpreadNormalized'] = [[round(np.clip((df['historySpread'][i][j] - mean_spread[i]) / np.sqrt(
        var_spread[i] + epsilon), -clip, clip), 5) for j in range(len(df['historySpread'][i]))] for i in
                                     range(len(df['historySpread']))]


    df['MACDNormalized'] = [round(np.clip((df['MACD'][i] - mean_MACD[i]) / np.sqrt(
                                  var_MACD[i] + epsilon), -clip, clip), 5) for i in range(len(df['MACD']))]

    df['RSINormalized'] = [round(np.clip((df['RSI'][i] - mean_RSI[i]) / np.sqrt(
                                  var_RSI[i] + epsilon), -clip, clip), 5) for i in range(len(df['RSI']))]

    df['SpreadNormalized'] = [round(np.clip((df['spread'][i] - mean_spread[i]) / np.sqrt(
        var_spread[i] + epsilon), -clip, clip), 5) for i in range(len(df['spread']))]


def calculate_MACD(df):
    price = df['close'].tolist()
    preFastEMA = 0
    preSlowEMA = 0
    preSignalLine = 0
    MACDList = []
    for i in range(len(price)):
        fastEMA, preFastEMA = _update_ema(price[i], preFastEMA,12,i)
        slowEMA, preSlowEMA = _update_ema(price[i], preSlowEMA,26,i)
        MACDLine = fastEMA - slowEMA
        signalLine, preSignalLine = _update_ema(MACDLine, preSignalLine,9,i)
        histogram = MACDLine - signalLine
        MACDList.append(histogram)
    df['MACD'] = MACDList

def _update_ema(sample, pre_ema, length, index=0):
    if index >= 1:
        ema = (sample - pre_ema) * 2 / (1 + length) + pre_ema
    else:
        ema = sample
    pre_ema = ema
    return ema, pre_ema

def calculate_RSI(df,window=20):
    spread = df.spread.tolist()
    RSIList = []
    for i in range(len(spread)):
        if i < window:
            RSIList.append(50)
        else:
            AUL = np.sum([tmp if tmp > 0 else 0 for tmp in spread[i - window + 1:i + 1]])
            ADL = np.sum([-tmp if tmp < 0 else 0 for tmp in spread[i - window + 1:i + 1]])
            RSIList.append(AUL / (AUL + ADL) * 100)
    df['RSI'] = RSIList



def calculate_history_spread(df, hist_window=20):
    history_spread = []
    price = df['spread']
    for i in range(len(df.datadate)):
        if i < hist_window:

            temp_price = [price[0]] if i == 0 else list(price[:i+1])
            price_list = temp_price + [round(np.mean(temp_price), 2)] * (hist_window - i - 1)
            history_spread.append(price_list)
        else:
            history_spread.append(list(price[i - hist_window + 1:i + 1]))
    df['historySpread'] = [[round(i, 3) for i in tmp]for tmp in history_spread]

def calculate_history_MACD(df, hist_window=48):
    history_MACD = []

    MACD = df['MACD']

    for i in range(len(df.datadate)):
        if i < hist_window:

            temp_MACD = [MACD[0]] if i == 0 else list(MACD[:i+1])
            MACD_list = temp_MACD + [round(np.mean(temp_MACD),2)] * (hist_window - i - 1)
            history_MACD.append(MACD_list)
        else:
            history_MACD.append(list(MACD[i - hist_window + 1:i + 1]))
    df['historyMACD'] = history_MACD

def calculate_history_RSI(df, hist_window=48):
    history_RSI = []

    RSI = df['RSI']

    for i in range(len(df.datadate)):
        if i < hist_window:

            temp_RSI = [RSI[0]] if i == 0 else list(RSI[:i+1])
            RSI_list = temp_RSI + [round(np.mean(temp_RSI),2)] * (hist_window - i - 1)
            history_RSI.append(RSI_list)
        else:
            history_RSI.append(list(RSI[i - hist_window + 1:i + 1]))
    df['historyRSI'] = history_RSI



if __name__ == '__main__':


    origin_data_path = "continuousdata_DJI.csv"
    preprocess_data(origin_data_path)

    print(14)
