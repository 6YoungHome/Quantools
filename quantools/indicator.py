
"""stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
date = 'date' if 'date' not in kwargs else kwargs['date']
high = 'high' if 'high' not in kwargs else kwargs['high']
open = 'open' if 'open' not in kwargs else kwargs['open']
low = 'low' if 'low' not in kwargs else kwargs['low']
close = 'close' if 'close' not in kwargs else kwargs['close']
volume = 'volume' if 'volume' not in kwargs else kwargs['volume']
amount = 'amount' if 'amount' not in kwargs else kwargs['amount']

stock_code: 股票代码字段名称
date: 日期字段名称
high: 股票最高价字段名称
open: 股票开盘价字段名称
low: 股票最低价字段名称
close: 股票收盘价字段名称
volume: 股票成交量字段名称
amount: 股票成交额字段名称"""
import pandas as pd

def sma(stock_data: pd.DataFrame, window: int, field: str, **kwargs) -> pd.DataFrame:
    """计算移动平均值

    Args:
        stock_data (pd.DataFrame): 股票数据
        window (int): 移动窗口
        field (str): 计算移动均值的字段名称
        kwargs:
            stock_code: 股票代码字段名称
            date: 日期字段名称

    Returns:
        pd.DataFrame: 移动平均值
    """
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']

    stock_data[f'ma{window}'] = stock_data.groupby(stock_code)[field] \
        .rolling(window).mean().reset_index().rename(columns={field: f'ma{window}'})
    return stock_data[[stock_code, date, f'ma{window}']]


def smsd(stock_data: pd.DataFrame, window: int, field: str, **kwargs) -> pd.DataFrame:
    """计算移动标准差

    Args:
        stock_data (pd.DataFrame): 股票数据
        window (int): 移动窗口
        field (str): 计算移动标准差的字段名称
        kwargs:
            stock_code: 股票代码字段名称
            date: 日期字段名称
    Returns:
        _type_: 移动标准差
    """
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs  else kwargs['date']

    stock_data[f'msd{window}'] = stock_data.groupby(stock_code)[field] \
        .rolling(window).std().reset_index().rename(columns={field: f'msd{window}'})

    return stock_data[[stock_code, date, f'ma{window}']]


def bollinger_bond(stock_data: pd.DataFrame, window: int, n_std: int, field: str, **kwargs) -> pd.DataFrame:
    """计算布林带

    Args:
        stock_data (pd.DataFrame): _description_
        window (int): _description_
        n_std (int): _description_
        field (str): _description_

    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称

    Returns:
        _pd.DataFrame_: 布林带上下界、布林带均线与标准差
    """
    bb = sma(stock_data, window, field, kwargs)
    bb[f'ms{window}'] = smsd(stock_data, window, field, kwargs)
    bb['upbond'] = bb[f'ma{window}'] + n_std * bb[f'ms{window}']
    bb['downbond'] = bb[f'ma{window}'] + n_std * bb[f'ms{window}']
    return bb


def rsi(stock_data: pd.DataFrame, field: str, window: int=14, **kwargs) -> pd.DataFrame: 
    """计算RSI指标
    相对强度指数（RSI）

    Args:
        stock_data (pd.DataFrame): 股票数据
        field (str): 计算RSI指标的字段（收益率）
        window (int): RSI参数
    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称
    Returns:
        pd.DataFrame: RSI指标值
    """
    rsi = pd.DataFrame()
    rsi['gain'] = stock_data[field].apply(lambda x: x if x > 0 else 0)
    rsi['loss'] = stock_data[field].apply(lambda x: x if x < 0 else 0)
    rsi['avg_gain'] = sma(rsi, window, 'gain', **kwargs)
    rsi['avg_loss'] = sma(rsi, window, 'loss', **kwargs)
    rsi['rs'] = rsi['avg_gain'] / rsi['avg_loss']
    rsi['rsi'] = 100 - 100 / (1 - rsi['rs'])
    return rsi[['rsi']]


# def macd(stock_data, field, short=12, long=26, m=9, **kwargs):
#     """移动平均收敛/发散线（Moving Average Convergence/Divergence，MACD）

#     Args:
#         stock_data (_type_): _description_
#         field (_type_): _description_
#         short (_type_, optional): _description_. Defaults to 12.
#         long (_type_, optional): _description_. Defaults to 26.
#         m (_type_, optional): _description_. Defaults to 9.
#         kwargs:
#             stock_code: 股票代码字段名称
#             date: 日期字段名称
    
#     Returns:
#         _type_: MACD相关指标值
#     """
#     l_ma = sma(stock_data, long, field, **kwargs)
#     s_ma = sma(stock_data, short, field, **kwargs)
#     dif = s_ma - l_ma
#     dea = sma(dif, m, field, **kwargs)
#     macd = dif - dea

# def rvi:
#     """
#     相对活力指数(Relative Vigor Index)
#     """
#     a = Close - Open
#     b = (Close - Open) of Last Day
#     c = (Close - Open) of Two Days Ago
#     d = (Close - Open) of Three Days Ago

#     e = High - Low
#     f = (High - Low) of Last Day
#     g = (High - Low) of Two Days Ago
#     h = (High - Low) of Three Days Ago

#     NUMERATOR = [a+(2×b)+(2×c)+d]/6
#     DENOMINATOR = [e+(2×f)+(2×g)+h]/6
#     RVI = SMA of DENOMINATOR for N periods /SMA of NUMERATOR for N periods
#     ​
#     i=RVI Value of Last Day
#     j=RVI Value of Two Days Ago
#     k=RVI Value of Three Days Ago	 
#     Signal Line = [RVI+(2×i)+(2×j)+k]/6















