
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


def delay(stock_data: pd.DataFrame, n: int, field: str, **kwargs):
    """计算偏移n期的数据

    Args:
        stock_data (pd.DataFrame): 股票数据
        window (int): 移动窗口
        field (str): 计算移动均值的字段名称

    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称

    Returns:
        pd.DataFrame: 偏移后的数据框
    """
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']
    
    delay_data = stock_data.sort_values([stock_code, date])
    delay_data[f'{field}_delay{n}'] = delay_data.groupby(stock_code)[field].shift(n)
    return delay_data[[stock_code, date, f'{field}_delay{n}']]


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
    
    sma_df = stock_data.sort_values([stock_code, date])
    sma_df = sma_df.groupby(stock_code)[field].rolling(window).mean()
    sma_df = sma_df.reset_index(stock_code)[[field]].rename(columns={field: f'{field}_ma{window}'})
    sma_df = pd.concat([stock_data[[stock_code, date]], sma_df], axis=1)
    return sma_df


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
        pd.DataFrame: 移动标准差
    """
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']
    
    smsd_df = stock_data.sort_values([stock_code, date])
    smsd_df = smsd_df.groupby(stock_code)[field].rolling(window).std()
    smsd_df = smsd_df.reset_index(stock_code)[[field]].rename(columns={field: f'{field}_msd{window}'})
    smsd_df = pd.concat([stock_data[[stock_code, date]], smsd_df], axis=1)
    return smsd_df


def bollinger_bond(stock_data: pd.DataFrame, window: int, n_std: int, field: str, **kwargs) -> pd.DataFrame:
    """计算布林带(Bollinger Bond)

    Args:
        stock_data (pd.DataFrame): 股票数据
        window (int): 布林带参数
        n_std (int): 布林带参数
        field (str): 计算布林带的字段名

    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称

    Returns:
        pd.DataFrame: 布林带上中下线
    """
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']

    bb = stock_data.sort_values([stock_code, date])
    bb = sma(bb, window, field, **kwargs)
    bb[f'ms{window}'] = smsd(stock_data, window, field, **kwargs)[f'{field}_msd{window}']
    bb['upbond'] = bb[f'ma{window}'] + n_std * bb[f'ms{window}']
    bb['downbond'] = bb[f'ma{window}'] + n_std * bb[f'ms{window}']
    return bb.drop(f'ms{window}', axis=1)


def rsi(stock_data: pd.DataFrame, window: int=14, field: str='ret', **kwargs) -> pd.DataFrame: 
    """计算相对强度指数(RSI)指标

    Args:
        stock_data (pd.DataFrame): 股票数据
        window (int): RSI参数. 默认为14
        field (str): 计算RSI指标的字段. 默认为'ret'

    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称

    Returns:
        pd.DataFrame: RSI指标值
    """
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']

    rsi = stock_data.sort_values([stock_code, date])

    rsi['gain'] = stock_data[field].apply(lambda x: x if x > 0 else 0)
    rsi['loss'] = stock_data[field].apply(lambda x: x if x < 0 else 0)
    rsi['avg_gain'] = sma(rsi, window, 'gain', **kwargs)[[f'gain_msa{window}']]
    rsi['avg_loss'] = sma(rsi, window, 'loss', **kwargs)[[f'loss_msa{window}']]
    rsi['rs'] = rsi['avg_gain'] / rsi['avg_loss']
    rsi['rsi'] = 100 - 100 / (1 - rsi['rs'])

    return rsi[[stock_code, date, 'rsi']]


def macd(stock_data, field, short=12, long=26, m=9, **kwargs):
    """移动平均收敛/发散线(Moving Average Convergence/Divergence，MACD)

    Args:
        stock_data (pd.DatFrame): 股票数据框
        field (str): 计算指标的字段
        short (int, optional): 短线参数. Defaults to 12.
        long (int, optional): 长线参数. Defaults to 26.
        m (int, optional): 均线参数. Defaults to 9.

    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称
    
    Returns:
        pd.DataFrame: MACD相关指标值
    """
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']

    macd = stock_data.sort_values([stock_code, date])
    macd['long'] = sma(macd, long, field, **kwargs)
    macd['short'] = sma(macd, short, field, **kwargs)[[f'close_ma{short}']]
    macd['dif'] = macd['short'] - macd['long']
    macd['dea'] = sma(macd, m, field, **kwargs)[[f'dif_ma{m}']]
    macd['macd'] = macd['dif'] - macd['dea']
    return macd[[stock_code, date, 'macd']]


def rvi(stock_data: pd.DataFrame, window: int, **kwargs):
    """相对活力指数(Relative Vigor Index, RVI)

    Args:
        stock_data (pd.DatFrame): 股票数据框
        field (str): 计算指标的字段
        short (int, optional): 短线参数. Defaults to 12.
        long (int, optional): 长线参数. Defaults to 26.
        m (int, optional): 均线参数. Defaults to 9.

    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称
        high: 股票最高价字段名称
        open: 股票开盘价字段名称
        low: 股票最低价字段名称
        close: 股票收盘价字段名称
        
    Returns:
        pd.DataFrame: RVI相关指标值
    """

    high = 'high' if 'high' not in kwargs else kwargs['high']
    open = 'open' if 'open' not in kwargs else kwargs['open']
    low = 'low' if 'low' not in kwargs else kwargs['low']
    close = 'close' if 'close' not in kwargs else kwargs['close']
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']

    rvi_df = stock_data.sort_values([stock_code, date])

    rvi_df['co_range'] = rvi_df[close]-rvi_df[open]
    rvi_df['co_range_d1'] = delay(rvi_df, 1, 'co_range', **kwargs)[f'co_range_delay1']
    rvi_df['co_range_d2'] = delay(rvi_df, 2, 'co_range', **kwargs)[f'co_range_delay2']
    rvi_df['co_range_d3'] = delay(rvi_df, 3, 'co_range', **kwargs)[f'co_range_delay3']

    rvi_df['hl_range'] = rvi_df[high]-rvi_df[low]
    rvi_df['hl_range_d1'] = delay(rvi_df, 1, 'hl_range', **kwargs)[f'hl_range_delay1']
    rvi_df['hl_range_d2'] = delay(rvi_df, 2, 'hl_range', **kwargs)[f'hl_range_delay2']
    rvi_df['hl_range_d3'] = delay(rvi_df, 3, 'hl_range', **kwargs)[f'hl_range_delay3']

    rvi_df['numerator'] = (rvi_df['co_range']+2*rvi_df['co_range_d1']+2*rvi_df['co_range_d2']+rvi_df['co_range_d3'])/6
    rvi_df['denominator'] = (rvi_df['hl_range']+2*rvi_df['hl_range_d1']+2*rvi_df['hl_range_d2']+rvi_df['hl_range_d3'])/6
    
    use_rvi_df = sma(rvi_df, window, 'numerator', **kwargs)
    use_rvi_df[f'denominator_ma{window}'] = sma(rvi_df, window, 'denominator', **kwargs)[[f'denominator_ma{window}']]

    use_rvi_df['rvi'] = use_rvi_df[f'numerator_ma{window}'] / use_rvi_df['denominator_ma{window}']
    use_rvi_df['rvi_d1'] = delay(rvi_df, 1, 'rvi', **kwargs)[f'rvi_delay1']
    use_rvi_df['rvi_d2'] = delay(rvi_df, 2, 'rvi', **kwargs)[f'rvi_delay2']
    use_rvi_df['rvi_d3'] = delay(rvi_df, 3, 'rvi', **kwargs)[f'rvi_delay3']
    use_rvi_df['rvi_signal'] = (use_rvi_df['rvi']+2*use_rvi_df['rvi_d1']+2*use_rvi_df['rvi_d2']+use_rvi_df['rvi_d3'])/6

    return use_rvi_df[[stock_code, date, 'rvi', 'rvi_signal']]


def obv(stock_data: pd.DataFrame, **kwargs):
    """能量潮（OBV指标）

    Args:
        stock_data (pd.DatFrame): 股票数据框
    
    kwargs:
        stock_code: 股票代码字段名称
        date: 日期字段名称
        close: 股票收盘价字段名称
        volume: 股票成交量字段名称

    Returns:
        pd.DataFrame: OBV相关指标值
    """

    volume = 'volume' if 'volume' not in kwargs else kwargs['volume']
    close = 'close' if 'close' not in kwargs else kwargs['close']
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']
    date = 'date' if 'date' not in kwargs else kwargs['date']

    obv_df = stock_data.sort_values([stock_code, date])[[stock_code, date, close, volume]]
    obv_df['pre_close'] = delay(obv_df, 1, close, **kwargs)['close_delay1']
    obv_df['v_vol'] = (
        (obv_df[close]>obv_df['pre_close']).astype(int) - \
        (obv_df[close]<obv_df['pre_close']).astype(int)
    ) * obv_df[volume]
    obv_df['obv'] = obv_df['v_vol'].cumsum()

    return obv_df[[stock_code, date, 'obv']]








