import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# 在我的个人网站www.6young.site中，分享过更多相关回测指标的实现方式
# 如果感兴趣可以进行查看（基于GitHub pages搭建，使用科学上网访问速度更快）
# 计算年(季/月/周)化收益的相关常数
BDAYS_PER_YEAR = 252
BDAYS_PER_QTRS = 63
BDAYS_PER_MONTH = 21
BDAYS_PER_WEEK = 5

DAYS_PER_YEAR = 365
DAYS_PER_QTRS = 90
DAYS_PER_MONTH = 30
DAYS_PER_WEEK = 7

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4


def get_period_days(period):
    '''不同时期指标转化'''
    period_days = {
        'yearly': BDAYS_PER_YEAR, 'quarterly': BDAYS_PER_QTRS,
        'monthly': BDAYS_PER_MONTH, 'weekly': BDAYS_PER_WEEK, 'daily': 1, 
        
        'monthly2yearly': MONTHS_PER_YEAR,
        'quarterly2yearly': QTRS_PER_YEAR, 
        'weekly2yearly': WEEKS_PER_YEAR, 
    }
    return period_days[period]

def annual_info(returns_df, period='yearly'):
    """
    年化收益与波动率
    """
    period_days = get_period_days(period)
    total_return = (returns_df + 1).prod(axis=0)
    annual_ret = total_return ** (period_days / returns_df.shape[0]) - 1
    annual_vol = returns_df.std() * (period_days ** 0.5)
    res_dict = {
        'annual_return': annual_ret,
        'annual_volatility': annual_vol,
    }
    return res_dict


def sharpe_ratio(returns_df, risk_free=0, period='yearly'):
    """
    计算（年化）夏普比率
    """
    period_days = get_period_days(period)
    sr = (returns_df.mean() - risk_free) * (period_days ** 0.5) / returns_df.std()
    res_dict = {'sharpe_ratio': sr}
    return res_dict


def maximum_drawdown(returns_df):
    """
    计算最大回撤
    """
    cum_returns = (returns_df + 1).cumprod(axis=0)
    peak = cum_returns.expanding().max()
    dd = ((peak - cum_returns)/peak)
    mdd = dd.max()
    end = dd[dd==mdd].dropna().index[0]
    start = peak[peak==peak.loc[end]].index[0]
    res_dict = {
        'max_drawdown': mdd, 'max_drawdown_start': start,
        'max_drawdown_end': end, 
    }
    return res_dict

def sortino_ratio(returns_df, minimum_acceptable_return=0, period='yearly'):
    """
    计算年化sortino比率
    """
    period_days = get_period_days(period)
    downside_returns = returns_df[returns_df < 0]  # 筛选出负收益
    downside_volatility = np.std(downside_returns, axis=0)
    excess_return = returns_df.mean() - minimum_acceptable_return
    sr = excess_return*(period_days**0.5) / downside_volatility
    res_dict = {'sortino_ratio': sr}
    return res_dict

def cal_ic(factor_b, return_n, method="normal"):
    """_summary_

    Args:
        factor_b (_pd.DataFrame_): t-1期因子数据
        return_n (_pd.DataFrame_): t期收益率数据（与factor相互对应）
        method (_str_, optional): 默认为"normal", 还可以是"rank".
    
    returns:
        ic: 信息系数
    """
    factor_b = factor_b.reset_index(drop=True)
    return_n = return_n.reset_index(drop=True)

    df = pd.concat([factor_b, return_n], axis=1)
    if method == "rank":
        df = df.rank()
    ic = df.corr().iloc[0,1]
    res_dict = {"IC": ic}
    return res_dict

def ic_ir(factor_df, return_df, method='normal'):
    """_summary_

    Args:
        factor_df (_pd.DataFrame_): columns: dates; index: tickers; values: factors
        return_df (_pd.DataFrame_): columns: dates; index: tickers; values: returns
        method (_str_, optional): 默认为"normal", 还可以是"rank".
    return:
        icir: ICIR
    """
    factors_b = factor_df.shift(1, axis=1)
    ic_ls = []
    for i in range(1, factors_b.shape[1]):
        date = factors_b.columns[i]
        ic = cal_ic(factors_b[date], return_df[date], method=method)["IC"]
        ic_ls.append(ic)
    ic_ir = np.mean(ic_ls) / np.std(ic_ls)
    res_dict = {"ICIR": ic_ir}
    return res_dict


def strategy_ir(return_p, return_i):
    """
    计算策略信息比率
    """
    excess_return = return_p-return_i
    ir = excess_return.mean() / excess_return.std()
    res_dict = {'strategy_infomation_ratio': ir}
    return res_dict


def win_rate(return_df):
    """_summary_

    Args:
        return_df (_pd.DataFrame_): 策略收益率数据框

    Returns:
        _type_: 策略胜率
    """
    win_trades = np.sum(return_df > 0)
    total_trades = len(return_df)
    res_dict = {"win_rate": win_trades / total_trades}
    return res_dict


def profit_loss_ratio(return_df):
    """_summary_

    Args:
        return_df (_pd.DataFrame_): 策略收益率数据框

    Returns:
        _type_: 策略盈亏比
    """
    total_profit = np.sum(return_df[return_df > 0])
    total_loss = abs(np.sum(return_df[return_df < 0]))
    res_dict = {"profit_loss_ratio": total_profit / total_loss}
    return res_dict

def calmar_ratio(returns_df, period='yearly'):
    """_summary_

    Args:
        returns_df (_type_): 策略收益率数据框
        period (_type_): 调整期限

    Returns:
        _type_: 策略的卡玛比率
    """
    annual_ret = annual_info(returns_df, period)['annual_return']
    mdd = maximum_drawdown(returns_df)['max_drawdown']
    cr = annual_ret / mdd
    res_dict = {'calmar_ratio': cr}
    return res_dict

def sterling_ratio(returns_df, period='yearly'):
    period_days = get_period_days(period)
    s = 0
    for i in range(3):
        returns_df_y = returns_df.iloc[-1-(i+1)*period_days:-1-i*period_days]
        s += maximum_drawdown(returns_df_y)['max_drawdown']
    s = s / 3 + 0.1
    slr = annual_info(returns_df, period=period)['annual_return'] / s
    res_dict = {'sterling_ratio': slr}
    return res_dict

def alpha_beta(ret_i, ret_m, risk_free=0, period='yearly'):
    """
    回归计算策略beta与alpha值
    """
    period_days = get_period_days(period)
    LR = LinearRegression()
    LR.fit(pd.DataFrame(ret_m-risk_free), pd.DataFrame(ret_i-risk_free))
    beta = LR.coef_[0][0]
    alpha = (1+LR.intercept_[0])**period_days-1
    res_dict = {'alpha': alpha, 'Beta': beta}
    return res_dict

def treynor_ratio(returns_df, ret_m, risk_free=0, period='yearly'):
    """_summary_

    Args:
        returns_df (_type_): 策略收益率数据框
        ret_m (_type_): 市场收益率数据框
        risk_free (_type_, optional): 无风险收益率.默认为0.
        period (_type_, optional): 调整期限. 默认为'yearly'（年化）.

    Returns:
        _type_: _description_
    """
    period_days = get_period_days(period)
    rp_f = (returns_df.mean() - risk_free)
    beta = alpha_beta(returns_df, ret_m, risk_free, period)['beta']
    tr = rp_f * (period_days ** 0.5) / beta
    res_dict = {'treynor_ratio': tr}
    return res_dict


def omega_ratio(returns_df, threshold=0, period='daily'):
    """
    此处的period参数为说明输入的threshold为年化收益率还是日收益率
    """
    period_days = get_period_days(period)
    returns_less_thresh = returns_df - (1+threshold)**(1/period_days)+1
    up = returns_less_thresh[returns_less_thresh > 0].sum()
    down = -1 * returns_less_thresh[returns_less_thresh < 0].sum()
    omega = up / down
    res_dict = {'omega': omega}
    return res_dict