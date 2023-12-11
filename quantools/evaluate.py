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
    """计算策略年化收益与年化波动率

    Args:
        returns_df (pd.DataFrame): 策略收益率数据框
        period (str): 调整期限. 默认为'yearly'（年化）.

    Returns:
        dict: 指标字典，索引分别为annual_return, annual_volatility
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
    """计算策略夏普比率

    Args:
        returns_df (pd.DataFrame): 策略收益率数据框
        risk_free (float, optional): 无风险收益. 默认为0
        period (str): 调整期限. 默认为'yearly'（年化）.

    Returns:
        dict: 指标字典，索引为sharpe_ratio
    """
    period_days = get_period_days(period)
    sr = (returns_df.mean() - risk_free) * (period_days ** 0.5) / returns_df.std()
    res_dict = {'sharpe_ratio': sr}
    return res_dict


def maximum_drawdown(returns_df):
    """计算策略最大回撤

    Args:
        returns_df (pd.DataFrame): 策略收益率数据框

    Returns:
        dict: 指标字典，索引为max_drawdown, max_drawdown_start, max_drawdown_end
    """
    cum_returns = (returns_df + 1).cumprod(axis=0)
    peak = cum_returns.expanding().max()
    dd = ((peak - cum_returns)/peak)
    mdd = dd.max()
    end = dd[dd==mdd].dropna().index[0]
    start = peak[peak==peak.loc[end]].dropna().index[0]
    if not isinstance(mdd, float):
        mdd = mdd.values[0]
        
    res_dict = {
        'max_drawdown': mdd, 'max_drawdown_start': start,
        'max_drawdown_end': end, 
    }
    return res_dict

def sortino_ratio(returns_df, minimum_acceptable_return=0, period='yearly'):
    """计算策略年化sortino比率

    Args:
        returns_df (pd.DataFrame): 策略收益率数据框
        minimum_acceptable_return (float, optional): 最低可接受收益率. 默认为0
        period (str): 调整期限. 默认为'yearly'（年化）.

    Returns:
        dict: 指标字典，索引为sortino_ratio
    """
    period_days = get_period_days(period)
    downside_returns = returns_df[returns_df < 0]  # 筛选出负收益
    downside_volatility = np.std(downside_returns, axis=0)
    excess_return = returns_df.mean() - minimum_acceptable_return
    sr = excess_return*(period_days**0.5) / downside_volatility
    res_dict = {'sortino_ratio': sr}
    return res_dict

def cal_ic(factor_b, return_n, method="normal"):
    """计算因子的IC值

    Args:
        factor_b (pd.DataFrame): t-2期因子数据
        return_n (pd.DataFrame): t期收益率数据（与factor相互对应）
        method (str, optional): 默认为"normal", 还可以是"rank".
    
    returns:
        dict: 指标字典，索引为IC
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
    """计算策略的IC信息比率

    Args:
        factor_df (pd.DataFrame): 列为dates; 行为tickers; 值为因子值的数据框
        return_df (pd.DataFrame): 列为dates; 行为tickers; 值为收益率值的数据框
        method (str, optional): 默认为"normal", 还可以是"rank".

    return:
        dict: 指标字典，索引为ICIR
    """
    factors_b = factor_df.shift(2, axis=1)
    ic_ls = []
    for i in range(1, factors_b.shape[1]):
        date = factors_b.columns[i]
        ic = cal_ic(factors_b[date], return_df[date], method=method)["IC"]
        ic_ls.append(ic)
    ic_ir = np.mean(ic_ls) / np.std(ic_ls)
    res_dict = {"ICIR": ic_ir}
    return res_dict


def strategy_ir(return_p, base_return):
    """计算策略的策略信息比率

    Args:
        return_p (pd.DataFrame): 策略收益率数据框
        base_return (pd.DataFrame): 基准收益率数据框

    Returns:
        dict: 指标字典，索引为strategy_infomation_ratio
    """
    excess_return = return_p-base_return
    ir = excess_return.mean() / excess_return.std()
    res_dict = {'strategy_infomation_ratio': ir}
    return res_dict


def win_rate(return_df, base_df=0):
    """计算策略胜率

    Args:
        return_df (pd.DataFrame): 策略收益率数据框
        base_df (pd.DataFrame or int, optional): 用以比较的基础收益率列表. 默认为0.

    Returns:
        dict: 指标字典，索引为win_rate
    """

    win_trades = np.sum(return_df > base_df)
    total_trades = len(return_df)
    res_dict = {"win_rate": win_trades / total_trades}
    return res_dict


def profit_loss_ratio(return_df):
    """计算策略盈亏比

    Args:
        return_df (pd.DataFrame): 策略收益率数据框

    Returns:
        dict: 指标字典，索引为profit_loss_ratio
    """
    total_profit = np.sum(return_df[return_df > 0])
    total_loss = abs(np.sum(return_df[return_df < 0]))
    res_dict = {"profit_loss_ratio": total_profit / total_loss}
    return res_dict

def calmar_ratio(returns_df, period='yearly'):
    """计算策略的calmar比率

    Args:
        returns_df (pd.DataFrame): 策略收益率数据框
        period (str): 调整期限. 默认为'yearly'（年化）.

    Returns:
        dict: 指标字典，索引为calmar_ratio
    """
    annual_ret = annual_info(returns_df, period)['annual_return']
    mdd = maximum_drawdown(returns_df)['max_drawdown']
    cr = annual_ret / mdd
    res_dict = {'calmar_ratio': cr}
    return res_dict

def sterling_ratio(returns_df, period='yearly'):
    """计算策略的sterling比率

    Args:
        returns_df (pd.DataFrame): _description_
        period (str, optional): _description_. Defaults to 'yearly'.

    Returns:
        dict: 指标字典，索引为sterling_ratio
    """
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
    """计算策略的alpha和beta

    Args:
        returns_df (pd.DataFrame): 策略收益率数据框
        ret_m (pd.DataFrame): 市场收益率数据框
        risk_free (float, optional): 无风险收益率.默认为0.
        period (str, optional): 调整期限. 默认为'yearly'（年化）.

    Returns:
        dict: 指标字典，索引分别为alpha，beta
    """
    period_days = get_period_days(period)
    LR = LinearRegression()
    LR.fit(pd.DataFrame(ret_m-risk_free), pd.DataFrame(ret_i-risk_free))
    beta = LR.coef_[0][0]
    alpha = (1+LR.intercept_[0])**period_days-1
    res_dict = {'alpha': alpha, 'Beta': beta}
    return res_dict

def treynor_ratio(returns_df, ret_m, risk_free=0, period='yearly'):
    """计算策略的treynor_ratio

    Args:
        returns_df (pd.DataFrame): 策略收益率数据框
        ret_m (pd.DataFrame): 市场收益率数据框
        risk_free (float, optional): 无风险收益率.默认为0.
        period (str, optional): 调整期限. 默认为'yearly'（年化）.

    Returns:
        dict: 指标字典，索引为treynor_ratio
    """
    period_days = get_period_days(period)
    rp_f = (returns_df.mean() - risk_free)
    beta = alpha_beta(returns_df, ret_m, risk_free, period)['beta']
    tr = rp_f * (period_days ** 0.5) / beta
    res_dict = {'treynor_ratio': tr}
    return res_dict


def omega_ratio(returns_df, threshold=0, period='daily'):
    """计算策略的omega_ratio

    Args:
        returns_df (pd.DataFrame): _description_
        threshold (float, optional): _description_. Defaults to 0.
        period (str, optional): _description_. Defaults to 'daily'.

    Returns:
        dict: 指标字典，索引为omega
    """
    period_days = get_period_days(period)
    returns_less_thresh = returns_df - (1+threshold)**(1/period_days)+1
    up = returns_less_thresh[returns_less_thresh > 0].sum()
    down = -1 * returns_less_thresh[returns_less_thresh < 0].sum()
    omega = up / down
    res_dict = {'omega': omega}
    return res_dict