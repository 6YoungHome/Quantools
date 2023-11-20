from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections.abc import Iterable
from .error import *


def _mad_winsorize(groups, n):
    med = groups.median()
    mad = abs(groups - med).median()
    up = med + n * 1.4826 * mad
    low = med - n * 1.4826 * mad
    groups = groups.clip(upper=up, lower=low)
    return groups


def _q_winsorize(groups, q):
    groups = groups.clip(upper=groups.quantile(1-q), lower=groups.quantile(q))
    return groups


def _std_winsorize(groups, n):
    avg = groups.mean()
    std = groups.std()
    groups = groups.clip(upper=avg+n*std, lower=avg-n*std)
    return groups


def winsorize(factor: pd.DataFrame, fac_name: str, method: str="quantile", **kwargs):
    """
    为了保证因子的稳定性,对因子进行截尾,
    可以选择均值标准差方法或者分位数方法截尾

    Args:
        factor (pd.DataFrame): 因子数据框
        fac_name (str): 准备温莎处理的因子名称
        method (str or int, optional): 选取的方法;
            "quantile": 分位数结尾
            "sigma": 根据正态分布特征的sigma原则截尾
            "MAD": 中位数绝对偏差去极值方法
    kwargs:
        q: quantile方法下的参数,一般取0.01
        n: sigma or MAD方法下的参数,一般取3
        date: factor数据框中时间对应的列名，未定义时默认为date

    Returns:
        pd.DataFrame: 处理后的因子数据框
    """
    date = 'date' if 'date' not in kwargs else kwargs['date']
    tqdm.pandas(desc=f"Winsorizing the factor {fac_name}: ")

    if method == "quantile":
        if 'q' not in kwargs:
            raise MissArgsError("q", "quantile方法下的参数,一般取0.01")
        factor[fac_name] = factor.groupby(date)[fac_name]. \
            progress_apply(lambda x: _q_winsorize(x, kwargs['q']))
        
    elif method == "sigma" or method == 1:
        if 'n' not in kwargs:
            raise MissArgsError("n", "sigma方法下的参数,一般取3")
        factor[fac_name] = factor.groupby(date)[fac_name]. \
            progress_apply(lambda x: _std_winsorize(x, kwargs['n']))
        
    elif method == "MAD" or method == 2:
        if 'n' not in kwargs:
            raise MissArgsError("n", "MAD方法下的参数,一般取3")
        factor[fac_name] = factor.groupby(date)[fac_name]. \
            progress_apply(lambda x: _mad_winsorize(x, kwargs['n']))
    else:
        raise InvalidArgsError("method", method)

    return factor

def standardize(factor: pd.DataFrame, fac_names: [str], **kwargs):
    """
    为了保证因子的稳定性,对因子进行标准化

    Args:
        factor (pd.DataFrame): 因子数据框
        fac_names ([str]): 准备标准化的的因子名称列表

    kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为date
    
    Returns:
        pd.DataFrame: 处理后的因子数据框
    """

    factors = factor.copy()
    date = 'date' if 'date' not in kwargs else kwargs['date']
    for fac_name in tqdm(fac_names, desc="Standardizing: "):
        factors[fac_name] = factors.groupby(date)[fac_name] \
            .apply(lambda x:(x - x.mean()) / x.std())
    return factors



def industry_neutralise(factor: pd.DataFrame, fac_name: str, method=0, **kwargs):
    """
    对因子在行业上的表现进行中性化

    Args:
        factor (pd.DataFrame): 因子数据框
        fac_name (str): 准备中性化的因子名称
        method (str or int): 选取的方法;
            "regression" or 0: 使用回归法进行中性化
            "group_std" or 1: 使用分组标准化法进行中性化
    kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"
        industry: factor数据框中行业对应的列名，未定义时默认为"industry"

    Returns:
        pd.DataFrame:中性化后的数据框
    """
    factors = factor.copy()
    date = 'date' if 'date' not in kwargs else kwargs['date']
    industry = 'industry' if 'industry' not in kwargs else kwargs['industry']

    if method == "regression" or method == 0:
        tqdm.pandas(desc="使用回归法进行中性化")
        dummies = pd.get_dummies(factors[industry])
        factors = pd.concat([factors, dummies], axis=1)
        dummies_col = list(dummies.columns)

        factors[fac_name] = factors.groupby(date)[fac_name] \
            .progress_apply(lambda x: sm.OLS(x[fac_name],x[dummies_col]).fit().resid)

    elif method == "group_std" or method == 1:
        tqdm.pandas(desc="使用分组标准化法进行中性化")
        factors[fac_name] = factors.groupby([date, industry])[fac_name] \
            .progress_apply(lambda x:(x - x.mean()) / x.std())

    else:
        raise InvalidArgsError("method", method, )

    return factors


def regression_neutralise(factor: pd.DataFrame, fac_name: str, style_names: [str], **kwargs):
    """
    对因子在各类风险因子上的表现进行回归中性化

    Args:
        factor (pd.DataFrame): 因子数据框
        fac_name (str): 准备中性化的因子名称
        style_names ([str]): 风格因子列表，准备消除的影响
    kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"

    Returns:
        pd.DataFrame:中性化后的数据框
    """
    factors = factor.copy()
    date = 'date' if 'date' not in kwargs else kwargs['date']
    tqdm.pandas(desc="Regression neutralising: ")
    factors[fac_name] = factors.groupby(date) \
        .progress_apply(lambda x: sm.OLS(x[fac_name],x[style_names]).fit().resid).reset_index('date')[0]
    
    return factors


def group_neutralise(factor: pd.DataFrame, fac_name: str, style_name: str, groups:int or Iterable, **kwargs):
    """
    对因子在各类风险因子上的表现进行分组中性化

    Args:
        factor (pd.DataFrame): 因子数据框
        fac_name (str): 准备中性化的因子名称
        style_name (str): 风格因子，准备消除的影响
        groups (int or Iterable): 分组数量；或者一个0~1的列表，作为分组切割方案
    kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"

    Returns:
        pd.DataFrame:中性化后的数据框
    """
    if isinstance(groups, Iterable):
        labels = range(1, len(groups))
    else:
        labels = range(1, groups+1)
        groups = np.linspace(0,1,groups+1)

    date = 'date' if 'date' not in kwargs else kwargs['date']
    gn_fac = factor.sort_values(date)
    gn_fac['rank'] = gn_fac.groupby(date)[style_name] \
        .apply(lambda x: x.rank(method='first')/x.count())
    gn_fac['level']=pd.cut(gn_fac['rank'], groups, right=True, labels=labels)

    tqdm.pandas(desc="Neutralising: ")
    gn_fac[fac_name] = gn_fac.groupby([date, 'level'])[fac_name] \
        .progress_apply(lambda x:(x - x.mean()) / x.std())
    gn_fac = gn_fac.drop(['level', 'rank'], axis=1)
    
    return gn_fac


