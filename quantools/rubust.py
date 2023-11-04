from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm


def winsorize(factor, fac_name, method="quantile", **kwargs):
    """
    为了保证因子的稳定性,对因子进行截尾,
    可以选择均值标准差方法或者分位数方法截尾

    Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_name (_str_): 准备温莎处理的因子名称
        method (_str_ or _int_, optional): 选取的方法;
            "quantile" or 0: 分位数结尾,一般取0.01
            "sigma" or 1: 根据正态分布特征的sigma原则截尾,一般取3
        kwargs:
            q: quantile方法下的参数
            n: sigma方法下的参数
            date: factor数据框中时间对应的列名，未定义时默认为date
    """
    def _winsorize(col, fac_name):
        if col[fac_name] > col['upper']:
            return col['upper']
        elif col[fac_name] < col['lower']:
            return col['lower']
        else:
            return col[fac_name]
    
    if method == "quantile" or method == 0:
        if 'q' not in kwargs:
            print("使用quantile方法去极值时需要有参数q")
            print("q取值为(0-0.5)，意为将q分位数两端的数据截尾，一般取0.01")
            return 
    elif method == "sigma" or method == 1:
        if 'n' not in kwargs:
            print("使用sigma方法去极值时需要有参数n")
            print("n取值为正整数，意为将因子nσ两端的数据截尾,一般取3")
            return 
    else:
        return
    
    date = 'date' if 'date' not in kwargs else kwargs['date']

    factors = factor.copy()
    if method == "quantile":
        bound = factors.groupby(date)[fac_name].quantile(1-kwargs['q']) \
            .reset_index().rename(columns={fac_name: 'upper'})
        bound['lower'] = factors.groupby(date)[fac_name].quantile(kwargs['q']) \
            .reset_index()[fac_name]
        
    if method == "sigma":
        avg = factors.groupby(date)[fac_name].mean().reset_index()
        std = factors.groupby(date)[fac_name].std().reset_index()
        bound['upper'] = avg + kwargs['n'] * std
        bound['lower'] = avg - kwargs['n'] * std
    factors = factors.merge(bound, on=date)

    tqdm.pandas(desc=f"Winsorizing the factor {fac_name}: ")
    factors[fac_name] = factors.progress_apply(lambda x: _winsorize(x, fac_name), axis=1)
    factors = factors.drop(['upper', 'lower'], axis=1)

    return factors

def standardize(factor, fac_names, **kwargs):
    """
    为了保证因子的稳定性,对因子进行截尾,
    可以选择均值标准差方法或者分位数方法截尾

    Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_names (_list(str)_): 准备标准化的的因子名称列表
        kwargs:
            date: factor数据框中时间对应的列名，未定义时默认为date
    """

    factors = factor.copy()
    date = 'date' if 'date' not in kwargs else kwargs['date']
    for fac_name in tqdm(fac_names, desc="Standardizing: "):
        factors[fac_name] = factors.groupby(date)[fac_name] \
            .apply(lambda x:(x - x.mean()) / x.std())
    return factors



def industry_neutralise(factor, fac_name, method=0, **kwargs):
    """
    对因子在行业上的表现进行中性化

    Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_name (_str_): 准备中性化的因子名称
        method (_str_ or _int_, optional): 选取的方法;
            "regression" or 0: 使用回归法进行中性化
            "group_std" or 1: 使用分组标准化法进行中性化
        kwargs:
            date: factor数据框中时间对应的列名，未定义时默认为"date"
            industry: factor数据框中行业对应的列名，未定义时默认为"industry"

    Returns:
        _pd.DataFrame_:中性化后的数据框
    """
    factors = factor.copy()
    date = 'date' if 'date' not in kwargs else kwargs['date']
    industry = 'industry' if 'industry' not in kwargs else kwargs['industry']

    if method == "regression" or method == 0:
        print("使用回归法进行中性化")
        dummies = pd.get_dummies(factors[industry])
        factors = pd.concat([factors, dummies], axis=1)
        dummies_col = list(dummies.columns)

        factors[fac_name] = factors.groupby(date)[fac_name] \
            .apply(lambda x: sm.OLS(x[fac_name],x[dummies_col]).fit().resid)

    elif method == "group_std" or method == 1:
        print("使用分组标准化法进行中性化")
        factors[fac_name] = factors.groupby([date, industry])[fac_name] \
            .apply(lambda x:(x - x.mean()) / x.std())

    else:
        print(f"不存在‘{method}’中性化方法")

    return factors


def neutralise(factor, fac_name, style_names, **kwargs):
    """
    对因子在行业上的表现进行中性化

    Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_name (_str_): 准备中性化的因子名称
        style_name (_list(_str_)_): 风格因子列表，准备消除的影响
        kwargs:
            date: factor数据框中时间对应的列名，未定义时默认为"date"

    Returns:
        _pd.DataFrame_:中性化后的数据框
    """
    factors = factor.copy()
    date = 'date' if 'date' not in kwargs else kwargs['date']
    
    factors[fac_name] = factors.groupby(date)[fac_name] \
        .apply(lambda x: sm.OLS(x[fac_name],x[style_names]).fit().resid)
    
    return factors