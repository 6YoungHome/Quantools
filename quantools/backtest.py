from tqdm import tqdm, trange
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from .evaluate import *
from .paint import *


def fama_macbeth(factor, fac_name, **kwargs):
    """返回fama_macbeth回归检验参数

    ##Args:
        factor (pd.DataFrame): _description_
        fac_name (str): _description_
    
    ##kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"
        pred_rtn: factor数据框中预测收益率对应的列名，未定义时默认为"pred_rtn"
    
    ##Returns:
        Fama-Macbeth检验结果字典，包含t值、p值、正/负数量
    """
    def cal_beta(data, fac_name, **kwargs):
        """线性回归获得斜率参数"""
        pred_rtn = 'pred_rtn' if 'pred_rtn' not in kwargs else kwargs['pred_rtn']
        x = data[fac_name].values.reshape((-1, 1))
        y = data[pred_rtn].values
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_[0]
    
    date = 'date' if 'date' not in kwargs else kwargs['date']
    
    factors = factor.copy()
    factors = factors[~factors[fac_name].isna()]
    tqdm.pandas(desc=f"Fama-MacBeth Regression by factor {fac_name}: ")
    fg = factors.groupby(date).progress_apply(lambda x: cal_beta(x, fac_name, **kwargs)).reset_index()
    t, p = stats.ttest_1samp(fg[0], 0)
    pos_count, neg_count = (fg[0]>0).sum(),(fg[0]<0).sum()
    res_dict = {
        'fac_name': fac_name, 't': t, 'p': p,
        'pos_count': pos_count, 'neg_count': neg_count
    }
    return res_dict


def group_return_analysis(factor, fac_name, group_num=10, plot=True, **kwargs):
    """因子分组检测

    ##Args:
        factor (pd.DataFrame): 因子与因子对应产生的收益率
        fac_name (str): 因子名称
        group_num (int, optional): 分组数量， 默认为10.
        plot (bool, optional): 是否绘图. 默认绘图.
    
    ##kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"
        pred_rtn: factor数据框中预测收益率对应的列名，未定义时默认为"pred_rtn"

    ##Returns:
        pd.DataFrame: 分组收益率情况
    """
    date = 'date' if 'date' not in kwargs else kwargs['date']
    pred_rtn = 'pred_rtn' if 'pred_rtn' not in kwargs else kwargs['pred_rtn']

    factors = factor.copy()
    factors = factors[~factors[fac_name].isna()]
    factors['rank'] = factors.groupby(date)[fac_name].rank(method='first')
    factors['group'] = factors.groupby(date)['rank'].apply(lambda x: pd.qcut(x, group_num, range(group_num)))

    group_cum_rtns = pd.DataFrame()
    group_rtns = pd.DataFrame()
    for i in trange(group_num, desc="Calculating groups"):
        rtn = factors[factors['group']==i].groupby(date).mean()[[pred_rtn]]
        rtn['cum_rtn'] = (rtn[pred_rtn] + 1).cumprod()
        group_cum_rtns[i] = rtn['cum_rtn']
        group_rtns[i] = rtn[pred_rtn]
    
    if plot:
        fig = plt.figure(figsize=(16, 5))
        ax1, ax2 = fig.subplots(1, 2)
        group_cummulate_returns_painter(group_rtns, ax1)
        ax1.set_title(f"{group_num} group return of factor {fac_name}")
        ax1.legend()

        group_annual_returns_painter(group_rtns, ax2)
        ax2.set_title(f"Annual return of {group_num} groups")
    return group_rtns


def get_strategy_rtn(factor, fac_name, reverse=False, n=100, head=1, **kwargs):
    """根据因子计算策略的收益

    ##Args:
        factor (pd.DataFrame): 因子与其预测收益数据框
        fac_name (str): 因子名称
        reverse (bool): 因子作用方向. 默认为False即因子越大收益越高.
        n (int,): 每周期选股数量. 默认为100只.
        head(float, optional): 从第几只股票开始选，默认从头开始(1),若小于1，则从head分位数开始
        
    ##kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"
        pred_rtn: factor数据框中预测收益率对应的列名，未定义时默认为"pred_rtn"

    ##Returns:
        pd.DataFrame: 策略收益数据框
    """
    date = 'date' if 'date' not in kwargs else kwargs['date']
    pred_rtn = 'pred_rtn' if 'pred_rtn' not in kwargs else kwargs['pred_rtn']

    factors = factor.copy()
    factors = factors[~factors[fac_name].isna()]
    factors['rank'] = factors.groupby(date)[fac_name].rank(method='first', ascending=reverse)
    if head>=1:
        factors = factors[(factors['rank']<(n+head)) & (factors['rank']>=head)]
    else:
        factors[factors.groupby(date)['rank'] \
                .apply(lambda x: (x>=int(x.quantile(head))) & (x<int(x.quantile(head))+n))]
    rtn = factors.groupby(date)[pred_rtn].mean().reset_index()
    rtn['cum_rtn'] = (1+rtn[pred_rtn]).cumprod()
    rtn = rtn.set_index(date)
    return rtn


def evaluate_strategy(return_df, yearly_evaluate=True, **kwargs):
    """根据收益率序列计算各类回测指标

    ##Args:
        return_df (pd.DataFrame): _description_
        yearly_evaluate (bool): 是否对策略进行分年评价，默认进行分年评价True

    ##kwargs:
        pred_rtn: factor数据框中预测收益率对应的列名，未定义时默认为"pred_rtn"
    
    ##Returns:
        pd.DataFrame: 策略评价指标数据框
    """
    def _get_index(return_df):
        """
        计算评价策略的部分指标
        """
        res_dict = {}
        res_dict.update(sharpe_ratio(return_df))
        res_dict.update(maximum_drawdown(return_df))
        res_dict.update(sortino_ratio(return_df))
        res_dict.update(annual_info(return_df))
        return res_dict
    
    pred_rtn = 'pred_rtn' if 'pred_rtn' not in kwargs else kwargs['pred_rtn']

    res_dict = _get_index(return_df[pred_rtn])
    res_dict['section'] = 'Sum'
    res_dict_ls = [res_dict]
    if yearly_evaluate:
        years = [i for i in range(return_df.index[0].year, return_df.index[-1].year+1)]
        for year in years:
            res_dict = _get_index(return_df.loc[str(year), pred_rtn])
            res_dict['section'] = str(year)
            res_dict_ls.append(res_dict)
    evaluate_result = pd.DataFrame(res_dict_ls)
    return evaluate_result


def backtest_nstock(factor, fac_name, reverse=False, n=100, head=1, plot=True, yearly_evaluate=True, **kwargs):
    """
    利用因子计算策略的收益率，绘图，并计算相关指标
    ##Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_name (_str_): 因子名称
        reverse (_bool_, optional): 正向因子or负向因子
        n (_int_, optional): 选取股票的数量（默认100只）
        head(float, optional): 从第几只股票开始选，默认从头开始(1),若小于1，则从head分位数开始
        plot (_bool_, optional): 是否绘图（默认绘图）
        yearly_evaluate (_bool_, optional): 是否评价策略每年情况（默认评价）

    ##kwargs:
        pred_rtn: factor数据框中预测收益率对应的列名，未定义时默认为"pred_rtn"

    ##Returns:
        pd.DataFrame: 策略收益数据框
        pd.DataFrame: 策略评价结果
    """
    pred_rtn = 'pred_rtn' if 'pred_rtn' not in kwargs else kwargs['pred_rtn']

    rtn = get_strategy_rtn(factor, fac_name, reverse, n, head)
    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.subplots(1, 1)
        diverse_cummulate_painter(rtn[[pred_rtn]], ax)
        ax.set_title(f"Cummulate return of factor {fac_name}")
    evaluate_result = evaluate_strategy(rtn[[pred_rtn]], yearly_evaluate)
    return rtn, evaluate_result


def mutifactor_score(factor, fac_names, group_num=10, stock_num=100, plot=True, **kwargs):
    """
    一个简单的通过打分的多因子选股方法
    ##Args:
        factor (pd.DataFrame): 因子数据框
        fac_names (list): 准备使用的多因子列表（反向因子前加"-"号即可）
        group_num (int, optional): 分组组数，默认为10
        stock_num (int, optional): 打分选股股票数，默认为100
        plot (bool, optional): 是否画图，默认为是
    
    ##kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"

    ##Returns:
        rtn (pd.DataFrame): 策略收益数据框
        evaluate_result: 策略评价结果数据框
    """
    def fac_name_parse(fac_name):
        """解析因子名，判断是否为反向因子"""
        if fac_name[0] == '-':
            return fac_name[1::], False
        else:
            return fac_name, True
    date = 'date' if 'date' not in kwargs else kwargs['date']
    
    factors = factor.copy()
    factors = factors.dropna()
    for fac_name in fac_names:
        fac_name, reverse = fac_name_parse(fac_name)
        factors['rank'] = factors.groupby(date)[fac_name] \
            .rank(method='first', ascending=reverse)
        factors[fac_name+'_score'] = factors.groupby(date)['rank'] \
            .apply(lambda x: pd.qcut(x, group_num, range(group_num)))
        
    factors['score'] = factors.loc[:, factors.columns.str.contains('score')].sum(axis=1)
    
    rtn, evaluate_result = backtest_nstock(factors, 'score', plot=plot, n=stock_num)
    return rtn, evaluate_result


def mutifactor_regression(factor, fac_names, stock_num=100, plot=True, **kwargs):
    """
    一个简单的通过打分的多因子选股方法
    ##Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_names (_list_): 准备使用的多因子列表（反向因子前加"-"号即可）
        stock_num (_int_, optional): _description_. 打分选股股票数，默认为100
        plot (_bool_, optional): _description_. 是否画图，默认为是
    
    ##kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"

    ##Returns:
        rtn (pd.DataFrame): 策略收益数据框
        evaluate_result: 策略评价结果数据框
    """
    def cal_beta(data, fac_names):
        """
        通过线性回归获得斜率参数
        按次序依此为alpha与beta_i
        """
        data = data.dropna()
        x = data[fac_names].values.reshape((-1, len(fac_names)))
        y = data['pred_rtn'].values
        model = LinearRegression()
        model.fit(x, y)
        return tuple([model.intercept_]+[i for i in model.coef_])
        # return model.intercept_, *model.coef_
    
    def cal_reg_pred_rtn(col, fac_names, beta_fac_names):
        """
        根据系数计算预测后的回归收益
        """
        factor_nums = len(fac_names)
        reg_pred_rtn = col['alpha']
        for i in range(factor_nums):
            reg_pred_rtn += col[fac_names[i]] * col[beta_fac_names[i+1]]
        return reg_pred_rtn
    
    date = 'date' if 'date' not in kwargs else kwargs['date']
    
    factors = factor.copy().dropna()
    
    # 计算回归系数
    beta_fac_names = ["alpha"] + [fac_name+"_beta" for fac_name in fac_names] # 各个因子回归系数的列表
    alpha_beta = factors.groupby(date).apply(lambda x: cal_beta(x, fac_names)).to_frame()
    alpha_beta[beta_fac_names] = alpha_beta[[0]].apply(lambda x: x[0], axis=1, result_type="expand")
    alpha_beta = alpha_beta.drop(0, axis=1)
    alpha_beta = alpha_beta.shift(2).reset_index() # 回归得到的系数事实上只能预测两周后的收益
    
    factors = pd.merge(factors, alpha_beta, on=date, how="left")
    
    tqdm.pandas(desc="Calculating regression return: ")
    factors['reg_pred_rtn'] = factors.progress_apply(lambda x: cal_reg_pred_rtn(x, fac_names, beta_fac_names), axis=1)

    rtn, evaluate_result = backtest_nstock(factors, 'reg_pred_rtn', plot=plot, n=stock_num)
    return rtn, evaluate_result


def mutifactor_icir(factor, fac_names, stock_num=100, plot=True, **kwargs):
    """
    # 一个ICIR加权的多因子选股方法
    ##Args:
        factor (_pd.DataFrame_): 因子数据框

        fac_names (_list_): 准备使用的多因子列表（反向因子前加"-"号即可）

        stock_num (_int_, optional): _description_. 打分选股股票数，默认为100

        plot (_bool_, optional): _description_. 是否画图，默认为是
    
    ##kwargs:

        date: factor数据框中时间对应的列名，未定义时默认为"date"

        pred_rtn: factor数据框中预测收益率对应的列名，未定义时默认为"pred_rtn"

        stock_code: factor数据框中股票代码对应的列名，未定义时默认为"stock_code"

    ##Returns:
    
        rtn (pd.DataFrame): 策略收益数据框

        evaluate_result: 策略评价结果数据框
        
    """
    date = 'date' if 'date' not in kwargs else kwargs['date']
    pred_rtn = 'pred_rtn' if 'pred_rtn' not in kwargs else kwargs['pred_rtn']
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']

    ic_df = pd.DataFrame()
    for fac_name in tqdm(fac_names):
        ic_df["w_"+fac_name] = factor.groupby(date)[[fac_name, pred_rtn]].apply(lambda x: x.corr().iloc[0,1])
    ic_ir_df = ic_df.rolling(8).apply(lambda x: x.mean()/x.std())

    fac_weight = factor[[stock_code, date, pred_rtn]+fac_names] \
        .merge(ic_ir_df, left_on=date, right_index=True, how='left')
    
    fac_weight['combo_factor'] = 0
    for fac_name in tqdm(fac_names):
        fac_weight['combo_factor'] = fac_weight['combo_factor'] + fac_weight[fac_name] * fac_weight['w_'+fac_name]
    fac_weight = fac_weight[fac_names].dropna()

    rtn, evaluate_result = backtest_nstock(factor, 'icir_combo_factor', plot=plot, n=stock_num)
    return rtn, evaluate_result


def mutifactor_ic(factor, fac_names, stock_num=100, plot=True, **kwargs):
    """
    # 一个IC加权的多因子选股方法
    ##Args:
        factor (_pd.DataFrame_): 因子数据框
        fac_names (_list_): 准备使用的多因子列表（反向因子前加"-"号即可）
        stock_num (_int_, optional): _description_. 打分选股股票数，默认为100
        plot (_bool_, optional): _description_. 是否画图，默认为是
    
    ##kwargs:
        date: factor数据框中时间对应的列名，未定义时默认为"date"
        pred_rtn: factor数据框中预测收益率对应的列名，未定义时默认为"pred_rtn"
        stock_code: factor数据框中股票代码对应的列名，未定义时默认为"stock_code"
    ##Returns:
        rtn (pd.DataFrame): 策略收益数据框
        evaluate_result: 策略评价结果数据框
    """
    date = 'date' if 'date' not in kwargs else kwargs['date']
    pred_rtn = 'pred_rtn' if 'pred_rtn' not in kwargs else kwargs['pred_rtn']
    stock_code = 'stock_code' if 'stock_code' not in kwargs else kwargs['stock_code']

    ic_df = pd.DataFrame()
    for fac_name in tqdm(fac_names):
        ic_df["w_"+fac_name] = factor.groupby(date)[[fac_name, pred_rtn]].apply(lambda x: x.corr().iloc[0,1])

    fac_weight = factor[[stock_code, date, pred_rtn]+fac_names] \
        .merge(ic_df, left_on=date, right_index=True, how='left')
    
    fac_weight['combo_factor'] = 0
    for fac_name in tqdm(fac_names):
        fac_weight['combo_factor'] = fac_weight['combo_factor'] + fac_weight[fac_name] * fac_weight['w_'+fac_name]
    fac_weight = fac_weight[fac_names].dropna()

    rtn, evaluate_result = backtest_nstock(factor, 'ic_combo_factor', plot=plot, n=stock_num)
    return rtn, evaluate_result