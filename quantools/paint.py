from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from .evaluate import maximum_drawdown

def return_painter(return_df: pd.DataFrame, **kwargs):
    """绘制收益率曲线图

    Args:
        return_df (pd.DataFrame): 收益率数据框，索引为时间，值为每日收益率

    kwargs:
        ret: 收益率列的名称，默认为'ret'
    """
    ret = 'ret' if 'ret' not in kwargs else kwargs['ret']
    # 计算最大回撤，获取所有回撤数据
    cum_returns = (return_df + 1).cumprod(axis=0)
    peak = cum_returns.expanding().max()
    dd = ((peak - cum_returns)/peak)
    mdd = dd.max()
    end = dd[dd==mdd].dropna().index[0]
    start = peak[peak==peak.loc[end]].dropna().index[0]

    # 计算第二大回撤
    mdd2_1 = maximum_drawdown(return_df.loc[return_df.index[0]: start])
    mdd2_2 = maximum_drawdown(return_df.loc[end: return_df.index[-1]])
    mdd2_max = mdd2_1 if mdd2_1['max_drawdown'] > mdd2_2['max_drawdown'] else mdd2_2
    start2, end2 = mdd2_max['max_drawdown_start'], mdd2_max['max_drawdown_end']

    # 绘图
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax2 = ax.twinx()

    cum_returns.plot(ax=ax, legend=None) # 绘制收益率
    dd.plot(ax=ax2, color="grey", alpha=0.5, legend=None) # 绘制回撤信息
    ax2.fill_between(cum_returns.index, [0]*cum_returns.shape[0], dd[ret], color="grey", alpha=0.5)
    ax2.fill_between(x=cum_returns.index, y1=[1]*cum_returns.shape[0], y2=[0]*cum_returns.shape[0], 
                    where=(cum_returns.index>start) & (cum_returns.index<end), color='orange', alpha=0.5) # 绘制回撤区间信息
    ax2.fill_between(x=cum_returns.index, y2=[0]*cum_returns.shape[0], y1=[1]*cum_returns.shape[0],
                    where=(cum_returns.index>start2) & (cum_returns.index<end2), color='orange', alpha=0.5)
    ax2.set_yticks(np.linspace(0,1,11))
    ax2.invert_yaxis()
    