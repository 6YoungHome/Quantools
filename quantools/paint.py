from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from .evaluate import *

def cumulate_return_painter(return_df: pd.DataFrame, ax: plt.Axes):
    """绘制累计收益率曲线图

    Args:
        return_df (pd.DataFrame): 收益率数据框，索引为时间，值为每日收益率
        ax (plt.Axes): 绘图所用的ax
    """
    # 计算最大回撤，获取所有回撤数据
    cum_returns = (return_df + 1).cumprod(axis=0)
    cum_returns.plot(ax=ax, legend=None) # 绘制收益率
    ax.set_title(f"Cumulate return")


def max_drawdown_painter(return_df: pd.DataFrame, ax: plt.Axes):
    """绘制最大回撤情况图

    Args:
        return_df (pd.DataFrame): 收益率数据框，索引为时间，值为每日收益率
        ax (plt.Axes): 绘图所用的ax
    """
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
    dd.plot(ax=ax, color="grey", alpha=0.5, legend=None) # 绘制回撤信息
    ax.fill_between(cum_returns.index, [0]*cum_returns.shape[0], dd.iloc[:,0], color="grey", alpha=0.5)
    ax.fill_between(x=cum_returns.index, y1=[1]*cum_returns.shape[0], y2=[0]*cum_returns.shape[0], 
                    where=(cum_returns.index>start) & (cum_returns.index<end), color='orange', alpha=0.5) # 绘制回撤区间信息
    ax.fill_between(x=cum_returns.index, y2=[0]*cum_returns.shape[0], y1=[1]*cum_returns.shape[0],
                    where=(cum_returns.index>start2) & (cum_returns.index<end2), color='orange', alpha=0.5)
    ax.set_yticks(np.linspace(0,1,11))
    ax.invert_yaxis()


def group_cummulate_returns_painter(group_ret: pd.DataFrame, ax: plt.Axes):
    """分组累计收益

    Args:
        group_ret (pd.DataFrame): 分组收益率数据框，索引为时间，列为每组每日收益率
        ax (plt.Axes): 绘图所用的ax
    """
    cum_returns = (group_ret + 1).cumprod(axis=0)
    cum_returns.plot(ax=ax)
    ax.set_title(f"Cumulate return of {group_ret.shape[1]} groups")


def group_annual_returns_painter(group_ret: pd.DataFrame, ax: plt.Axes):
    """分组年化收益

    Args:
        group_ret (pd.DataFrame): 分组收益率数据框，索引为时间，列为每组每日收益率
        ax (plt.Axes): 绘图所用的ax
    """
    annual_rtns = [annual_info(group_ret.iloc[:, i])['annual_return'] for i in range(group_ret.shape[1])]
    ax.bar(group_ret.columns, annual_rtns)
    ax.axhline(np.mean(annual_rtns), color='orange')


def diverse_cummulate_painter(return_df: pd.DataFrame, ax: plt.Axes):
    """同时绘制累计收益与最大回撤情况

    Args:
        return_df (pd.DataFrame): _description_
        ax (plt.Axes): 绘图所用的ax
    """
    cumulate_return_painter(return_df, ax)
    ax_twin = ax.twinx()
    max_drawdown_painter(return_df, ax_twin)