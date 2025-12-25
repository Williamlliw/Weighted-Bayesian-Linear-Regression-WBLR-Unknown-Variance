'''
This module implements Weighted Bayesian Linear Regression (WBLR)
with analytical solutions for training and prediction.
Version: 1.0
Author: Your Name
Date: 2025-10-13
'''


import scipy
import numpy as np
import scipy
import scipy.stats as stats
import warnings


def wblr_fit(X, y, prior, l = None):
    """
    训练 WBLR, 返回解析后验参数
    X: (n,d) 数据矩阵, 自变量
    y: (n,)  观测目标
    l: (n,)  每条样本权重 ∈[0,1]
    prior: dict 必须包含 wo, Vo, ao, bo
    return: dict 包含 WN, VN, aN, bN
    """
    n, d = X.shape
    if l is None:
        L = scipy.sparse.eye(n, format='csr')  # L = I
    else:
        L = np.diag(l)
    wo, Vo, ao, bo = prior['wo'], prior['Vo'], prior['ao'], prior['bo']
    
    # 按公式 (10)-(13)
    Vn_inv = np.linalg.inv(Vo) + X.T @ L @ X
    VN = np.linalg.inv(Vn_inv)
    WN = VN @ (np.linalg.inv(Vo) @ wo + X.T @ L @ y)
    aN = ao + 0.5 * L.diagonal().sum() 
    bN = bo + 0.5 * (wo.T @ np.linalg.inv(Vo) @ wo +
                     y.T @ L @ y -
                     WN.T @ Vn_inv @ WN)
    return [WN, VN, aN, bN]


def wblr_fit_multiout(X, y, prior, l = None):
    """
    针对多输出的情况，训练 WBLR, 返回解析后验参数
    X: (N,nx) 数据矩阵, 自变量
    y: (N,ny)  观测目标
    l: (N,)  每条样本权重 ∈[0,1]
    prior: dict 必须包含 wo, Vo, ao, bo
    return: dict 包含 WN, VN, aN, bN
    """
    n, nx = X.shape
    ny = y.shape[1]
    if l is None:
        L = scipy.sparse.eye(n, format='csr')  # L = I
    else:
        L = scipy.sparse.diags(l, format='csr')
    wo, Vo, ao, bo = prior['wo'], prior['Vo'], prior['ao'], prior['bo']
    
    # 按公式 (10)-(13)
    Vo_inv = np.linalg.inv(Vo)
    Vn_inv = Vo_inv + X.T @ L @ X
    VN = np.linalg.inv(Vn_inv)
    WN = VN @ (Vo_inv @ wo + X.T @ L @ y)
    aN = ao + 0.5 * L.diagonal().sum()
    
    bN = bo + 0.5 * np.diag(wo.T @ Vo_inv @ wo + y.T @ L @ y - WN.T @ Vn_inv @ WN)

    if np.any(bN < 0):
        warnings.warn("bN < 0, something wrong!")

    return [WN, VN, aN, bN]



def wblr_pred(X_test, post, option = 0):
    """
    给定测试集 X_test(m x n), 返回解析后验预测分布的均值与方差
    m: 测试集样本数
    n: 自变量维度
    预测分布是 Student-t: y* ~ T(mean, scale**2, df)
    返回 mean, scale**2, df
    """
    WN, VN, aN, bN = post
    eigval = np.linalg.eigvals(VN)
    y_mean = X_test @ WN
    m = X_test.shape[0]
    scale2 = np.kron(bN.reshape(1, -1) / aN, np.eye(m) + X_test @ VN @ X_test.T)
    df = 2 * aN

    if option == 0:
        # calculate credible_interval
        alpha = 0.01 # 99% 置信区间
        t_val = stats.t.ppf(1 - alpha/2, df=df) # 查找临界值
        bound = t_val * np.sqrt(np.diag(scale2))

        return y_mean.reshape((-1,)), bound.reshape((-1,))
    elif option == 1:
        # return variance
        variance = np.diag(df / (df - 2) * scale2.reshape((-1,))) if df > 2 else np.inf
        return y_mean.reshape((-1,)), variance
    else:
        # for monte carlo sampling
        return y_mean, scale2, df