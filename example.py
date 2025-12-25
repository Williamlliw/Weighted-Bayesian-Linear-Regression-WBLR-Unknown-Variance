"""
demo_wblr.py
Weighted Bayesian Linear Regression 示例（方差未知）
依赖：numpy、scipy、matplotlib、bayes_module.py（同目录）
"""
import numpy as np
import matplotlib.pyplot as plt
from bayes_module import wblr_fit, wblr_pred

# ------------------------------------------------------------------
# 1. 造一份带异方差噪声的 1-D 数据
# ------------------------------------------------------------------
np.random.seed(42)
n_train = 20
X_train = np.linspace(-3, 3, n_train).reshape(-1, 1)          # (60,1)
w_true = np.array([2.0, -1.5])                                # 真实参数 [bias, slope]
y_true = X_train @ w_true[1:] + w_true[0]                     # 无噪声输出
# 异方差噪声：|x| 越大噪声越大
sigma_noise = 0.2 * (1 + np.abs(X_train[:, 0]))
y_train = y_true + sigma_noise * np.random.randn(n_train)

# 样本权重示例：中间段更可信
weights = np.ones(n_train)
weights[np.abs(X_train[:, 0]) > 2] = 0.3

# ------------------------------------------------------------------
# 2. 构造先验
# ------------------------------------------------------------------
d = X_train.shape[1] + 1               # 加上 bias 项
prior = {
    "wo": np.zeros(d),                  # 回归系数先验均值
    "Vo": np.eye(d) * 10,               # 先验协方差（越大约=越弱）
    "ao": 2.0,                          # Gamma 先验 shape
    "bo": 1.0                           # Gamma 先验 scale
}

# 训练：增广 X 带一列 1 做 bias
X_aug = np.c_[np.ones(n_train), X_train]
post = wblr_fit(X_aug, y_train, prior, l=weights)  # 如果处理多输出，把函数换成 wblr_fit_multiout即可
WN, VN, aN, bN = post

print("后验均值 wN:", WN)
print("后验噪声精度 Gamma({:.2f}, {:.2f})".format(aN, bN))

# ------------------------------------------------------------------
# 3. 预测网格 & 可视化
# ------------------------------------------------------------------
x_plot = np.linspace(-7, 7, 200).reshape(-1, 1)
X_plot = np.c_[np.ones(x_plot.shape[0]), x_plot]
y_true = x_plot @ w_true[1:] + w_true[0]

# 返回 99% 可信区间
y_mean, bound = wblr_pred(X_plot, post, option=0)
y_lower, y_upper = y_mean - bound, y_mean + bound

plt.figure(figsize=(8, 5))
plt.scatter(X_train[:, 0], y_train, c=weights, cmap='coolwarm',
            s=50, edgecolors='k', label='train (size∝weight)')
plt.plot(x_plot, y_true, 'k--', lw=2, label='true mean')
plt.plot(x_plot, y_mean, 'r-', lw=2, label='posterior mean')
plt.fill_between(x_plot[:, 0], y_lower, y_upper,
                 color='red', alpha=0.2, label=r'99% credible')
plt.colorbar(label='sample weight')
plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.title('Weighted Bayesian Linear Regression (unknown variance prior)')
plt.tight_layout()
plt.grid()
plt.show()