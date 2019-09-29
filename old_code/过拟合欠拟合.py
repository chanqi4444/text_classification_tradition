#coding:utf-8
"""
过拟合和欠拟合示例
"""

import matplotlib.pyplot as plt
import numpy as np

# 画出拟合出来的多项式所表达的曲线以及原始的点
def plot_polynominal_fit(x, y, order):
    p = np.poly1d(np.polyfit(x, y, order))
    t = np.linspace(0, 1, 200)
    plt.plot(x, y, 'ro', t, p(t), '-', t, np.sqrt(t), 'r--')
    return p

# 生成20个点的训练样本
n_dots = 20
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2*np.random.rand(n_dots) - 0.1

plt.figure(figsize=(18, 4))
titles = ['Under Fitting', 'Fitting', 'Over Fitting']
models = [None, None, None]
for index, order in enumerate([1, 3, 10]):
    plt.subplot(1, 3, index + 1)
    models[index] = plot_polynominal_fit(x, y, order)
    plt.title(titles[index], fontsize=20)
plt.show()