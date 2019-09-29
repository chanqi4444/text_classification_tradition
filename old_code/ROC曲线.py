#coding:utf-8
"""
绘制ROC曲线，计算AUC值示例
"""

import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

# 实际类别只能取{0,1}或{1,-1}
y = np.array([1,0,1,0,0,0,1,0,1,1])
# 对应预测为正（即1）的得分。注意：得分相同的实例只保留一个
scores = np.array([0.25,0.43,0.53,0.76,0.85,0.86,0.87,0.89,0.93,0.95])
# pos_label 假定为正类的类别标记，这里是1
fpr, tpr, thresholds = roc_curve(y_true = y, y_score = scores, pos_label=1)
print("tpr=", tpr)
print("fpr=", fpr)
print("thresholds=", thresholds)
# 计算auc值
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()