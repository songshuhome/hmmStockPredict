# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from hmmlearn.hmm import GaussianHMM

# 导入数据
data=ts.get_hist_data('600640',start='2018-01-01',end='2019-12-31')
close_v=data['close'].values
volume=data['volume'].values
dates=np.array([i for i in range(data.shape[0])])
fig1=plt.figure()
plt.plot(close_v,color='blue')
plt.show()
fig1.savefig('stocks.svg')

# 处理数据
diff=np.diff(close_v)
dates=dates[1:]
close_v=close_v[1:]
volume=volume[1:]
x=np.column_stack([diff,volume])
diff=diff.reshape(-1,1) # 二维矩阵

model=GaussianHMM(n_components=2,n_iter=1000) # n_components 状态序列的种类，n_iter 迭代次数
model.fit(diff)
hidden_states=model.predict(diff)
fig2=plt.figure()
colors=['yellow','blue']
for j in range(len(close_v)-1):
    for i in range(model.n_components):
        if hidden_states[j] == i:
            plt.plot([dates[j],dates[j+1]],[close_v[j],close_v[j+1]],color = colors[i])
plt.show()
fig2.savefig('hidden_state.svg')
# 分为震荡和剧烈涨跌
