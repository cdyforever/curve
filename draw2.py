import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
sns.set()  #切换到sns的默认运行配置
import warnings
warnings.filterwarnings('ignore')

sns.set(style="darkgrid", font_scale=1.2) 

num_list =  []
with open("curv.txt", "r") as f:
    allitem = f.readlines()
    for item in allitem:
        item = item.strip()
        num = eval(item)
        num_list.append(num)
npy = np.array(num_list)

print(npy)

saveids = np.where(npy < 1)[0]
npy = npy[saveids]


plt.figure(dpi=120)
sns.set(style='dark')
sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
g=sns.distplot(npy,
               hist=True,
               bins=200,#修改箱子个数
               kde=False,
               color="#098154")
plt.show()