from lzma import CHECK_SHA256
import os
import numpy as np
import matplotlib.pyplot as plt
from lib.util.table import *

# rw = [-1418.5732911288014, -969.9296553406258, -1652.748147293833, -1300.019242296337, -503.3613256613123, -1663.7292422597498, -1213.576445859204, -1159.4444132292595, -1167.3485911329735, -863.703162729154]

# q = create_Qtable(4, 4)

plt.rcParams['font.family'] = 'Hiragino Sans'
cs = list(np.random.normal(50, 10, size=(10,10)))
# print(np.shape(cs))

# c_array = np.percentile(c_list, q=[25,50, 75])
f = [np.percentile(data, q=[25,50, 75]) for data in cs]
f = np.array(f).T

x = np.linspace(0,10,10)

print(np.shape(f[0]))

plt.plot(x,f[0], marker='o')
plt.plot(x,f[1], marker='o')
plt.plot(x,f[2], marker='o')
plt.fill_between(x, f[0], f[2], alpha=0.4)
# plt.xticks(x, "step")
# plt.xlim(0, 13)
# plt.ylim(0, 35) #y軸の最大値を変更30→35

plt.title('The average temperature in 2016', fontsize = 20)
plt.xlabel('step', fontsize = 16)
plt.ylabel('累積報酬', fontsize = 16)
plt.tick_params(labelsize=14)
plt.grid(True)
# plt.show()

plt.show()