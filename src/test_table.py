import numpy as np
import pandas as pd

############################
# パラメータの表を作成する #
############################
from itertools import product

class N():
    theta = 2
    theta_dot =2
    tau = 2
    actions = tau

##################################
# 表に使用するインデックスの定義 #
##################################
# theta, tau, theta_dot の範囲を定義
bin_theta = np.linspace(0, 360, N.theta + 1)
bin_theta_dot = np.linspace(-8, 8, N.theta_dot + 1)
bin_tau = np.linspace(-2, 2, N.tau + 1)
# 角度、速度、トルク を離散化(pd.IntervalIndex.from_breaks を使用)
theta_index = pd.IntervalIndex.from_breaks(bin_theta)
theta_dot_index = pd.IntervalIndex.from_breaks(bin_theta_dot)
tau_index = pd.IntervalIndex.from_breaks(bin_tau)
print(theta_index)
# インデックスの作成
index_0 = pd.MultiIndex.from_product(
    [theta_index, theta_dot_index, tau_index],
    names=["theta_cat", "theta_dot_cat", "tau_cat"])
index_1 = pd.MultiIndex.from_product([theta_index, theta_dot_index], 
                                    names=["theta_2_cat", "theta_dot_2_cat"])
print(index_0)

############
# P の作成 #
############
# 「角度、速度、トルク」 - 「次の角度、次の速度」の表を作る
P = pd.DataFrame(1, index=index_0, columns=index_1)
P.loc[(35, 4, -1), (60, 4)] += 100
print("うぁぁぁ",P.loc[(35, 4, -1), (60, 4)])

##################
# 方策の表を作成 #
##################
policy = pd.Series(1 / N.actions, index=index_0)

########
# 表示 #
########
print("状態遷移確率の表")
print(P)
print("方策の表")
print(policy)