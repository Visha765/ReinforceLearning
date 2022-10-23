import numpy as np
import pandas as pd
import gym
import time

# 乱数のシードを設定
seed = 100
np.random.seed(seed)

# 離散化に使用する分割数を定義
class N():
    theta = 25
    theta_dot = 50
    tau = 25
    actions = tau

# 角度を [0-360) にする関数を定義
f = lambda x: x - np.floor(x/360)*360

######################################
# 状態遷移確率の表の実際の値を求める #
######################################
##################
# ビンを作成する #
##################
# theta, tau, theta_dot の範囲を定義
bin_theta = np.linspace(0, 360, N.theta + 1)
bin_theta_dot = np.linspace(-8.01, 8.01, N.theta_dot + 1)
bin_tau = np.linspace(-2.01, 2.01, N.tau + 1)
# 角度、速度、トルク を離散化(pd.IntervalIndex.from_breaks を使用)
theta_index = pd.IntervalIndex.from_breaks(bin_theta)
theta_dot_index = pd.IntervalIndex.from_breaks(bin_theta_dot)
tau_index = pd.IntervalIndex.from_breaks(bin_tau)
# インデックスの作成
index_0 = pd.MultiIndex.from_product(
    [theta_index, theta_dot_index, tau_index],
    names=["theta_cat", "theta_dot_cat", "tau_cat"])
index_1 = pd.MultiIndex.from_product([theta_index, theta_dot_index],
                                    names=["theta_2_cat", "theta_dot_2_cat"])

print("Psasを作成")

# 状態遷移確率用
Psas = pd.DataFrame(0, index=index_0, columns=index_1)

print("Rsasを作成")

# 報酬は theta が 0°に近いほどよいかつ、運動エネルギーが小さいほどよいようにした。
data = []
for theta_2_cat, theta_dot_2_cat in Psas.columns:
    data.append(np.cos(np.deg2rad(theta_2_cat.mid))**3/(theta_dot_2_cat.mid**2 + 0.00001))

arr = np.zeros((len(Psas),1)) +  np.array(data)
Rsas = pd.DataFrame(arr, index=index_0, columns=index_1)

print(Rsas)

# ########################
# # シミュレータを動かす #
# ########################
# ###############
# # gymの初期化 #
# ###############
# # 環境の作成
# env = gym.make('Pendulum-v0') 
# # 環境の初期化(明示的に行う必要がある)
# env.reset()  
# for theta_cat, theta_dot_cat, tau_cat in Psas.index:
#     # 振り子の theta, speed を設定
#     # ビンの中間の値を取得するため Interval オブジェクトの mid プロパティを使用
#     env.env.state = np.deg2rad(theta_cat.mid), theta_dot_cat.mid
#     obser, r, done, info = env.step([tau_cat.mid])
#     theta_2, theta_dot_2 = env.env.state
#     # デフォルトで rad なので degree に変換
#     theta_2 = np.rad2deg(theta_2)
#     theta_2 = f(theta_2)
#     # カウントを +1 する
#     Psas.loc[(theta_cat, theta_dot_cat, tau_cat), (theta_2, theta_dot_2)] += 1

# # Psas 確率に変換
# Psas = Psas / Psas.sum(axis=1).values.reshape(-1,1)

# # 方策の初期化
# PIas =  pd.DataFrame(1/N.actions, index=index_0, columns=["PI_tau_Prob"])

# cnt = 0
# gamma = 0.99

# while True:
#     print(cnt)
#     # 状態遷移確率と方策の確率を合わせたものを求める
#     P_PI_ss = Psas * PIas.values.reshape(-1, 1)
#     P_PI_ss = P_PI_ss.sum(level=[0, 1])

#     # R_PI_s を求める
#     R_PI_s = Psas * PIas.values.reshape(-1, 1) * Rsas
#     R_PI_s = R_PI_s.sum(axis=1).sum(level=[0, 1])

#     # pre_V_s = V_s.copy()
#     pre_PIas = PIas.copy()

#     # 状態価値 V を求める(ベルマン方程式を解析的に解く)
#     size = len(R_PI_s)
#     V_values = np.linalg.inv(np.eye(size) - gamma*P_PI_ss.values).dot(R_PI_s)
#     V_s = pd.Series(V_values, index=R_PI_s.index)

#     # 行動価値 Q を求める
#     Q_sa = Psas*(Rsas + gamma*V_s.values.reshape(1,-1))
#     Q_sa = Q_sa.sum(axis=1)

#     # 方策を更新
#     # 状態のグループから行動価値が最大となるインデックスを求める
#     # https://stackoverflow.com/questions/27842613/pandas-groupby-sort-within-groups
#     largest_Qsa =  Q_sa.groupby(level=[0,1], group_keys=False).nlargest(1)
#     PIas.loc[:] = 0
#     for (theta_cat, theta_dot_cat), df in largest_Qsa.groupby(level=[0, 1]):
#         PIas.loc[df.index] = 1

#     # 状態価値の最小値を表示
#     print(V_s.min())
#     # 方策が更新されなくなったら終了
#     if (pre_PIas.values==PIas.values).all():
#         break
#     # 上限回数を超えても終了する
#     cnt += 1
#     if cnt > 10:
#         break
#     first = False

# ##########################################
# # 求めた方策に従い、シミュレータを動かす #
# ##########################################

# # 環境の作成
# env = gym.make('Pendulum-v0') 
# env.seed(1)
# for __ in range(10):
#     # 環境の初期化(明示的に行う必要がある)
#     env.reset()
#     for _ in range(400):
#         # 現在の状態を取得
#         theta, theta_dot = env.env.state
#         # デフォルトで rad なので degree に変換
#         theta = np.rad2deg(theta)
#         theta = f(theta)
#         tau = PIas.loc[theta].loc[theta_dot].idxmax().values[0].mid
#         env.step([tau])
#         env.render()
#         # time.sleep(0.03)

# env.close()