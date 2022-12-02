import numpy as np
import os
from lib.util.linePlot import LinePlot

def Visuallize(data_list, d, train_seeds):
    data_list = np.array(data_list)
    len_s, len_d, len_e = data_list.shape
    data_list = data_list.transpose(1,0,2).reshape(len_d, len_s*len_e)
    saved_steps = [i for i in range(d.interval, d.train_step+1, d.interval)]
    
    LinePlot(data_list ,saved_steps, f"{d.env_name}_{d.agent_name}", 'out')
    
    import pickle
    with open(os.path.join('out', f'{d.env_name}_{d.agent_name}_{train_seeds}.pickel'), 'wb') as f:
        data = (data_list, saved_steps)
        pickle.dump(data, f)