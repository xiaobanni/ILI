import os
import numpy as np
import pandas as pd


def store_rule_data(buffer, env, path, idx=0, len=1000):
    raw_data = buffer.get_lastest_k(len)
    data = np.array([np.concatenate(
        (item[0], np.expand_dims(item[1], axis=0))
    ) for item in raw_data])
    colunms = [f"state {i}" for i in range(data.shape[1]-1)]+["action"]
    f = pd.DataFrame(data, columns=colunms).astype({"action": int})
    f.to_csv(path+os.sep+f"{env}_{idx}.csv", index=False)
