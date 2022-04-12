import matplotlib.pyplot as plt
import numpy as np
import json


def eval_box():
    pass



if __name__ == '__main__':
    data = np.array([])
    with open("../exp/test/test.json") as f:
        json_data = json.load(f)

    for item in json_data.items():
        tmp = []
        for c in item[1]:
            if type(c) is dict:
                #print(c)
                tmp.append(c['fitness'])
            print(tmp)
        data = np.append(data, tmp)
        #print(item[1])
    print(data)
    pass
