import numpy as np
import matplotlib.pyplot as plt

import ReadData
import RemoveBG


if __name__ == "__main__":
    HSI_info = ReadData.ReadData()
    channels = HSI_info[2]
    plantPos = RemoveBG.getPlant_pos(HSI_info) # a list of plant position
    HSI = np.array(HSI_info[3])
    mean_values = HSI.mean(axis=(0,2))
    print(mean_values.shape)
    print(mean_values)
    print(type(list(mean_values)))

