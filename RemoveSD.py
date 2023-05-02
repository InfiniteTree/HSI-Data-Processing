import numpy as np

import ReadData
import RemoveBG

ShadowLeavesValue = 1000 # the set mean Hyspectra value for the leaves in shadow

def calcAmplMean(HSI):
    return HSI.mean(axis=1)

def RemoveSD(HSI_info, set_value):
    samples = HSI_info[1]
    HSI = np.array(HSI_info[3])
    HSI_bandmean = calcAmplMean(HSI)
    #print(HSI_bandmean)
    SD_Counter = 0
    for i in range(HSI_bandmean.shape[0]):
        for j in range(HSI_bandmean.shape[1]):
            pixel_value = HSI_bandmean[i][j]
            if pixel_value > 0 and pixel_value < ShadowLeavesValue: 
                SD_Counter += 1
                HSI[i,:,j] = set_value[0]*4096/256
                HSI[i,:,j] = set_value[1]*4096/256
                HSI[i,:,j] = set_value[2]*4096/256
    #print(HSI)
    #print("\n SDCounter is",SDCounter)

    return HSI,SD_Counter


if __name__ == "__main__":
    HSI_info = ReadData.ReadData()
    HSI = RemoveBG.getPlantPos(HSI_info)[0]
    HSI_info_new = [HSI_info[0],HSI_info[1],HSI_info[2],HSI]
    #set_value = [178, 34, 34]
    set_value = [0, 0, 0]
    HSI_new = RemoveSD(HSI_info_new, set_value)
    
    #print(calcMean(HSI).shape)
    ReadData.drawImg(HSI_new, "RemoveShadow")
