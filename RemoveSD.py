import numpy as np

import ReadData
import RemoveBG

ShadowLeavesValue = 300 # the set mean Hyspectra value for the leaves in shadow

def calcAmplMean(HSI, proportion):
    return HSI.mean(axis=1) / proportion 


def RemoveSD(HSI_info, set_value, cur_proportion):
    samples = HSI_info[1]
    HSI = np.array(HSI_info[3])
    HSI_bandmean = calcAmplMean(HSI, cur_proportion)
    #print(HSI_bandmean)
    SD_Counter = 0
    HL_position = [] # High lighted plant position
    for i in range(HSI_bandmean.shape[0]):
        for j in range(HSI_bandmean.shape[1]):
            pixel_value = HSI_bandmean[i][j]
            if pixel_value > 0 and pixel_value < ShadowLeavesValue: 
                SD_Counter += 1
                '''
                HSI[i,105,j] = set_value[0]*4096/256
                HSI[i,59,j] = set_value[1]*4096/256
                HSI[i,34,j] = set_value[2]*4096/256
                '''
                HSI[i,:,j] = set_value[0]*4096/256
                HSI[i,:,j] = set_value[1]*4096/256
                HSI[i,:,j] = set_value[2]*4096/256
            else:
                HL_position.append([i,j])  
    #print(HSI)
    #print("SDCounter is",SD_Counter)
    return HSI,SD_Counter, np.array(HL_position)


if __name__ == "__main__":
    HSI_info = ReadData.Read()
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    ret_RemoveBG = RemoveBG.getPlantPos(HSI_info)
    HSI_1 = ret_RemoveBG[0]
    HSI_info_1 = [lines,channels,samples,HSI_1]

    #set_value = [178, 34, 34]
    PixelSum = lines * samples
    BG_Counter = ret_RemoveBG[2]
    cur_proportion = float((PixelSum - BG_Counter)/PixelSum)
    set_value = [0, 0, 0]
    HSI_2 = RemoveSD(HSI_info_1, set_value, cur_proportion)[0]
    HSI_info_2 = [lines,channels,samples,HSI_2]
    #print(calcMean(HSI).shape)
    ReadData.drawImg(HSI_info_2, "Level2_img")
