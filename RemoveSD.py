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
    return HSI, SD_Counter, np.array(HL_position)

# Get the Level2 HSI by removing the shadows
def getLevel2(HSI_info_L1, BG_Counter, proportion_1):
    wavelengths = HSI_info_L1[4]
    lines = HSI_info_L1[0] 
    channels = HSI_info_L1[1]
    samples = HSI_info_L1[2]
    set_value = [0, 0, 0]
    PixelSum = lines * samples
    
    HSI_2, SD_Counter, HL_position = RemoveSD(HSI_info_L1,set_value, proportion_1)
    HSI_info_L2 = [lines, channels, samples, HSI_2,  wavelengths]
    proportion_2 = float((PixelSum - SD_Counter - BG_Counter)/PixelSum)

    return HSI_info_L2, SD_Counter, proportion_2

if __name__ == "__main__":
    HSI_info = ReadData.Read()
    # Level 1
    HSI_info_L1, BG_Counter, PixelSum, proportion_1 = RemoveBG.getLevel1(HSI_info)
    # Level 2
    HSI_info_L2, SD_Counter = getLevel2(HSI_info_L1, proportion_1)
    #print(calcMean(HSI).shape)
    ReadData.drawImg(HSI_info_L2, "Level2_img")
