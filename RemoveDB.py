import numpy as np

import ReadData
import RemoveBG
import math

ShadowLeavesValue = 300 # the set mean Hyspectra value for the leaves in shadow
BrightLeavesValue = 800

SD_Counter, BT_Counter, DB_Counter = 0, 0, 0

def calcAmplMean(HSI, proportion):
    return HSI.mean(axis=1) / proportion 


def RemoveDB(HSI_info, set_value, cur_proportion, Remove_type):
    global SD_Counter, BT_Counter, DB_Counter
    samples = HSI_info[1]
    HSI = np.array(HSI_info[3])
    HSI_bandmean = calcAmplMean(HSI, cur_proportion)
    #print(HSI_bandmean.mean())
    #print(HSI_bandmean)
    HL_position = [] # High lighted plant position
    for i in range(HSI_bandmean.shape[0]):
        for j in range(HSI_bandmean.shape[1]):
            pixel_value = HSI_bandmean[i][j]
            if Remove_type == "SD":
                if (pixel_value > 0 and pixel_value < ShadowLeavesValue): 
                    SD_Counter += 1
                    HSI[i,:,j] = set_value[0]*4096/256
                    HSI[i,:,j] = set_value[1]*4096/256
                    HSI[i,:,j] = set_value[2]*4096/256
            elif Remove_type == "BT":
                if pixel_value > BrightLeavesValue:
                    BT_Counter += 1
                    HSI[i,:,j] = set_value[0]*4096/256
                    HSI[i,:,j] = set_value[1]*4096/256
                    HSI[i,:,j] = set_value[2]*4096/256
            else:
                    HL_position.append([i,j])  
    DB_Counter = SD_Counter + BT_Counter
    #print(HSI)
    if Remove_type == "BT":
        print("BTCounter is",BT_Counter)
    if Remove_type == "SD":
        print("SDCounter is",SD_Counter)
    
    return HSI, DB_Counter, np.array(HL_position)


# Get the Level2 HSI by removing the shadows
def getLevel2(HSI_info_L1, BG_Counter, proportion_1):
    wavelengths = HSI_info_L1[4]
    lines = HSI_info_L1[0] 
    channels = HSI_info_L1[1]
    samples = HSI_info_L1[2]
    set_value = [0, 0, 0]
    PixelSum = lines * samples
    SD_Counter, BT_Counter, DB_Counter = 0, 0, 0
    # Remove the blades in shadow
    HSI_2, DB_Counter, HL_position = RemoveDB(HSI_info_L1,set_value, proportion_1, "SD")
    # Remove the blades that are too bright
    HSI_2, DB_Counter, HL_position = RemoveDB(HSI_info_L1,set_value, proportion_1, "BT")
    HSI_info_L2 = [lines, channels, samples, HSI_2,  wavelengths]
    proportion_2 = float((PixelSum - DB_Counter - BG_Counter)/PixelSum)

    return HSI_info_L2, DB_Counter, proportion_2

if __name__ == "__main__":
    HSI_info = ReadData.Read()
    # Level 1
    HSI_info_L1, BG_Counter, proportion_1 = RemoveBG.getLevel1(HSI_info)
    # Level 2
    HSI_info_L2, DB_Counter, proportion_2 = getLevel2(HSI_info_L1, BG_Counter, proportion_1)
    #print(calcMean(HSI).shape)
    ReadData.drawImg(HSI_info_L2, "wheat/Pre_Processing/Level2_img_new")

