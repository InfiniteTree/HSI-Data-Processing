import ReadData
import numpy as np
from PIL import Image
import setting

band800 = 195
band670 = 134
NDVI_SET_VALUE = 0 # Need to be reset here

#class HSI
'''
def getPlantPos(HSI_info, NDVi_TH):
    lines = HSI_info[0]
    channels = HSI_info[1]
    samples =  HSI_info[2]
    HSI = HSI_info[3]
    NDVI_matrix = []
    Plant_pos = []
    BG_counter = 0
    for i in range(lines):
        for j in range(samples):
            if  (HSI[i][band800][j] + HSI[i][band670][j]) == 0:
                NDVI = 0
            else:
                NDVI = (HSI[i][band800][j] - HSI[i][band670][j]) / (HSI[i][band800][j] + HSI[i][band670][j])
            #print(HSI)
            #NDVI = (HSI[i,band800,j] - HSI[i,band670,j]) / (HSI[i,band800,j] + HSI[i,band670,j])
            if NDVI< NDVi_TH:
                BG_counter += 1
                # Set the value of HSI's RGB channel as the value of soil ground (178,34,34)
                HSI[i,:,j] = 0*4096/256
                HSI[i,:,j] = 0*4096/256
                HSI[i,:,j] = 0*4096/256
            else:
                Plant_pos.append([i,j])
    
    return HSI, np.array(Plant_pos), BG_counter
'''

def getPlantPos(HSI_info, NDVi_TH):
    HSI = HSI_info[3]
    NDVI_matrix = []
    Plant_pos = []
    BG_counter = 0
    HSI = np.array(HSI)

    numerator = HSI[:, band800, :] - HSI[:, band670, :]
    denominator = HSI[:, band800, :] + HSI[:, band670, :]
    denominator[denominator == 0] = 1  # Avoid the denominator is zero, set it as 1 
    NDVI = numerator / denominator
    NDVI[denominator == 0] = 0  # the denominator is zero, set it as 1
    
    # Build Mask 
    mask = NDVI < NDVi_TH

    # Generate the HSI
    BG_counter += np.count_nonzero(mask)
    HSI[mask] = np.array([0, 0, 0]) * (4096 / 256)
    Plant_pos = np.argwhere(~mask).tolist()

    return HSI, np.array(Plant_pos), BG_counter

# Get the Level1 HSI by removing the background 
def getLevel1(HSI_info, NDVi_TH):
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    wavelengths = HSI_info[4]
    PixelSum = lines * samples

    ### Level 1. Remove the background
    Level_1 = getPlantPos(HSI_info, NDVi_TH)
    HSI_1 = Level_1[0]
    BG_Counter = Level_1[2]
    HSI_info_L1 = [lines, channels, samples, HSI_1, wavelengths]
    proportion_1 = float((PixelSum - BG_Counter)/PixelSum)

    return HSI_info_L1, BG_Counter, proportion_1

if __name__ == "__main__":
    HSI_info = ReadData.Read()
    Level1 = getLevel1(HSI_info, NDVI_SET_VALUE)
    #Plant_Pos = getPlantPos(HSI_info)[1]
    #print(len(Plant_Pos))
    ReadData.drawImg(HSI_info,"pre_processing/Level1_img")


    