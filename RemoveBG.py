import ReadData
import numpy as np
from PIL import Image
import setting

band800 = 195
band670 = 134
NDVI_SET_VALUE = 0 # Need to be reset here

#class HSI

def getPlantPos(HSI_info):

    lines = HSI_info[0]
    channels = HSI_info[1]
    samples =  HSI_info[2]
    HSI = HSI_info[3]
    NDVI_matrix = []
    Plant_pos = []
    BG_counter = 0
    for i in range(lines):
        for j in range(samples):
            NDVI = (HSI[i][band800][j] - HSI[i][band670][j]) / (HSI[i][band800][j] + HSI[i][band670][j])
            #print(HSI)
            #NDVI = (HSI[i,band800,j] - HSI[i,band670,j]) / (HSI[i,band800,j] + HSI[i,band670,j])
            if NDVI< NDVI_SET_VALUE:
                BG_counter += 1
                # Set the value of HSI's RGB channel as the value of soil ground (178,34,34)
                HSI[i,:,j] = 0*4096/256
                HSI[i,:,j] = 0*4096/256
                HSI[i,:,j] = 0*4096/256
            else:
                Plant_pos.append([i,j])
    
    return HSI, np.array(Plant_pos), BG_counter

if __name__ == "__main__":
    HSI_info = ReadData.ReadData("M:/m-CTP_DATA/2023.1.9/Vegetables/TASK2023-01-06-10-52/Hyperspectral/wave.hdr",'M:/m-CTP_DATA/2023.1.9/Vegetables/TASK2023-01-06-10-52/Hyperspectral/2023-01-06-10-56-46.spe')
    Plant_Pos = getPlantPos(HSI_info)[1]
    #print(len(Plant_Pos))
    ReadData.drawImg(HSI_info,"RemoveBG_NDVI_LessThanZero")




    