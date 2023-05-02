import ReadData
import numpy as np
from PIL import Image

band800 = 195
band670 = 134

map_band = {"band430": 21, "band531":68, "band550":77, "band570":87, "band635":117, "band670":134, "band680":138, "band705":150, "band750":171,"band780":185, "band800":195}

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
            #NDVI = (HSI[i][band800][j] - HSI[i][band670][j]) / (HSI[i][band800][j] + HSI[i][band670][j])
            #print(HSI)
            NDVI = (HSI[i,band800,j] - HSI[i,band670,j]) / (HSI[i,band800,j] + HSI[i,band670,j])
            if NDVI< 0.2:
                BG_counter += 1
                Plant_pos.append([i,j])
                # Set the value of HSI's RGB channel as the value of soil ground (178,34,34)
                HSI[i,:,j] = 0*4096/256
                HSI[i,:,j] = 0*4096/256
                HSI[i,:,j] = 0*4096/256
    return HSI, Plant_pos, BG_counter

if __name__ == "__main__":
    HSI_info = ReadData.ReadData()
    lines = HSI_info[0]
    samples = HSI_info[2]
    HSI = HSI_info[3]
    Plant_Pos = getPlantPos(HSI_info)[1]
    #print(len(Plant_Pos))
    ReadData.drawImg(HSI,"RemoveBG")




    