import ReadData
import numpy as np

band430 = 21
band531 = 68
band570 = 87
band670 = 134
band680 = 138
band705 = 150
band750 = 171
band780 = 185 
band800 = 195

def getPlant_pos(HSI_info):
    samples =  HSI_info[0]
    lines = HSI_info[1]
    channels = HSI_info[2]
    HSI = HSI_info[3]
    NDVI_matrix = []
    Plant_pos = []

    for i in range(lines):
        for j in range(samples):
            NDVI = (HSI[i][band800][j] - HSI[i][band670][j]) / (HSI[i][band800][j] + HSI[i][band670][j])
            if NDVI< 0.4:
                Plant_pos.append([i,j])
    return Plant_pos

if __name__ == "__main__":
    HSI_info = ReadData.ReadData()
    Plant_Pos = getPlant_pos(HSI_info)
    
    print(len(Plant_Pos))


    