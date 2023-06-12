import ReadData
import numpy as np
from PIL import Image

class level1:
    NDVI_TH = 0.8
    HSI_info = [] # shape = (n, k, m)
    NDVI = [] # shape = (n, 1, m)
    band800 = 195
    band670 = 134
    
    def __init__(self, hsiInfo, ndviTh):
        self.NDVI_TH = ndviTh
        self.HSI_info = hsiInfo

    def getPlantPos(self):
        HSI = self.HSI_info[3]
        NDVI_matrix = []
        Plant_pos = []
        BG_counter = 0
        HSI = np.array(HSI)

        numerator = HSI[:, self.band800, :] - HSI[:, self.band670, :]
        denominator = HSI[:, self.band800, :] + HSI[:, self.band670, :]
        denominator[denominator == 0] = 1  # Avoid the denominator is zero, set it as 1 
        NDVI = numerator / denominator
        NDVI[denominator == 0] = 0  # the denominator is zero, set it as 1
        
        # Build Mask 
        mask = NDVI < self.NDVI_TH

        # Generate the HSI
        Transposed_HSI = np.transpose(HSI, (0, 2, 1)) # exChange the second and the third dimension
        Transposed_HSI[mask,:] = 0
        HSI = np.transpose(Transposed_HSI, (0, 2, 1))

        # Get the plants position
        Plant_pos = np.argwhere(~mask).tolist()

        # Get the number of remove
        BG_counter += np.count_nonzero(mask)
        print("BG_counter is", BG_counter)

        return HSI, np.array(Plant_pos), BG_counter, NDVI

    # Get the Level1 HSI by removing the background 
    def getLevel1(self):
        lines = self.HSI_info[0]
        channels= self.HSI_info[1]
        samples = self.HSI_info[2]
        wavelengths = self.HSI_info[4]
        PixelSum = lines * samples

        ### Level 1. Remove the background
        Level_1 = self.getPlantPos()
        HSI_1 = Level_1[0]
        BG_Counter = Level_1[2]
        NDVI = Level_1[3]
        HSI_info_L1 = [lines, channels, samples, HSI_1, wavelengths]
        proportion_1 = float((PixelSum - BG_Counter)/PixelSum)

        return HSI_info_L1, BG_Counter, proportion_1, NDVI


if __name__ == "__main__":
    HSI_info = ReadData.Read()
    l1 = level1
    #Level1 = l1.getLevel1(HSI_info, NDVI_SET_VALUE)
    #Plant_Pos = getPlantPos(HSI_info)[1]
    #print(len(Plant_Pos))
    ReadData.drawImg(HSI_info,"pre_processing/Level1_img")


    