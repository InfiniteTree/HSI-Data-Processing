import ReadData
import numpy as np
from PIL import Image

class preprocess:
    HSI_info = [] # shape = (n, k, m)
    HSI = []
    lines = 0
    channels = 0
    samples = 0
    wavelengths = []
    PixelSum = 0
    cur_proportion = 1

    ######------------------------------ Level 1 Data -------------------------------######
    NDVI_TH = 0.8
    NDVI = [] # shape = (n, 1, m)
    band800 = 195
    band670 = 134

    ######------------------------------ Level 2 Data -------------------------------######
    ShadowTHValue = 100 # the set mean threshold Hyspectra value for the leaves in shadow
    BrightTHValue = 800 # the set mean threshold Hyspectra value for the leaves in bright
    set_value = [0, 0, 0] # set the dealt value as 0 to represent remove the useless information

    BG_Counter = 0 # the counter of the background pixels
    SD_Counter = 0 # the counter of the shadow pixels
    BT_Counter = 0 # the counter of the bright pixels
    DB_Counter = 0 # the counter of the bright pixels + shadow pixels
    
    def __init__(self, hsiInfo, ndviTh, ampl_LowTH, ampl_HighTH):
        self.NDVI_TH = ndviTh
        self.HSI_info = hsiInfo
        self.ShadowTHValue = ampl_LowTH
        self.BrightTHValue = ampl_HighTH

        self.HSI = self.HSI_info[3]
        self.lines = self.HSI_info[0]
        self.channels = self.HSI_info[1]
        self.samples = self.HSI_info[2]
        self.wavelengths = self.HSI_info[4]
        self.PixelSum = self.lines * self.samples
        

    ######-------------------------- Level 1 proprocessing ---------------------------######
    ###----------------------------- Remove the background  -----------------------------###
    def getPlantPos(self):
        Plant_pos = []
        BG_Counter = 0
        HSI = np.array(self.HSI)

        numerator = HSI[:, self.band800, :] - HSI[:, self.band670, :]
        denominator = HSI[:, self.band800, :] + HSI[:, self.band670, :]
        denominator[denominator == 0] = 1  # Avoid the denominator is zero, set it as 1 
        NDVI = numerator / denominator
        NDVI[denominator == 0] = 0  # the denominator is zero, set it as 1
        
        # Build Mask 
        level1_mask = NDVI < self.NDVI_TH

        # Generate the HSI
        Transposed_HSI = np.transpose(HSI, (0, 2, 1)) # exChange the second and the third dimension
        Transposed_HSI[level1_mask,:] = 0
        HSI = np.transpose(Transposed_HSI, (0, 2, 1))

        # Get the plants position
        Plant_pos = np.argwhere(~level1_mask).tolist()

        # Get the number of remove
        self.BG_Counter = np.count_nonzero(level1_mask)
        print("BG_counter is", self.BG_Counter)
        self.HSI_info = [self.lines, self.channels, self.samples, HSI, self.wavelengths]

        return HSI, np.array(Plant_pos), self.BG_Counter, NDVI

    # Get the Level1 HSI by removing the background 
    def getLevel1(self):
        ### Level 1. Remove the background
        Level_1 = self.getPlantPos()
        HSI_1 = Level_1[0]
        self.BG_Counter = Level_1[2]
        NDVI = Level_1[3]
        HSI_info_L1 = [self.lines, self.channels, self.samples, HSI_1, self.wavelengths]
        self.cur_proportion = float((self.PixelSum - self.BG_Counter)/self.PixelSum)

        return HSI_info_L1, self.BG_Counter, self.cur_proportion, NDVI
    
    ######---------------------------- Level 2 proprocessing ----------------------------######
    ###----------------------------- Remove the dark and bright  ---------------------------###
    def calcAmplMean(self):
        return self.HSI.mean(axis=1) / self.cur_proportion

    def RemoveDB(self, Remove_type):
        HSI = self.HSI
        #print(HSI_bandmean.mean())
        #print(HSI_bandmean)
        HL_position = [] # High lighted plant position
        
        MaxAmplMatrix = np.max(HSI, axis=1)
        
        if Remove_type == "SD":
            level2_1_mask = MaxAmplMatrix < self.ShadowTHValue
            # Generate the HSI
            Transposed_HSI = np.transpose(HSI, (0, 2, 1)) # exChange the second and the third dimension
            Transposed_HSI[level2_1_mask,:] = 0
            HSI = np.transpose(Transposed_HSI, (0, 2, 1))

            self.SD_Counter = np.count_nonzero(level2_1_mask)
            # Get the position withought dark and bright
            HL_position = np.argwhere(~level2_1_mask).tolist() # Bugs remains here !!! Need more numpy operation

        elif Remove_type == "BT":
            level2_2_mask = MaxAmplMatrix > self.BrightTHValue
            # Generate the HSI
            Transposed_HSI = np.transpose(HSI, (0, 2, 1)) # exChange the second and the third dimension
            Transposed_HSI[level2_2_mask,:] = 0
            HSI = np.transpose(Transposed_HSI, (0, 2, 1))

            self.BT_Counter = np.count_nonzero(level2_2_mask)
            print("BTCounter is",self.BT_Counter)
            # Get the position withought dark and bright
            HL_position = np.argwhere(~level2_2_mask).tolist() # Bugs remains here !!! Need more numpy operation     
        
        self.DB_Counter = self.SD_Counter + self.BT_Counter
        if Remove_type == "SD":
            print("SDCounter is",self.SD_Counter)
            self.cur_proportion = float((self.PixelSum * self.cur_proportion - self.SD_Counter) / self.PixelSum)
            print("cur_proportion is", self.cur_proportion)

        if Remove_type == "BT":
            print("BTCounter is",self.BT_Counter)
            self.cur_proportion = float((self.PixelSum * self.cur_proportion - self.BT_Counter) / self.PixelSum)
            print("cur_proportion is", self.cur_proportion)

        return HSI, self.DB_Counter, np.array(HL_position), self.cur_proportion


    # Get the Level2 HSI by removing the shadows
    def getLevel2(self):
        # Remove the blades in shadow
        HSI_2, DB_Counter, HL_position, proportion_2 = self.RemoveDB("SD")
        # Remove the blades that are too bright
        HSI_2, DB_Counter, HL_position, proportion_2 = self.RemoveDB("BT")
        HSI_info_L2 = [self.lines, self.channels, self.samples, HSI_2, self.wavelengths]

        return HSI_info_L2, DB_Counter, HL_position, proportion_2


    