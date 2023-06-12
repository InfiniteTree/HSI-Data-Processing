import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d

import ReadData
import RemoveBG
import RemoveDB

class Reflectance:
    HSI_info = []
    cur_proportion = 1
    BRF_positionRange = []  # position range of the refer board
    BRF_files = "" # path location fo the 3% and 30% Refer board Cali files

    def __init__(self, HSI_info, cur_proportion, BRF_positionRange, brf_files):
        self.HSI_info = HSI_info
        #print(self.HSI_info)
        self.cur_proportion = cur_proportion
        self.BRF_positionRange = BRF_positionRange
        self.BRF_files = brf_files
        #print(self.BRF_positionRange)


    def getReferAmplititudes(self, BRF_flag):
        channels = self.HSI_info[1]
        HSI = self.HSI_info[3]
        #positionRange  i.e. (860,83),(890,120) for 3% Ref Board
        Amplititudes = []

        if BRF_flag == "3":
            RefHSI = HSI[self.BRF_positionRange[0][0][1]:self.BRF_positionRange[0][1][1],:,self.BRF_positionRange[0][0][0]:self.BRF_positionRange[0][1][0]]
        elif BRF_flag == "30":
             RefHSI = HSI[self.BRF_positionRange[1][0][1]:self.BRF_positionRange[1][1][1],:,self.BRF_positionRange[1][0][0]:self.BRF_positionRange[1][1][0]]
        
        for i in range(channels):
            Amplititudes.append(np.array(RefHSI[:,i,:]).mean())

        return Amplititudes


    def writeRef(self, filename, wavelengths, RefAmplititudes1, RefAmplititudes2):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(wavelengths)
            writer.writerow(RefAmplititudes1)
            writer.writerow(RefAmplititudes2)

    def readRef(self, filename):
        with open(filename,"r") as f:
            raw = f.readlines()
            str = ""
            str = ','.join(raw)
            str = str.replace("\n","").replace("波长,反射率(%)","").replace("波长","").replace("反射率%","").replace(" ","")
            str = str.split(",")
            result_list = [x for x in str if x != '']
            #print("Now is",result_list)
            waves = [float(x) for x in result_list[0::2]] 
            reflect = [float(x) for x in result_list[1::2]] 
        return waves,reflect

    def mapRef(self, inputfile, outputfile, lineNum):
        X, Y = [], []

        Read_info = self.readRef(outputfile)
        waves = Read_info[0] 
        k = 0
        with open(inputfile, "r") as f1:
            contents = f1.readlines()
            RefWaves = [float(items) for items in contents[0].split(",")]
            RefReflect = [float(items) for items in contents[lineNum].split(",")]

        #print (contents[0])
        #print("------\n",contents[1])
            channels = 300
            num = 60
            #print("RefWaves is",RefWaves)
            #print("waves is", waves)
            for i in range(channels):
                for k in range(num):
                    if abs(RefWaves[i] - waves[k]) < 1:
                        X.append(RefReflect[i])
                        #print(Read_info[1])
                        Y.append(Read_info[1][k]/100)
                    
        return X, Y

    def getReflectEquation(self):
        channels = self.HSI_info[1]
        RefAmplititudes_file ="Results/test/RefAmplititudes.csv"
        #print("------------3Ref----------")
        refSample1= self.mapRef(RefAmplititudes_file, self.BRF_files[0], 1)
        X1 = refSample1[0]
        Y1 = refSample1[1]
        '''
        print("-----------------X1-------------------")
        print(X1)
        print("-----------------Y1-------------------")
        print(Y1)
        '''
        #print("------------30Ref----------")

        refSample2 = self.mapRef(RefAmplititudes_file, self.BRF_files[1], 2)

        X2 = refSample2[0]
        Y2 = refSample2[1] 
        '''
        print("-----------------X2-------------------")
        print(X2)
        print("-----------------Y2-------------------")
        print(Y2)
        '''

        # get the vector of k and b
        num = len(X2)
        k, b = [], []

        for i in range(num):
            k.append((Y2[i] - Y1[i]) / (X2[i] - X1[i]))
            b.append(Y1[i] - k[i]*X1[i])
        '''
        print("-----------------k-------------------")
        print(k)
        print("-----------------b-------------------")
        print(b)
        '''
        
        # Now length of k, b is 57
        # we need 300 k,b
        # For the rest of k,b we can use interpolation algorithm to get the rest of them
        k = self.interpolate_list(k, channels)
        b = self.interpolate_list(b, channels)
        # Need to be checked here!!!
        return k,b


    def getReflectance(self, k, b):
        HSI = self.HSI_info[3]
        lines =  self.HSI_info[0]
        channels = self.HSI_info[1]
        samples = self.HSI_info[2]
        wavelengths = self.HSI_info[4]

        ReflectMatrix = np.zeros((lines, channels, samples))
        #print(ReflectMatrix.shape)

        for idx in range(channels):
            ReflectMatrix[:, idx, :] = (HSI[:, idx, :] * k[idx] + b[idx]) * self.cur_proportion ### errors remian here
        
        #print(ReflectMatrix.shape)
        print("ReflectMatrix is obtained.")
        return ReflectMatrix


    # interpolation algorithm to come up with the rest of k,b
    def interpolate_list(self, input_list, output_length):
        # x-axis raw_Data
        x = np.linspace(0, 1, len(input_list))
        # use the interpolation function
        f = interp1d(x, input_list, kind='linear')
        # make the output x
        output_x = np.linspace(0, 1, output_length)
        # interpolate
        output_list = f(output_x)
        
        return output_list.tolist()

    def getRefBoard(self):
        wavelengths = self.HSI_info[4]

        #"------------For the 3% Ref Board------------")
        three_RefAmplititudes = self.getReferAmplititudes("3")
        #"------------For the 30% Ref Board------------")
        thirty_RefAmplititudes = self.getReferAmplititudes("30")
        RefAmplititudes_file ="Results/test/RefAmplititudes.csv"
        self.writeRef(RefAmplititudes_file, wavelengths, three_RefAmplititudes, thirty_RefAmplititudes)

    def getLeafAvgReflect(self, ReflectMatrix):
        AvgReflect = ReflectMatrix.mean(axis=(0,2))
        return AvgReflect

    def getReflectMatrix(self):
        # Part 1. Get the reference Board Amplititudes
        #positionRange2 = [[870,95],[880,105]] 

        self.getRefBoard()

        equation = self.getReflectEquation()
        k = equation[0]
        b = equation[1]

        # Part 2. Map the RefBoard Amplititudes with the reflections
        # Read the particular plot image
        HSI_info_L2, proportion_2 = self.HSI_info, self.cur_proportion
        #print(proportion_2)
        # Level 3. Get the HSI reflectances
        ReflectMatrix = self.getReflectance(k, b)
        return ReflectMatrix

