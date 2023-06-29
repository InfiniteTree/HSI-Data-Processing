import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d

import ReadData
import Preprocess

class Reflectance:
    HSI_info = []
    cur_proportion = 1
    BRF_positionRange = []  # position range of the refer board
    BRF_files = "" # path location fo the 3% and 30% Refer board Cali files
    AVG_reflect = [] # 1-D average reflectance for the whole plot in different bands
    ReflectMatrix = [] # 3-D reflectance matrix for the whole plot
    k = [] # i.e. 300 k for 300 bands
    b = [] # i.e. 300 b for 300 bands
    plantMask = []
    
    def __init__(self, HSI_info, cur_proportion, BRF_positionRange, brf_files, k, b, plant_mask):
        self.HSI_info = HSI_info
        #print(self.HSI_info)
        self.cur_proportion = cur_proportion
        self.BRF_positionRange = BRF_positionRange
        self.BRF_files = brf_files
        self.k = k
        self.b = b
        self.plantMask = plant_mask
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
        X_map, Y_map, orig_X = [], [], []

        Read_info = self.readRef(outputfile)
        waves = Read_info[0] 
        k = 0
        with open(inputfile, "r") as f1:
            contents = f1.readlines()
            RefWaves = [float(items) for items in contents[0].split(",")]
            RefReflect = [float(items) for items in contents[lineNum].split(",")]

        #print (contents[0])
        #print("------\n",contents[1])
            channels = self.HSI_info[1]
            num = len(waves)
            #print(num)
            #print("RefWaves is",RefWaves)
            #print("waves is", waves)
            min_diff = float('inf')
            min_index = None
            '''
            for k in range(num):
                for i in range(channels):
                    diff = abs(RefWaves[i] - waves[k])
                    if diff < min_diff:
                        min_diff = diff
                        min_index = i
                if min_index is not None:
                    X_map.append(RefReflect[min_index])
                Y_map.append(Read_info[1][k]/100)

            
            '''
            for i in range(channels):
                for k in range(num):
                    if abs(RefWaves[i] - waves[k]) < 1:
                        X_map.append(RefReflect[i])
                        #print(Read_info[1])
                        Y_map.append(Read_info[1][k]/100)
            orig_X = RefReflect

        return X_map, Y_map, orig_X

    def getReflectEquation(self):
        self.getRefBoard()

        channels = self.HSI_info[1]
        RefAmplititudes_file ="results/test/RefAmplititudes.csv"
        #print("------------3Ref----------")

        if self.BRF_files[0][-6:] == "30.csv":
            BRF_30 = self.BRF_files[0]
            BRF_3 = self.BRF_files[1]
        else:
            BRF_30 = self.BRF_files[1]
            BRF_3 = self.BRF_files[0]


        refSample1= self.mapRef(RefAmplititudes_file, BRF_3, 1) # 30Ref 

        X1 = refSample1[2]
        X1_map = refSample1[0]
        Y1_map = refSample1[1]

    
        refSample2 = self.mapRef(RefAmplititudes_file, BRF_30, 2)

        X2 = refSample2[2]
        X2_map = refSample2[0]
        Y2_map = refSample2[1]
        
        #Y1 = np.interp(X1, X1_map, Y1_map)
        #Y2 = np.interp(X2, X2_map, Y2_map)

        # get the vector of k and b
        '''
        Y1 = self.interpolate_list(X1_map, Y1_map, X1)
        Y2 = self.interpolate_list(X2_map, Y2_map, X2)
        print("-----------------X1-------------------")
        print(X1)
        print("-----------------Y1-------------------")
        print(Y1)
        print("-----------------X2-------------------")
        print(X2)
        print("-----------------Y2-------------------")
        print(Y2)
        '''
        for i in range(len(X1_map)):
            self.k.append((Y2_map[i] - Y1_map[i]) / (X2_map[i] - X1_map[i]))
            self.b.append(Y2_map[i] - self.k[i]*X2_map[i])
        #print(len(self.k))
        # By Here the k and b is absolutely correct!!!
        
        # Now length of k, b is 60
        # we need 300 k,b
        # For the rest of k,b we can use interpolation algorithm to get the rest of them
        self.k = self.interpolate_list(self.k, channels)
        self.b = self.interpolate_list(self.b, channels)
        '''
        print("-----------------k-------------------")
        print(self.k)
        print("-----------------b-------------------")
        print(self.b)
        '''
        
        # Need to be checked here!!!
        return self.k, self.b


    def getReflectance(self):
        HSI = self.HSI_info[3]
        lines =  self.HSI_info[0]
        channels = self.HSI_info[1]
        samples = self.HSI_info[2]

        ReflectMatrix = np.zeros((lines, channels, samples))
        #print(ReflectMatrix.shape)

        for idx in range(channels):
            ReflectMatrix[:, idx, :] = HSI[:, idx, :] * self.k[idx] + self.b[idx] 
        
        # Ensure the values in the reflect matrix within [0,1]
        ReflectMatrix = np.where(ReflectMatrix < 0, 0, ReflectMatrix)
        ReflectMatrix = np.where(ReflectMatrix > 1, 1, ReflectMatrix)

        #print(ReflectMatrix.shape)
        print("ReflectMatrix is obtained.")
        return ReflectMatrix


    # interpolation algorithm to come up with the rest of k,b
    def interpolate_list(self, input_list, output_length):
        # x-axis raw_Data
        x = np.linspace(400, 990, len(input_list))
        # use the interpolation function
        f = interp1d(x, input_list, kind='linear',fill_value='extrapolate')
        # make the output x
        output_x = np.linspace(394, 1034, output_length)
        # interpolate
        output_list = f(output_x)
        return output_list

    def getRefBoard(self):
        wavelengths = self.HSI_info[4]
        #"------------For the 3% Ref Board------------")
        three_RefAmplititudes = self.getReferAmplititudes("3")
        #"------------For the 30% Ref Board------------")
        thirty_RefAmplititudes = self.getReferAmplititudes("30")
        RefAmplititudes_file ="results/test/RefAmplititudes.csv"
        self.writeRef(RefAmplititudes_file, wavelengths, three_RefAmplititudes, thirty_RefAmplititudes)

    def getLeafAvgReflect(self):
        Transposed_ReflectMatrix = np.transpose(self.ReflectMatrix, (0, 2, 1))
        plant_ReflectMatrix = Transposed_ReflectMatrix[~self.plantMask, :]
        self.AVG_reflect = plant_ReflectMatrix.mean(axis=0)
        
    def getReflect(self):
        self.ReflectMatrix = self.getReflectance()
        self.getLeafAvgReflect()
        wavelengths = self.HSI_info[4][2:-22]
        FirstRow = wavelengths
        ReflectRow = self.AVG_reflect[2:-22]
        with open("results/test/ReflectCurve.csv","w",newline='') as f:
            writer = csv.writer(f)
            # Write the first row
            writer.writerow(FirstRow)
            # Write the remaining rows
            writer.writerow(ReflectRow)

    def visualizeReflect(self, saveFlag):
        # Write the reflectvector into a local csv file
        # Plotting
        x = np.array(self.HSI_info[4])[2:-22] # only use the data within 400nm to 990nm
        y = np.array(self.AVG_reflect)[2:-22] # only use the data within 400nm to 990nm
        plt.xticks(range(400, 1000, 100))
        plt.plot(x,y,c='b',label='Curve_poly_Fit')
        #plt.xticks(range(400, 1000, 100))
        plt.title("The Reflectance curve of the whole plot")
        if saveFlag ==1:
            plt.savefig("results/test/Reflectance_curve_new.jpg")
        elif saveFlag ==0:
            plt.show()

