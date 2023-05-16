import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d

import ReadData
import RemoveBG
import RemoveSD

def getReferAmplititudes(HSI_info, positionRange):
    channels = HSI_info[1]
    HSI = HSI_info[3]
    #positionRange  i.e. (860,83),(890,120) for 3% Ref Board
    Amplititudes = []
    RefHSI = HSI[positionRange[0][0]:positionRange[1][0],:,positionRange[0][1]:positionRange[1][1]]
    for i in range(channels):
        Amplititudes.append(np.array(RefHSI[:,i,:]).mean())
    #print(Amplititudes)
    
    return Amplititudes


def writeRef(filename, wavelengths, RefAmplititudes1, RefAmplititudes2):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(wavelengths)
        writer.writerow(RefAmplititudes1)
        writer.writerow(RefAmplititudes2)

def readRef(filename):
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

def mapRef(inputfile, outputfile,lineNum):
    X, Y = [], []

    Read_info = readRef(outputfile)
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

def getReflectEquation(HSI_info):
    channels = HSI_info[1]
    RefAmplititudes_file ="Results/RefAmplititudes.csv"
    #print("------------3Ref----------")
    refSample1= mapRef(RefAmplititudes_file, "RefBoard/3Ref.csv", 1)
    X1 = refSample1[0]
    Y1 = refSample1[1]
    '''
    print("-----------------X1-------------------")
    print(X1)
    print("-----------------Y1-------------------")
    print(Y1)
    '''
    #print("------------30Ref----------")

    refSample2 = mapRef(RefAmplititudes_file, "RefBoard/30Ref.csv", 2)

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
    k = interpolate_list(k, channels)
    b = interpolate_list(b, channels)
    # Need to be checked here!!!
    return k,b


def getReflectance(HSI_info, proportion_2, k, b):
    HSI = HSI_info[3]
    lines =  HSI_info[0]
    channels = HSI_info[1]
    samples = HSI_info[2]
    wavelengths = HSI_info[4]

    ReflectMatrix = np.zeros((lines, channels, samples))
    #print(ReflectMatrix.shape)

    for idx in range(channels):
        ReflectMatrix[:, idx, :] = (HSI[:, idx, :] * k[idx] + b[idx]) * proportion_2 ### errors remian here
    
    #print(ReflectMatrix.shape)
    print("ReflectMatrix is obtained.")
    return ReflectMatrix


# interpolation algorithm to come up with the rest of k,b
def interpolate_list(input_list, output_length):
    # x-axis raw_Data
    x = np.linspace(0, 1, len(input_list))
    # use the interpolation function
    f = interp1d(x, input_list, kind='linear')
    # make the output x
    output_x = np.linspace(0, 1, output_length)
    # interpolate
    output_list = f(output_x)
    
    return output_list.tolist()

def getRefBoard(HSI_info,positionRange1,positionRange2):
    wavelengths = HSI_info[4]
    #"------------For the 3% Ref Board------------")
    three_RefAmplititudes = getReferAmplititudes(HSI_info, positionRange1)
    #"------------For the 30% Ref Board------------")
    thirty_RefAmplititudes = getReferAmplititudes(HSI_info, positionRange2)
    RefAmplititudes_file ="Results/RefAmplititudes.csv"
    writeRef(RefAmplititudes_file, wavelengths, three_RefAmplititudes, thirty_RefAmplititudes)

def getLeafAvgReflect(ReflectMatrix):
    AvgReflect = ReflectMatrix.mean(axis=(0,2))
    return AvgReflect

def getReflectMatrix():
    # Part 1. Get the reference Board Amplititudes
    Ref_HSI_info = ReadData.ReadRef()
    wavelengths = Ref_HSI_info[4]
    lines = Ref_HSI_info[0]
    channels= Ref_HSI_info[1]
    samples = Ref_HSI_info[2]

    positionRange1 = [[870,205],[880,215]] # The right black board with more degree of darkness(r=3%)
    positionRange2 = [[870,95],[880,105]] # The left black board with less degree of darkness(r=30%)
    getRefBoard(Ref_HSI_info, positionRange1, positionRange2)
    equation = getReflectEquation(Ref_HSI_info)
    k = equation[0]
    b = equation[1]

    # Part 2. Map the RefBoard Amplititudes with the reflections
    # Read the particular plot image
    HSI_info = ReadData.Read()
    wavelengths = HSI_info[4]
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    PixelSum = lines * samples
    ### Level 1. Remove the background
    Level_1 = RemoveBG.getPlantPos(HSI_info)
    HSI_1 = Level_1[0]
    BG_Counter = Level_1[2]

    ### Level 2. Remove the shadow leaves
    set_value = [0, 0, 0]
    HSI_info_L1 = [lines, channels, samples, HSI_1, wavelengths]
    proportion_1 = float((PixelSum - BG_Counter)/PixelSum)
    Level_2 = RemoveSD.RemoveSD(HSI_info_L1,set_value, proportion_1)

    #getReflectance(HSI_info)
    HSI_2 = Level_2[0]
    HSI_info_L2 = [lines, channels, samples, HSI_2,  wavelengths]
    # Draw the img
    #ReadData.drawImg(HSI_info_L2, "RemoveL2_new")

    SD_Counter = Level_2[1]
    proportion_2 = float((PixelSum - SD_Counter - BG_Counter)/PixelSum)
    #print(proportion_2)
    ### Level 3. Get the HSI reflectances
    ReflectMatrix = getReflectance(HSI_info, proportion_2, k, b)
    return ReflectMatrix



if __name__ == "__main__":
    HSI_info = ReadData.Read()
    wavelengths = HSI_info[4]
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    ReflectMatrix = getReflectMatrix()
    avg_reflect = getLeafAvgReflect(ReflectMatrix)
    #print(avg_reflect)

    # Write the reflectvector into a local csv file
    FirstRow = wavelengths
    ReflectRow = avg_reflect
    with open("Results/ReflectCurve.csv","w",newline='') as f:
        writer = csv.writer(f)
        # Write the first row
        writer.writerow(FirstRow)
        # Write the remaining rows
        writer.writerow(ReflectRow)

    # Plotting
    x = np.array(HSI_info[4])
    y = np.array(avg_reflect)
    plt.plot(x,y,c='b',label='Curve_poly_Fit')
    #plt.xticks(range(400, 1000, 100))
    plt.savefig("Results/Reflectance_curve_new.jpg")
    plt.show()
    