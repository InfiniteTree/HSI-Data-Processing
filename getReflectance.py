import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d

import ReadData
import RemoveBG

def getReferAmplititudes(HSI_info, positionRange):
    channels = HSI_info[1]
    HSI = HSI_info[3]
    #positionRange  i.e. (860,83),(890,120) for 3% Ref Board
    Amplititudes = []
    RefHSI = HSI[positionRange[0][0]:positionRange[1][0],:,positionRange[0][1]:positionRange[1][1]]
    for i in range(channels):
        Amplititudes.append(np.array(RefHSI[:,i,:]).mean())
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
                
    #print("X is",X)
    #print("Y is",Y)
    #print("length of x",len(X))
    #print("length of y",len(Y))
    return X, Y



def getReflectance(HSI_info):
    k , b = [], []
    HSI = HSI_info[3]
    lines =  HSI_info[0]
    channels = HSI_info[1]
    samples = HSI_info[2]

    #print("------------3Ref----------")
    refSample1= mapRef(RefAmplititudes_file, "RefBoard/3Ref.csv", 1)
    X1 = refSample1[0]
    Y1 = refSample1[1]

    #print("------------30Ref----------")
    refSample2 = mapRef(RefAmplititudes_file, "RefBoard/30Ref.csv", 2)

    X2 = refSample2[0]
    Y2 = refSample2[1] 
    # get the vector of k and b
    num = 57 # Need to reset here
    print("-------------------------------------------------")
    print("X2 is", X2)
    print("Y2 is", Y2)

    for i in range(num):
        k.append((Y2[i] - Y1[i]) / (X2[i] - X1[i]))
        b.append(Y2[i] - k[i]*X2[i])
    

    #print("----------k, b----------------")

    # Now length of k, b is 57
    # we need 300 k,b
    # For the rest of k,b we can use interpolation algorithm to get the rest of them
    k = interpolate_list(k, channels)
    b = interpolate_list(b, channels)
    #print(len(k))
    ReflectMatrix = np.zeros((lines, channels, samples))
    #print(ReflectMatrix.shape)

    for idx in range(channels):
        ReflectMatrix[:, idx, :] = HSI[:, idx, :] * k[idx] + b[idx]

    return ReflectMatrix, k, b


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




if __name__ == "__main__":
    HSI_info = ReadData.ReadData()
    #getReflectance(HSI_info)
    wavelengths = HSI_info[4]

    # Part 1. Get the reference Board Amplititudes
    #"------------For the 3% Ref Board------------")
    positionRange = [[870,95],[880,105]]
    three_RefAmplititudes = getReferAmplititudes(HSI_info, positionRange)
    #"------------For the 30% Ref Board------------")
    positionRange = [[870,205],[880,215]]
    thirty_RefAmplititudes = getReferAmplititudes(HSI_info, positionRange)
    RefAmplititudes_file ="Results/RefAmplititudes.csv"
    writeRef(RefAmplititudes_file, wavelengths, three_RefAmplititudes, thirty_RefAmplititudes)
    # Part 2. Map the RefBoard Amplititudes with the reflections
    getReflectance(HSI_info)
 