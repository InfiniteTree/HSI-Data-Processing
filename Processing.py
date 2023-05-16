import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split

import ReadData
import RemoveBG
import RemoveSD
from setting import map_band
import GetReflectance


def calImgSpecMean(HSI,proportion):
    return HSI.mean(axis=(0,2)) / proportion

# Calculate the relative parameters of the photosynthesis
def getPhenotypeParas(reflectances, avg_flag):
    if reflectances[map_band["band800"]] == 0 and avg_flag == 1:
        PhenotypeParas = ["NA"]*8
    else:
        # Get the first six parameters of the photosynthesis by simple mathmatical calculation
        NDVI = (reflectances[map_band["band800"]] - reflectances[map_band["band680"]]) / (reflectances[map_band["band800"]] + reflectances[map_band["band680"]])
        OSAVI = (1+0.16) * (reflectances[map_band["band800"]] - reflectances[map_band["band670"]]) / (reflectances[map_band["band800"]] + reflectances[map_band["band670"]]+ 0.16)
        PSSRa = reflectances[map_band["band800"]] / reflectances[map_band["band680"]]
        PSSRb = reflectances[map_band["band800"]] / reflectances[map_band["band635"]]
        PRI = (reflectances[map_band["band570"]] - reflectances[map_band["band531"]]) / (reflectances[map_band["band570"]] + reflectances[map_band["band531"]])
        #MTVI2 = 1.5 * (1.2 * (reflectances[map_band["band800"]] - reflectances[map_band["band550"]]) - 2.5 * (reflectances[map_band["band670"]] - reflectances[map_band["band550"]])) / math.sqrt(((2 * reflectances[map_band["band800"]]+1)*2 - (6*reflectances[map_band["band800"]]-5*math.sqrt(reflectances[map_band["band670"]]))-0.5))
        MTVI2 = 0
        # Get the remaining parameters by using the trained model to predict
        if avg_flag == 1:
            data = pd.read_csv("model/LearningData/TrainData.csv")
            print("Dataset of Train Model loaded...")
            train_x = data.drop(['SPAD',"A1200", "N", "Ca", "Cb"],axis=1)
            train_y = data[['SPAD',"A1200", "N", "Ca", "Cb"]].copy()
            train_x = pd.DataFrame(train_x, dtype='float32')
            train_y = pd.DataFrame(train_y, dtype='float32')

            # pls_param_grid = {'n_components': list(range(10,20))}
            pls_param_grid = {'n_components':[10]}  
            pls = GridSearchCV(PLSRegression(), param_grid=pls_param_grid,scoring='r2',cv=10)
            pls.fit(train_x, train_y)

            test_x = reflectances[6:-16] # The data set of the train model only contains HS in parts of wavelength range
            test_x = pd.Series(test_x, dtype='float32')
            test_x = test_x.to_frame().T
            
            #print(test_x)
            y_pre = pls.predict(test_x)
            new_data = y_pre

            SPAD = new_data[0][0]
            A1200 = new_data[0][1]
            N = new_data[0][2]
            Ca = new_data[0][3]
            Cb = new_data[0][4]
            PhenotypeParas = [NDVI, OSAVI, PSSRa, PSSRb, PRI, MTVI2, SPAD, A1200, N, Ca, Cb]

        elif avg_flag == 0:
            PhenotypeParas = [NDVI, OSAVI, PSSRa, PSSRb, PRI, MTVI2]
    
    return PhenotypeParas

# export file 1 to store the spectra data  
def HyperspectraCurve(HSI_info, proportion):
    # Show the spectra curve
    wavelengths = HSI_info[4]
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    HSI = HSI_info[3]
    remainRow = []

    spec_mean = calImgSpecMean(HSI,proportion)
    x = [float(num) for num in wavelengths] # change str to float
    y = np.array(spec_mean)
    plt.xlabel("Wavelength(nm)")
    plt.ylabel("Hyperspectral Luminance")
    plt.title("The Average Hyperspectral of the Light Blades")
    plt.plot(x,y,c='lightcoral',label='Curve_poly_Fit')
    plt.savefig("Results/Hyperspec_curve.jpg")
    remainRow = spec_mean
        
    #plt.show()

    # Export the data of hyperspectra curve into the local .csv
    FirstRow = wavelengths
    
    curveFile = "Results/Hyperspectra_Avg_curve.csv"
    with open(curveFile,"w",newline='') as f:
        writer = csv.writer(f)
        # Write the first row
        writer.writerow(FirstRow)
        # Write the remaining rows
        writer.writerow(remainRow)

    return curveFile

# export file to store the Parameters of Phenotype in terms of the single plot
def exportPhenotypeParas(Readfilename):

    FirstRow = ["NDVI", "OSAVI", "PSSRa", "PSSRb", "RPI", "MTVI2", "SPAD","A1200", "N", "Ca", "Cb"]
    avg_flag = 1
    # Read the reflectance file and calculate the Phenotype parameters
    with open(Readfilename,"r",newline='') as f:
        contents = f.readlines()
        reflectances = contents[1].split(",")
        #print(reflectances)
        reflectances = [float(num) for num in reflectances] # change str to float
        PhenotypeParas= getPhenotypeParas(reflectances, avg_flag)

    # Export the results
    with open("Results/Phenotype_Paras_withReflectance.csv","w",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FirstRow)
        writer.writerow(PhenotypeParas)

    return

# export file to store the Parameters of Phenotype in terms of the single pixel
def exportPhenotypeParas_eachPixel(HSI_info,reflectanceMatrix):
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]

    FirstRow = ["Loc","NDVI", "OSAVI", "PSSRb", "PSSRb", "RPI", "MTVI2"]

    avg_flag = 0
    PhenotypeParas= []


    # Export the results
    with open("Results/Phenotype_Paras_eachPixel.csv","w",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FirstRow)
        for i in range(samples*lines):
            row = i//samples
            col = i%samples
            PhenotypeParas = getPhenotypeParas(reflectanceMatrix[row,:,col], avg_flag)
            writer.writerow([(row,col)]+PhenotypeParas)

if __name__ == "__main__":
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
    HSI_2 = Level_2[0]
    HSI_info_L2 = [lines, channels, samples, HSI_2,  wavelengths]

    SD_Counter = Level_2[1]
    proportion_2 = float((PixelSum - SD_Counter - BG_Counter)/PixelSum)
    print("PixelSum is",PixelSum)
    '''
    print("SD_Counter is",SD_Counter)
    print("BG_Counter is",BG_Counter)
    '''
    print("The remaining proportion is ",proportion_2)

    #Get the reference Board Amplititudes
    Ref_HSI_info = ReadData.ReadRef()
    wavelengths = Ref_HSI_info[4]
    lines = Ref_HSI_info[0]
    channels= Ref_HSI_info[1]
    samples = Ref_HSI_info[2]

    positionRange1 = [[870,205],[880,215]] # The right black board with more degree of darkness(r=3%)
    positionRange2 = [[870,95],[880,105]] # The left black board with less degree of darkness(r=30%)
    GetReflectance.getRefBoard(Ref_HSI_info, positionRange1, positionRange2)
    equation = GetReflectance.getReflectEquation(Ref_HSI_info)
    k = equation[0]
    b = equation[1]

    reflectanceMatrix = GetReflectance.getReflectance(HSI_info, proportion_2, k, b)
    #print(reflectanceMatrix)
    
    avg_flag = 0
    if (avg_flag == 1):
        HyperspectraCurve(HSI_info_L2, proportion_2)
        avg_flag = 0

    if (avg_flag == 0):
        exportPhenotypeParas_eachPixel(HSI_info_L2, reflectanceMatrix)

    curve_file = "Results/ReflectCurve.csv"
    ### exportPhenotypeParas(curve_file)
