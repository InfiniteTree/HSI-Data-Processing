import numpy as np
import math
import csv
import matplotlib.pyplot as plt

import ReadData
import RemoveBG
import RemoveSD
from setting import map_band


def calImgSpecMean(HSI,proportion):
    return HSI.mean(axis=(0,2)) / proportion

# Calculate the relative parameters of the photosynthesis
def calcPhenotypeParas(reflectances):
    NDVI = (reflectances[map_band["band800"]] - reflectances[map_band["band680"]]) / (reflectances[map_band["band800"]] + reflectances[map_band["band680"]])
    OSAVI = (1+0.16) * (reflectances[map_band["band800"]] - reflectances[map_band["band670"]]) / (reflectances[map_band["band800"]] + reflectances[map_band["band670"]]+ 0.16)
    PSSRa = reflectances[map_band["band800"]] / reflectances[map_band["band680"]]
    PSSRb = reflectances[map_band["band800"]] / reflectances[map_band["band635"]]
    PRI = (reflectances[map_band["band570"]] - reflectances[map_band["band531"]]) / (reflectances[map_band["band570"]] + reflectances[map_band["band531"]])
    MTVI2 = 1.5 * (1.2 * (reflectances[map_band["band800"]] - reflectances[map_band["band550"]]) - 2.5 * (reflectances[map_band["band670"]] - reflectances[map_band["band550"]])) / math.sqrt(((2 * reflectances[map_band["band800"]]+1)*2 - (6*reflectances[map_band["band800"]]-5*math.sqrt(reflectances[map_band["band670"]]))-0.5))
    #MTVI2 = 0
    PhenotypeParas = [NDVI, OSAVI, PSSRa, PSSRb, PRI, MTVI2]
    
    return PhenotypeParas

# export file 1 to store the spectra data  
def HyperspectraCurve(HSI_info, proportion, avg_flag):
    # Show the spectra curve
    wavelengths = HSI_info[4]
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    HSI = HSI_info[3]
    if avg_flag == 1:
        spec_mean = calImgSpecMean(HSI,proportion)
        x = [float(num) for num in wavelengths] # change str to float
        y = np.array(spec_mean)
        plt.xlabel("Wavelength(nm)")
        plt.ylabel("Hyperspectral Luminance")
        plt.title("The Average Hyperspectral of the Light Blades")
        plt.plot(x,y,c='lightcoral',label='Curve_poly_Fit')
        plt.savefig("Results/Hyperspec_curve.jpg")
        remainRow = spec_mean
    elif avg_flag == 0 :
        remainRow = []
        
    #plt.show()

    # Export the data of hyperspectra curve into the local .csv
    FirstRow = wavelengths
    
    curveFile = "Results/Hyperspectra_curve.csv"
    with open(curveFile,"w",newline='') as f:
        writer = csv.writer(f)
        # Write the first row
        writer.writerow(FirstRow)
        # Write the remaining rows
        if avg_flag == 1:
            writer.writerow(remainRow)
        elif avg_flag == 0:
            for i in range(samples*lines):
                writer.writerow(remainRow[i])

    return curveFile

# export file 2 to store the Parameters of Phenotype
def exportPhenotypeParas(Readfilename):
    FirstRow = ["NDVI","OSAVI","PSSRb","PSSRb", "RPI","MTVI2","AC","SPAD"]
    
    with open(Readfilename,"r",newline='') as f:
        contents = f.readlines()
        reflectances = contents[1].split(",")
        #print(reflectances)
        reflectances = [float(num) for num in reflectances] # change str to float
        PhenotypeParas= calcPhenotypeParas(reflectances)

    with open("Results/Phenotype_Paras_withReflectance.csv","w",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FirstRow)
        writer.writerow(PhenotypeParas)
    return

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
    
    
    HyperspectraCurve(HSI_info_L2, proportion_2, 1)
    #curve_file = HyperspectraCurve(HSI_info_L2, proportion_2, 1)
    curve_file = "Results/ReflectCurve.csv"
    exportPhenotypeParas(curve_file)
