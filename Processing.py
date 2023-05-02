import numpy as np
import math
import csv
import matplotlib.pyplot as plt

import ReadData
from RemoveBG import map_band 
import RemoveBG
import RemoveSD


def calImgSpecMean(HSI,proportion):
    return HSI.mean(axis=(0,2))

def calcPhenotypeParas(reflectances):
    NDVI = (reflectances[map_band["band800"]] - reflectances[map_band["band680"]]) / (reflectances[map_band["band800"]] + reflectances[map_band["band680"]])
    OSAVI = (1+0.16) * (reflectances[map_band["band800"]] - reflectances[map_band["band670"]]) / (reflectances[map_band["band800"]] + reflectances[map_band["band670"]]+ 0.16)
    PSSRa = reflectances[map_band["band800"]] / reflectances[map_band["band680"]]
    PSSRb = reflectances[map_band["band800"]] / reflectances[map_band["band635"]]
    PRI = (reflectances[map_band["band570"]] - reflectances[map_band["band531"]]) / (reflectances[map_band["band570"]] + reflectances[map_band["band531"]])
    MTVI2 = 1.5 * (1.2 * (reflectances[map_band["band800"]] - reflectances[map_band["band550"]]) - 2.5 * (reflectances[map_band["band670"]] - reflectances[map_band["band550"]])) / math.sqrt(((2 * reflectances[map_band["band800"]]+1)*2 - (6*reflectances[map_band["band800"]]-5*math.sqrt(reflectances[map_band["band670"]]))-0.5))
    PhenotypeParas = [NDVI, OSAVI, PSSRa, PSSRb, PRI, MTVI2]
    
    return PhenotypeParas

# export file 1 to store the spectra data  
def HyperspectraCurve(HSI,wavelengths, proportion):
    # Show the spectra curve
    spec_mean = calImgSpecMean(HSI,proportion)
    x = [float(num) for num in wavelengths] # change str to float
    y = np.array(spec_mean)
    plt.plot(x,y,c='lightcoral',label='Curve_poly_Fit')
    plt.savefig("Results/Hyperspec_curve.jpg")
    #plt.show()

    # Export the data of hyperspectra curve into the local .csv
    FirstRow = wavelengths
    curveFile = "Results/Hyperspectra_curve.csv"
    with open(curveFile,"w",newline='') as f:
        writer = csv.writer(f)
        # Write the first row
        writer.writerow(FirstRow)
        # Write the remaining rows
        writer.writerow(spec_mean)
    f.close()

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
        
    f.close()

    with open("Results/Phenotype_Paras_nonew.csv","w",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FirstRow)
        writer.writerow(PhenotypeParas)
    f.close()
    return

if __name__ == "__main__":
    HSI_info = ReadData.ReadData()
    wavelengths = HSI_info[4]
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    ### Level 1
    Level_1 = RemoveBG.getPlantPos(HSI_info)
    HSI = Level_1[0]
    BG_Counter = Level_1[2]

    ### Level 2
    set_value = [0, 0, 0]
    HSI_info_new = [lines, channels, samples, HSI]
    Level_2 = RemoveSD.RemoveSD(HSI_info_new,set_value)
    HSI = Level_2[0]
    SD_Counter = Level_2[1]
    
    PixelSum = lines * samples
    
    proportion = (PixelSum - SD_Counter - BG_Counter)/PixelSum
    print("PixelSum is",PixelSum)
    print("SD_Counter is",SD_Counter)
    print("BG_Counter is",BG_Counter)
    print("The remaining proportion is ",proportion)
    
    curve_file = HyperspectraCurve(HSI, wavelengths, proportion)
    exportPhenotypeParas(curve_file)
