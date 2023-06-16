import numpy as np
import matplotlib.pyplot as plt

import ReadData
import preprocess
import RemoveDB
import GetReflectance
import Processing

def getPhenotypeParasMatrix(reflectanceMatrix):
    PhenotypeParasMatrix = np.zeros((reflectanceMatrix.shape[0], 6, reflectanceMatrix.shape[2]))
    samples = reflectanceMatrix.shape[2]
    lines = reflectanceMatrix.shape[0]
    for i in range(samples*lines-1):
        row = i//samples
        col = i%samples
        PhenotypeParasMatrix[row,:,col] = Processing.getPhenotypeParas(reflectanceMatrix[row,:,col], 0)

    return PhenotypeParasMatrix


if __name__ == "__main__":
   # Level 0
    HSI_info = ReadData.Read()
    # Level 1
    HSI_info_L1, BG_Counter, proportion_1 = preprocess.getLevel1(HSI_info)
    # Level 2
    HSI_info_L2, SD_Counter, proportion_2 = RemoveDB.getLevel2(HSI_info_L1, BG_Counter,proportion_1)
    # Level 3
    reflectanceMatrix = GetReflectance.getReflectMatrix(HSI_info_L2, proportion_2) 

    # Draw pseudo-Color Images
    PhenotypeParas = getPhenotypeParasMatrix(reflectanceMatrix)
    draw_pseudoColorImgs(HSI_info, PhenotypeParas)


