import numpy as np
import matplotlib.pyplot as plt

import ReadData
import RemoveBG
import RemoveSD
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

def draw_pseudoColorImg(HSI_info, PhenotypeParas,filename,idx):
    lines = HSI_info[0]
    PhenotypeParasNum = 6
    samples = HSI_info[2]
    HSI = HSI_info[3]
    
    slice_2d = PhenotypeParas[:, idx, :]

    fig, ax = plt.subplots(figsize=(6, 8))
    if idx == 0:
        im = ax.imshow(slice_2d, cmap='hot',interpolation='nearest')
        ax.set_title("Pseudo_Color Map of the Relative Values on NDVI", y=1.05)
    elif idx == 1:
        im = ax.imshow(slice_2d, cmap='viridis',interpolation='nearest')
        ax.set_title("Pseudo_Color Map of the Relative Values on OSAVI", y=1.05)
    elif idx == 2:
        im = ax.imshow(slice_2d, cmap='seismic',interpolation='nearest')
        ax.set_title("Pseudo_Color Map of the Relative Values on PSSRa", y=1.05)
    elif idx == 3:
        im = ax.imshow(slice_2d, cmap='coolwarm',interpolation='nearest')
        ax.set_title("Pseudo_Color Map of the Relative Values on PSSRb", y=1.05)
    elif idx == 4:
        im = ax.imshow(slice_2d, cmap='magma',interpolation='nearest')
        ax.set_title("Pseudo_Color Map of the Relative Values on PRI", y=1.05)
    elif idx == 5:
        im = ax.imshow(slice_2d, cmap='hot',interpolation='nearest')
        ax.set_title("Pseudo_Color Map of the Relative Values on MTVI2", y=1.05)

    cbar = fig.colorbar(im)
    plt.savefig("figures/pseudoColorImg/"+filename)
    
    return

def draw_pseudoColorImgs(HSI_info, PhenotypeParas):
    draw_pseudoColorImg(HSI_info, PhenotypeParas, "pseudoColorImg_NDVI.png",0)
    draw_pseudoColorImg(HSI_info, PhenotypeParas, "pseudoColorImg_OSAVI.png",1)
    draw_pseudoColorImg(HSI_info, PhenotypeParas, "pseudoColorImg_PSSRa.png",2)
    draw_pseudoColorImg(HSI_info, PhenotypeParas, "pseudoColorImg_PSSRb.png",3)
    draw_pseudoColorImg(HSI_info, PhenotypeParas, "pseudoColorImg_PRI.png",4)
    draw_pseudoColorImg(HSI_info, PhenotypeParas, "pseudoColorImg_MTVI2.png",5)

if __name__ == "__main__":
   # Level 0
    HSI_info = ReadData.Read()
    # Level 1
    HSI_info_L1, BG_Counter, proportion_1 = RemoveBG.getLevel1(HSI_info)
    # Level 2
    HSI_info_L2, SD_Counter, proportion_2 = RemoveSD.getLevel2(HSI_info_L1, BG_Counter,proportion_1)
    # Level 3
    reflectanceMatrix = GetReflectance.getReflectMatrix(HSI_info_L2, proportion_2) 

    # Draw pseudo-Color Images
    PhenotypeParas = getPhenotypeParasMatrix(reflectanceMatrix)
    draw_pseudoColorImgs(HSI_info, PhenotypeParas)


