import numpy as np
from PIL import Image
import re

global data
global samples, lines, channels

def ReadData():
    global samples, lines, channels
    data = []
    ### Read .hdr file to store the infomation
    with open("M:/m-CTP_DATA/2023.1.9/TeeSapling/wave.hdr") as hdr_file:
        for num, line in enumerate(hdr_file):
            data.append(line.split(" "))
    #print(data)
    wavelengths= []
    waveFlag = 0
    for row in range(len(data)):
        if data[row][0] == 'lines':
            #lines = int(re.findall("\d{4}",data[row][2])[0])
            lines = int(195840000/300/480)
            continue
        if data[row][0] == 'samples':
            samples = int(re.findall("\d{3}",data[row][2])[0])
            continue
        if data[row][0] == 'bands':
            channels = int(re.findall("\d{3}",data[row][2])[0])
            continue
        if data[row][0] == "wavelength" and data[row][1] != "units":
            str = data[row][2]
            str = str.replace("{","")
            str = str.replace("\n","")
            wavelengths.append(str)
            waveFlag = 1
            continue
        if waveFlag == 1:
            str = ','.join(data[row])
            str = str.replace("\n","")
            str = str.replace("}","")
            wavelengths.append(str)
    raw = ""
    wavelengths = raw.join(wavelengths)
    wavelengths = wavelengths.split(",")
    #print(wavelengths)
    

    ### Read .spe file
    # Load file and reshape
    #imgs = np.fromfile('M:/m-CTP_DATA/2023.1.9/Vegetables/TASK2023-01-06-10-52/Hyperspectral/2023-01-06-10-56-46.spe', dtype=np.int16).reshape(lines,channels,samples)
    imgs = np.fromfile('M:/m-CTP_DATA/2023.1.9/TeeSapling/2022-07-27-06-32.spe', dtype=np.int16).reshape(lines,channels,samples)
    imgs = imgs.astype(np.int16) # change the value range from (0, 4095) to (0, 255)
    print("Successfully read Spectral data")
    print("The height of imgs is",lines)
    print("The width of imgs is",samples)
    print("The length of bands",channels)
    #print(imgs.shape)
    return lines, channels, samples, imgs, wavelengths

def drawImg(HSI, filename):
    newimg = Image.new('RGB',(samples,lines))
    for i in range(lines):
        for j in range (samples):
            newimg.putpixel((j,i),((int(HSI[i][105][j]*256/4095)),(int(HSI[i][59][j]*256/4096)),(int(HSI[i][34][j]*256/4096))))
    fileName = "figures/" + filename + ".jpg"
    newimg.save(fileName)

# Plotting
if __name__ == "__main__":
    HSI_info = ReadData()
    HSI = HSI_info[3]
    lines = HSI_info[0]
    channels = HSI_info[1]
    samples = HSI_info[2]
    #print(imgs)
    
    #drawImg(HSI, "Original_Tee")












