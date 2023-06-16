import numpy as np
from PIL import Image
import re
import csv

def Read():
    HSI_info = ReadData("M:/m-CTP_DATA/2023.1.9/wheat/TASK2023-01-08-02-42/Hyperspectral/2023-01-08-06-01-59.hdr","M:/m-CTP_DATA/2023.1.9/wheat/TASK2023-01-08-02-42/Hyperspectral/2023-01-08-06-01-59.spe", 1)
    print("Successfully read the testPlant HSI")
    return HSI_info

def ReadRef():
    HSI_info = ReadData("M:/m-CTP_DATA/2023.1.9/TeeSapling/wave.hdr",'M:/m-CTP_DATA/2023.1.9/TeeSapling//2022-07-27-06-32.spe', 0)
    print("Successfully read the refBoard HSI")
    return HSI_info

def ReadData(hdrfileName,spefileName, flag):
    data = []
    ### Read .hdr file to store the infomation
    with open(hdrfileName) as hdr_file:
        for num, line in enumerate(hdr_file):
            data.append(line.split(" "))
    #print(data)
    wavelengths= []
    waveFlag = 0
    for row in range(len(data)):
        if data[row][0] == 'lines':
            #print(data[row])
            if (flag==0):
                lines = int(195840000/300/480)
            if (flag==1):
                try:
                    lines = int(re.findall("\d{4}",data[row][2])[0])
                except:
                    lines = int(re.findall("\d{3}",data[row][2])[0])
            continue
        if data[row][0] == 'samples':
            #print(data[row])
            samples = int(re.findall("\d{3}",data[row][2])[0])
            continue
        if data[row][0] == 'bands':
            #print(data[row])
            channels = int(re.findall("\d{3}",data[row][2])[0])
            continue
        if data[row][0] == "wavelength" and data[row][1] != "units":
            '''
            # Version for the previous CCD
            str = data[row][2]
            str = str.replace("{","")
            str = str.replace("\n","")
            wavelengths.append(str)
            '''
            waveFlag = 1
            continue
        if waveFlag == 1:
            data_row = data[row]
            str = ''.join(data[row])
            str = str.replace("\n","")
            str = str.replace("}","")
            wavelengths.append(str)
                
    #print(wavelengths)
    raw = ""
    wavelengths = raw.join(wavelengths)
    wavelengths = wavelengths.split(",")

    ### Read .spe file
    # Load file and reshape
    imgs = np.fromfile(spefileName, dtype=np.int16).reshape(lines,channels,samples)
    #imgs = np.fromfile('M:/m-CTP_DATA/2023.1.9/TeeSapling/2022-07-27-06-21.spe', dtype=np.int16).reshape(lines,channels,samples)
    imgs = imgs.astype(np.int16) # change the value range from (0, 4095) to (0, 255)

    return lines, channels, samples, imgs, wavelengths

def drawImg(HSI_info):
    lines = HSI_info[0]
    channels = HSI_info[1]
    samples = HSI_info[2]
    HSI = HSI_info[3]
    newimg = Image.new('RGB',(samples,lines))
    for i in range(lines):
        for j in range (samples):
            newimg.putpixel((j,i),((int(HSI[i][105][j]*256/4095)),(int(HSI[i][59][j]*256/4096)),(int(HSI[i][34][j]*256/4096))))
    return(newimg)
