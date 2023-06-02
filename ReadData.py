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
            if (flag==0):
                lines = int(195840000/300/480)
            if (flag==1):
                lines = int(re.findall("\d{4}",data[row][1])[0])
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
    imgs = np.fromfile(spefileName, dtype=np.int16).reshape(lines,channels,samples)
    #imgs = np.fromfile('M:/m-CTP_DATA/2023.1.9/TeeSapling/2022-07-27-06-21.spe', dtype=np.int16).reshape(lines,channels,samples)
    imgs = imgs.astype(np.int16) # change the value range from (0, 4095) to (0, 255)
    print("The height of imgs is",lines)
    print("The width of imgs is",samples)
    print("The length of bands",channels)
    #print("wavelength is")
    #print(wavelengths)
    #print(imgs.shape)
    return lines, channels, samples, imgs, wavelengths

def drawImg(HSI_info, filename):
    lines = HSI_info[0]
    channels = HSI_info[1]
    samples = HSI_info[2]
    HSI = HSI_info[3]
    newimg = Image.new('RGB',(samples,lines))
    for i in range(lines):
        for j in range (samples):
            newimg.putpixel((j,i),((int(HSI[i][105][j]*256/4095)),(int(HSI[i][59][j]*256/4096)),(int(HSI[i][34][j]*256/4096))))
    fileName = "figures/" + filename + ".jpg"
    newimg.save(fileName)

# Plotting
if __name__ == "__main__":
    HSI_info = Read()
    wavelengths = HSI_info[4]
    HSI_img = HSI_info[3]
    #print(imgs)
    print("------------Begin to write each row-------")
    height, bands, width = HSI_img.shape
    spectral_data_2d = np.reshape(HSI_img, (height * width, bands))
    bands_info = np.reshape(wavelengths, (1, bands))
    spectral_data_2d_with_bands = np.concatenate((bands_info, spectral_data_2d))
    csv_file_path = 'spectral_data.csv'
    np.savetxt(csv_file_path, spectral_data_2d_with_bands, delimiter=',')
    print("------------Finish writing each row-------")
    
    

