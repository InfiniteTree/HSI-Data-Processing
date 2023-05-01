import numpy as np
from PIL import Image
import re
import matplotlib.pyplot as plt
global data
global samples, lines, channels

def ReadData():
    global samples, lines, channels
    data = []
    ### Read .hdr file to store the infomation
    with open("M:/m-CTP_DATA/2023.1.9/Vegetables/TASK2023-01-06-10-52/Hyperspectral/2023-01-06-10-56-46.hdr") as hdr_file:
        for num, line in enumerate(hdr_file):
            data.append(line.split(" "))

    for foo in data:
        if foo[0] == 'lines':
            lines = int(re.findall("\d{4}",foo[1])[0])
        if foo[0] == 'samples':
            samples = int(re.findall("\d{3}",foo[2])[0])
        if foo[0] == 'bands':
            channels = int(re.findall("\d{3}",foo[2])[0])
    ### Read .spe file
    # Load file and reshape
    imgs = np.fromfile('M:/m-CTP_DATA/2023.1.9/Vegetables/TASK2023-01-06-10-52/Hyperspectral/2023-01-06-10-56-46.spe', dtype=np.int16).reshape(lines,channels,samples)
    imgs = imgs.astype(np.int16) # change the value range from (0, 4095) to (0, 255)
    print("Successfully read Spectral data")
    print("The height of imgs is",lines)
    print("The width of imgs is",samples)
    print("The length of bands",channels)
    #print(imgs.shape)
    return samples, lines, channels, imgs


# Plotting

if __name__ == "__main__":
    imgs = ReadData()
    #print(imgs)
    newimg = Image.new('RGB',(samples,lines))

    for i in range(lines):
        for j in range (samples):
            newimg.putpixel((j,i),((int(imgs[i][105][j])),(int(imgs[i][59][j])),(int(imgs[i][34][j]))))
    newimg.save("2023-01-06-10-56-46.jpg")











