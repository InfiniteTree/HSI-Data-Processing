import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split


class process:
    Reflect_Info = []
    hsPara = ""
    phenotypePara = ""
    ptsthsParaModel = ""
    ReflectMatrix = []

    ParaMatrix = []
    cur_proportion = 1

    lines = 0
    channels = 0
    samples = 0
    waveStart = 0

    ### Need to remap the band intelligently
    # map_num = ("wavelengh" - 400) / ((waveEnd - waveStart) / channels) 
    map_band = {"band430":16, "band531":62, "band550":70, "band570":80, "band635":110, "band670":126, "band680":131, "band705":143, "band750":164,"band780":178, "band800":188}
    
    def __init__(self, reflectInfo, hsParaType, phenotypeParaType, phenotypeParaModelType):
        self.Reflect_Info = reflectInfo
        self.hsPara = hsParaType
        self.phenotypePara = phenotypeParaType
        self.phenotypeParaModel = phenotypeParaModelType

        self.ReflectMatrix = self.Reflect_Info[3]
        self.lines = self.Reflect_Info[0]
        self.channels = self.Reflect_Info[1]
        self.samples = self.Reflect_Info[2]
        self.cur_proportion = self.Reflect_Info[5]
        self.waveStart = int(float(self.Reflect_Info[4][0]))
    
    def calImgSpecMean(self):
        return self.ReflectMatrix.mean(axis=(0,2)) / self.cur_proportion

    # Calculate the relative values the photosynthesis by the design formulas
    def calcHsParas(self):
        self.ReflectMatrix = np.where(self.ReflectMatrix < 0, 0, self.ReflectMatrix)
        #self.ReflectMatrix = np.where(self.ReflectMatrix > 1, 1, self.ReflectMatrix)

        match self.hsPara:
            case "NDVI":
                self.ParaMatrix = (self.ReflectMatrix[:,self.map_band["band800"],:] - self.ReflectMatrix[:,self.map_band["band680"],:]) / (self.ReflectMatrix[:,self.map_band["band800"],:] + self.ReflectMatrix[:,self.map_band["band680"],:])
            case "OSAVI":
                self.ParaMatrix = (1+0.16) * (self.ReflectMatrix[:,self.map_band["band800"],:] - self.ReflectMatrix[:,self.map_band["band670"],:]) / (self.ReflectMatrix[:,self.map_band["band800"],:] + self.ReflectMatrix[:,self.map_band["band670"],:]+ 0.16)
            case "PSSRa":
                self.ParaMatrix = self.ReflectMatrix[:,self.map_band["band800"],:] / self.ReflectMatrix[:,self.map_band["band680"],:]
            case "PSSRb":
                self.ParaMatrix = self.ReflectMatrix[:,self.map_band["band800"],:] / self.ReflectMatrix[:,self.map_band["band635"],:]
            case "PRI":
                self.ParaMatrix = (self.ReflectMatrix[:,self.map_band["band570"],:] - self.ReflectMatrix[:,self.map_band["band531"],:]) / (self.ReflectMatrix[:,self.map_band["band570"],:] + self.ReflectMatrix[:,self.map_band["band531"],:])
            case "MTVI2":
                self.ParaMatrix = 1.5 * (1.2 * (self.ReflectMatrix[:,self.map_band["band800"],:] - self.ReflectMatrix[:,self.map_band["band550"],:]) - 2.5 * (self.ReflectMatrix[:,self.map_band["band670"],:] - self.ReflectMatrix[:,self.map_band["band550"],:])) / math.sqrt(((2 * self.ReflectMatrix[:,self.map_band["band800"],:]+1)*2 - (6*self.ReflectMatrix[:,self.map_band["band800"],:]-5*math.sqrt(self.ReflectMatrix[:,self.map_band["band670"],:]))-0.5))
        
        if np.any(self.ParaMatrix > 10):
            print("Yes")
        self.ParaMatrix[self.ParaMatrix < -1] = -1
        self.ParaMatrix[self.ParaMatrix > 1] = 1

        #print(self.ParaMatrix)

    def draw_pseudoColorImg(self, flag):
        #print(self.ParaMatrix)
        fig, ax = plt.subplots(figsize=(6, 8))
        #print(self.hsPara)
        match self.hsPara:
            case "NDVI":
                im = ax.imshow(self.ParaMatrix, cmap='hot',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on NDVI", y=1.05)
            case "OSAVI":
                im = ax.imshow(self.ParaMatrix, cmap='viridis',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on OSAVI", y=1.05)
            case "PSSRa":
                im = ax.imshow(self.ParaMatrix, cmap='seismic',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on PSSRa", y=1.05)
            case "PSSRb":
                im = ax.imshow(self.ParaMatrix, cmap='coolwarm',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on PSSRb", y=1.05)
            case "PRI":
                im = ax.imshow(self.ParaMatrix, cmap='magma',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on PRI", y=1.05)
            case "MTVI2":
                im = ax.imshow(self.ParaMatrix, cmap='hot',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on MTVI2", y=1.05)

        cbar = fig.colorbar(im)
        if flag == "Save":
            plt.savefig("figures/test/process/" + self.hsPara + ".jpg")
        if flag == "View": # Consider to just load the figure here!!!!!!!!!
            plt.show()

    # Machine learning prediction
    def CalcPhenotypeParas(self):
        # Fault Value Detection
            # Get the remaining parameters by using the trained model to predict
            if self.phenotypeParaModel == "PLSR":
                data = pd.read_csv("model/LearningData/TrainData.csv")
                print("Dataset of Train Model loaded...")
                train_x = data.drop(['SPAD',"A1200", "N", "Ca", "Cb"],axis=1)
                #train_y = data[['SPAD',"A1200", "N", "Ca", "Cb"]].copy()
                train_y = data[[self.phenotypePara]]

                train_x = pd.DataFrame(train_x, dtype='float32')
                train_y = pd.DataFrame(train_y, dtype='float32')

                # pls_param_grid = {'n_components': list(range(10,20))}
                # Train the data
                pls_param_grid = {'n_components':[10]}  
                pls = GridSearchCV(PLSRegression(), param_grid=pls_param_grid,scoring='r2',cv=10)
                pls.fit(train_x, train_y)

                test_x = []

                for i in range(self.lines * self.samples):
                    test_x.append = self.ReflectMatrix[i/self.samples:,6:-16,:i%self.samples] # The data set of the train model only contains HS in parts of wavelength range
                test_x = pd.Series(test_x, dtype='float32')
                test_x = test_x.to_frame().T
                y_pre = pls.predict(test_x)


    # export file 1 to store the spectra data  
    def HyperspectraCurve(self, HSI_info, proportion):
        # Show the spectra curve
        wavelengths = HSI_info[4]
        lines = HSI_info[0]
        channels= HSI_info[1]
        samples = HSI_info[2]
        HSI = HSI_info[3]
        remainRow = []

        spec_mean = self.calImgSpecMean(HSI,proportion)
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
    def exportPhenotypeParas(self, Readfilename):

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
    def exportPhenotypeParas_eachPixel(self, HSI_info,reflectanceMatrix):
        lines = HSI_info[0]
        channels= HSI_info[1]
        samples = HSI_info[2]

        FirstRow = ["Loc","NDVI", "OSAVI", "PSSRa", "PSSRb", "RPI", "MTVI2"]

        avg_flag = 0
        PhenotypeParas= []

        # Export the results
        with open("Results/Phenotype_Paras_eachPixel.csv","w",newline='') as f:
            writer = csv.writer(f)
            writer.writerow(FirstRow)
            for i in range(samples*lines):
                row = i//samples
                col = i%samples
                PhenotypeParas = self.getPhenotypeParas(reflectanceMatrix[row,:,col], avg_flag)
                writer.writerow([(row,col)]+PhenotypeParas)
