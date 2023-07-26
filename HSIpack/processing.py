import numpy as np
import warnings
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split

class Process:
    Reflect_Info = []
    hsPara = ""
    phenotypePara = ""
    ptsthsParaModel = ""

    cur_proportion = 1
    lines = 0
    channels = 0
    samples = 0
    waveStart = 0

    ReflectMatrix = [] # Used to store the reflectance matricx
    ParaMatrix = [] # Used to store the parameter matricx: hyperspectra para matrix or Phenotype matrix
    ### masked_array = [] # Used to store the parameter matricx that will be shown in color map
    # used to get the range of the cropped image
    min_row = 0
    max_row = lines
    min_col = 0
    max_col = samples
    min_row_f = 0
    max_row_f = lines
    min_col_f = 0
    max_col_f = lines

    PtMin = 0
    PtMax = 0

    count = 0 # temp value

    FirstRow = []  # The first row to write
    pls = None # the PLSR model

    filename = ""

    ### Need to remap the band intelligently
    # map_num = ("wavelengh" - 400) / ((waveEnd - waveStart) / channels) 
    map_band = {"band430":16, "band445":22, "band500":47, "band510":51,"band531":62, "band550":70, "band570":80, "band635":110, "band670":126, "band680":131, "band700":139, "band705":143, "band750":164,"band780":178, "band800":188, "band900":235,"band970":268}
    
    def __init__(self, reflectInfo, hsParaType, phenotypeParaType, phenotypeParaModelType, plant_mask, filename, ParaMatrixValMin, ParaMatrixValMax):
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
        self.plant_mask = plant_mask
        self.ParaMatrix = np.zeros((self.lines, self.samples))
        self.filename = filename
        self.ValueMin = ParaMatrixValMin 
        self.ValueMax = ParaMatrixValMax

        # Make a bounding box to show the plant graph only
        rows, cols = np.nonzero(~self.plant_mask)
        self.min_row, self.max_row = np.min(rows), np.max(rows)
        self.min_col, self.max_col = np.min(cols), np.max(cols)
        # Remain some blank for the outside of the graph
        self.min_col_f = self.min_col - int(1/8 *(self.max_col - self.min_col))
        self.max_col_f = self.max_col + int(1/8 *(self.max_col - self.min_col))
        self.min_row_f = self.min_row - int(1/8 *(self.max_row - self.min_row))
        self.max_row_f = self.max_row + int(1/8 *(self.max_row - self.min_row))
        # False dectection for the plot bound
        if self.min_col_f < 0:
            self.min_col_f =  0
        if self.min_row_f < 0:
            self.min_row_f =  0
        if self.max_col_f > self.samples:
            self.max_col_f = self.samples
        if self.max_row_f > self.lines:
            self.max_row_f = self.lines

        # Train the model
        data = pd.read_csv("model/LearningData/TrainData.csv")
        #print("Dataset of Train Model loaded...")
        train_x = data.drop(['SPAD',"A1200", "N", "Ca", "Cb"],axis=1)
        #train_y = data[['SPAD',"A1200", "N", "Ca", "Cb"]].copy()
        train_y = data[[self.phenotypePara]]
        train_x = pd.DataFrame(train_x, dtype='float32')
        train_y = pd.DataFrame(train_y, dtype='float32')
        # pls_param_grid = {'n_components': list(range(10,20))}
        # Train the data
        pls_param_grid = {'n_components':[10]}  
        warnings.filterwarnings('ignore', category=UserWarning)
        self.pls = GridSearchCV(PLSRegression(), param_grid=pls_param_grid,scoring='r2',cv=10)
        self.pls.fit(train_x, train_y)
    
    # Calculate the relative values the photosynthesis by the design formulas
    def calcHsParas(self):
        #print(self.hsPara)
        match self.hsPara:
            case "NDVI":
                numerator =  self.ReflectMatrix[:,self.map_band["band800"],:] - self.ReflectMatrix[:,self.map_band["band680"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band800"],:] + self.ReflectMatrix[:,self.map_band["band680"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 0  # the denominator is zero, set it as 0
                # All range in [-1, 1], while plant in [0.2, 0.8]
                self.ParaMatrix[self.ParaMatrix < 0] = 0
                self.ParaMatrix[self.ParaMatrix > 1] = 0
                
            case "OSAVI":
                numerator =  (1+0.16) * (self.ReflectMatrix[:,self.map_band["band800"],:] - self.ReflectMatrix[:,self.map_band["band670"],:])
                denominator = self.ReflectMatrix[:,self.map_band["band800"],:] + self.ReflectMatrix[:,self.map_band["band670"],:]+ 0.16
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 0  # the denominator is zero, set it as 0
                
            case "PRI":
                numerator =  self.ReflectMatrix[:,self.map_band["band531"],:] - self.ReflectMatrix[:,self.map_band["band570"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band531"],:] + self.ReflectMatrix[:,self.map_band["band570"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = -0.2  # the denominator is zero, set it as 0
                # All range in [-1, 1], while plant in [-0.2, 0.2]
                self.ParaMatrix[self.ParaMatrix < -0.2] = -0.2
                self.ParaMatrix[self.ParaMatrix > 0.2] = 0.2

            case "MTVI2":
                numerator =  1.5 * (1.2 * (self.ReflectMatrix[:,self.map_band["band800"],:] - self.ReflectMatrix[:,self.map_band["band550"],:]) - 2.5 * (self.ReflectMatrix[:,self.map_band["band670"],:] - self.ReflectMatrix[:,self.map_band["band550"],:]))
                denominator = np.sqrt(((2 * self.ReflectMatrix[:,self.map_band["band800"],:]+1)*2 - (6*self.ReflectMatrix[:,self.map_band["band800"],:]-5*np.sqrt(self.ReflectMatrix[:,self.map_band["band670"],:]))-0.5))
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 0  # the denominator is zero, set it as 0
            
            case "SR":
                numerator =  self.ReflectMatrix[:,self.map_band["band800"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band680"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 2  # the denominator is zero, set it as 0
                self.ParaMatrix[self.ParaMatrix < 2] = 2
                self.ParaMatrix[self.ParaMatrix > 8] = 8
            
            case "DVI":
                var_1 =  self.ReflectMatrix[:,self.map_band["band800"],:]
                var_2 = self.ReflectMatrix[:,self.map_band["band680"],:]
                self.ParaMatrix = var_1 - var_2
                
            case "SIPI":
                numerator =  self.ReflectMatrix[:,self.map_band["band800"],:] - self.ReflectMatrix[:,self.map_band["band445"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band800"],:] + self.ReflectMatrix[:,self.map_band["band680"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 0  # the denominator is zero, set it as 0
                self.ParaMatrix[self.ParaMatrix < 0] = 0
                self.ParaMatrix[self.ParaMatrix > 2] = 2

            case "PSRI":
                numerator =  self.ReflectMatrix[:,self.map_band["band680"],:] - self.ReflectMatrix[:,self.map_band["band500"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band750"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = -1  # the denominator is zero, set it as 0
                self.ParaMatrix[self.ParaMatrix < -1] = -1
                self.ParaMatrix[self.ParaMatrix > 1] = 1

            case "CRI1":
                denominator_1 = self.ReflectMatrix[:,self.map_band["band510"],:]
                denominator_2 = self.ReflectMatrix[:,self.map_band["band550"],:]
                denominator_1[denominator_1 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                denominator_2[denominator_2 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = 1/denominator_1 - 1/denominator_2
                self.ParaMatrix[self.ParaMatrix < 0] = 0 
                self.ParaMatrix[self.ParaMatrix > 15] = 0
            
            case "CRI2":
                denominator_1 = self.ReflectMatrix[:,self.map_band["band510"],:]
                denominator_2 = self.ReflectMatrix[:,self.map_band["band700"],:]
                denominator_1[denominator_1 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                denominator_2[denominator_2 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = 1/denominator_1 - 1/denominator_2
                self.ParaMatrix[self.ParaMatrix < 0] = 0 
                self.ParaMatrix[self.ParaMatrix > 15] = 15


            case "ARI1":
                denominator_1 = self.ReflectMatrix[:,self.map_band["band550"],:]
                denominator_2 = self.ReflectMatrix[:,self.map_band["band700"],:]
                denominator_1[denominator_1 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                denominator_2[denominator_2 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = 1/denominator_1 - 1/denominator_2
                self.ParaMatrix[self.ParaMatrix < 0] = 0 
                self.ParaMatrix[self.ParaMatrix > 0.2] = 0.2

            case "ARI2":
                denominator_1 = self.ReflectMatrix[:,self.map_band["band550"],:]
                denominator_2 = self.ReflectMatrix[:,self.map_band["band700"],:]
                denominator_1[denominator_1 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                denominator_2[denominator_2 <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = self.ReflectMatrix[:,self.map_band["band800"],:] * (1/denominator_1 - 1/denominator_2)
                self.ParaMatrix[self.ParaMatrix < 0] = 0 
                self.ParaMatrix[self.ParaMatrix > 0.2] = 0.2

            case "WBI":
                numerator =  self.ReflectMatrix[:,self.map_band["band900"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band970"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 0  # the denominator is zero, set it as 0
                # plant in [0.8, 1.2]
                self.ParaMatrix[self.ParaMatrix <= 0.8] = 0.8
                self.ParaMatrix[self.ParaMatrix >= 1.2] = 1.2

            # Bugs remain in PSSRa and PSSRb, which is caused by the raw data of band680/band635 
            # (reflectance is approximately zero and make the calculation result too large)
            case "PSSRa":
                numerator =  self.ReflectMatrix[:,self.map_band["band800"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band680"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 2  # the denominator is zero, set it as 0
                self.ParaMatrix[self.ParaMatrix < 2] = 2
                self.ParaMatrix[self.ParaMatrix > 8] = 8
            
            case "PSSRb":
                numerator =  self.ReflectMatrix[:,self.map_band["band800"],:]
                denominator = self.ReflectMatrix[:,self.map_band["band635"],:]
                denominator[denominator <= 0] = 1  # Avoid the denominator is zero, set it as 1 
                self.ParaMatrix = numerator / denominator
                self.ParaMatrix[denominator == 0] = 2  # the denominator is zero, set it as 0
                self.ParaMatrix[self.ParaMatrix < 2] = 2
                self.ParaMatrix[self.ParaMatrix > 8] = 8

            # Self defined formular
            case "user-defined":
                print("ok")
        '''
        if np.any(self.ParaMatrix > 10):
            print("Yes >10")
        
        if np.any(self.ParaMatrix < -10):
            print("Yes <10")
        '''

        #self.ParaMatrix[self.ParaMatrix < -1] = -1
        #self.ParaMatrix[self.ParaMatrix > 1] = 1

        #print(self.ParaMatrix)

    def draw_pseudoColorImg(self, op_flag, para_flag, colorMapType):
        #print(self.ParaMatrix)
        fig, ax = plt.subplots(figsize=(6, 8))
        # get the cropped image
        masked_array = np.ma.array(self.ParaMatrix, mask=self.plant_mask)
        cropped_image = masked_array[self.min_row_f:self.max_row_f+1, self.min_col_f:self.max_col_f+1]
        #print(self.hsPara)
        match para_flag:
            case 1:
                match self.hsPara:
                    case "NDVI":
                        im = ax.imshow(cropped_image, cmap='gray',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on NDVI", y=1.05)
                    case "OSAVI":
                        im = ax.imshow(cropped_image, cmap='viridis',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on OSAVI", y=1.05)
                    case "PSSRa":
                        im = ax.imshow(cropped_image, cmap='spring',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on PSSRa", y=1.05)
                    case "PSSRb":
                        im = ax.imshow(cropped_image, cmap='summer',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on PSSRb", y=1.05)
                    case "PRI":
                        im = ax.imshow(cropped_image, cmap='magma',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on PRI", y=1.05)
                    case "MTVI2":
                        im = ax.imshow(cropped_image, cmap='hot',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on MTVI2", y=1.05)
                    case "SR":
                        im = ax.imshow(cropped_image, cmap='gray',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on SR", y=1.05)
                    case "DVI":
                        im = ax.imshow(cropped_image, cmap='viridis',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on Ca", y=1.05)
                    case "SIPI":
                        im = ax.imshow(cropped_image, cmap='spring',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on SIPI", y=1.05)
                    case "PSRI":
                        im = ax.imshow(cropped_image, cmap='summer',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on PSRI", y=1.05)
                    case "CRI1":
                        im = ax.imshow(cropped_image, cmap='magma',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on CRI1", y=1.05)
                    case "CRI2":
                        im = ax.imshow(cropped_image, cmap='hot',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on CRI2", y=1.05)
                    case "ARI1":
                        im = ax.imshow(cropped_image, cmap='summer',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on ARI1", y=1.05)
                    case "ARI2":
                        im = ax.imshow(cropped_image, cmap='magma',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on ARI2", y=1.05)
                    case "WBI":
                        im = ax.imshow(cropped_image, cmap='hot',interpolation='nearest')
                        ax.set_title("Pseudo_Color Map of the Relative Values on WBI", y=1.05)

                cbar = fig.colorbar(im)
                if op_flag == "Save": 
                    plt.savefig("Outputs/figures/" + self.filename + "/process/" + self.hsPara + ".jpg")
                    plt.close()

                if op_flag == "View": # Consider to just load the figure here!!!!!!!!!
                    plt.show()
            
            case 2:
                im = ax.imshow(cropped_image, cmap=colorMapType,interpolation='nearest', vmin=self.ValueMin, vmax=self.ValueMax) # bug! need to be reset here
                ax.set_title("Pseudo_Color Map of the Relative Values on " + self.phenotypePara, y=1.05)
                cbar = fig.colorbar(im)
                if op_flag == "Save": 
                    plt.savefig("Outputs/figures/" + self.filename + "/process/" + self.phenotypePara + ".jpg")
                    plt.close()

                if op_flag == "View": # Consider to just load the figure here!!!!!!!!!
                    plt.show()

    # Machine learning prediction 
    def CalcPhenotypeParas(self, index):
        # Fault Value Detection
            # Get the remaining parameters by using the trained model to predict
            row_t = index//self.samples
            col_t = index%self.samples
            if self.phenotypeParaModel == "PLSR" and row_t > self.min_row and row_t < self.max_row and col_t > self.min_col and col_t < self.max_col:
                # test_x stores the raw data for one pixel; y_pre stores the dealt results for all pixels
                if self.plant_mask[row_t][col_t] == False: # Pixs that belongs to the plant
                    test_x = self.ReflectMatrix[row_t,6:-16,col_t] # The data set of the train model only contains HS in parts of wavelength range
                    test_x = pd.Series(test_x, dtype='float32')
                    test_x = test_x.to_frame().T
                    self.ParaMatrix[row_t, col_t] = self.pls.predict(test_x)
                else:
                    self.ParaMatrix[row_t, col_t] = 0
                
                # handle fault value; any reason that produce them???
                if self.ParaMatrix[row_t, col_t] < self.PtMin or self.ParaMatrix[row_t, col_t] > self.PtMax:
                    self.count += 1
                    self.ParaMatrix[row_t, col_t] = 0

    # export file 1 to store the spectra data  
    def HyperspectraCurve(self, HSI_info, proportion):
        # Show the spectra curve
        wavelengths = HSI_info[4]
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
    def exportHsParas(self, filename,idx):
        FirstRow = ["photoIdx","fileName", "NDVI","OSAVI", "PSSRa","PSSRb", "PRI","MTVI2","SR", "DVI", "SIPI", "PSRI", "CRI1", "CRI2", "ARI1", "ARI2", "WBI"]
        # if self.hsPara not in self.FirstRow:
        # Export the results
        # csv_folder = os.path.dirname(os.path.abspath(filename))
        writeFlag = "w"
        if idx > 1:
            writeFlag = "a"
        with open(filename,writeFlag,newline='') as f:
            writer = csv.writer(f)
            if idx == 1:
                writer.writerow(FirstRow)
            meanHsPara = [] # initialize
            meanHsPara.append(idx)
            meanHsPara.append(self.filename[:-4])
            for j in range(len(FirstRow)-2):
                self.hsPara = FirstRow[j+1]
                self.calcHsParas()
                non_zero_matrix = self.ParaMatrix[self.ParaMatrix != 0]
                meanHsPara.append(np.mean(non_zero_matrix)) # bugs remain here
            writer.writerow(meanHsPara)
        # subprocess.run(['start', '', csv_folder], shell=True)

    def exportPhenotypeParas(self, filename, idx):
        FirstRow = ["Idx","fileName","SPAD", "A1200", "N", "Ca", "Cb"]
        # if self.hsPara not in self.FirstRow:
        # Export the results
        # csv_folder = os.path.dirname(os.path.abspath(filename))
        writeFlag = "w"
        if idx > 1:
            writeFlag = "a"
        with open(filename, writeFlag, newline='') as f:
            writer = csv.writer(f)
            if idx == 1:
                writer.writerow(FirstRow)
            dataRow = [] # initialize
            dataRow.append(idx)
            dataRow.append(self.filename[:-4])
            for j in range(len(FirstRow)-2):
                if self.phenotypeParaModel == "PLSR":
                    data = pd.read_csv("model/LearningData/TrainData.csv")
                    #print("Dataset of Train Model loaded...")
                    train_x = data.drop(['SPAD',"A1200", "N", "Ca", "Cb"],axis=1)
                    #train_y = data[['SPAD',"A1200", "N", "Ca", "Cb"]].copy()
                    match j:
                        case 0:
                            self.phenotypePara = "SPAD"
                        case 1:
                            self.phenotypePara = "A1200"
                        case 2:
                            self.phenotypePara = "N"
                        case 3:
                            self.phenotypePara = "Ca"
                        case 4:
                            self.phenotypePara = "Cb"
                    train_y = data[[self.phenotypePara]]
                    train_x = pd.DataFrame(train_x, dtype='float32')
                    train_y = pd.DataFrame(train_y, dtype='float32')
                    # pls_param_grid = {'n_components': list(range(10,20))}
                    # Train the data
                    pls_param_grid = {'n_components':[10]}  
                    warnings.filterwarnings('ignore', category=UserWarning)
                    pls = GridSearchCV(PLSRegression(), param_grid=pls_param_grid,scoring='r2',cv=10)
                    pls.fit(train_x, train_y)

                    # test_x stores the raw data for one pixel; y_pre stores the dealt results for all pixels
                    expanded_mask = np.expand_dims(self.plant_mask, axis=1)
                    expanded_mask = np.repeat(expanded_mask, self.channels, axis=1)
                    masked_array = np.ma.array(self.ReflectMatrix, mask=expanded_mask)

                    test_x = np.mean(masked_array[:, 6:-16, :],axis=(0,2)) # The data set of the train model only contains HS in parts of wavelength range

                    test_x = pd.Series(test_x, dtype='float32')
                    test_x = test_x.to_frame().T
                    predict_y = pls.predict(test_x)
                    dataRow.append(predict_y[0][0])
            writer.writerow(dataRow)
        
        # subprocess.run(['start', '', csv_folder], shell=True)



