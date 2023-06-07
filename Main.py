
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap

import sys
import os
import cv2

from MainWindow import Ui_MainWindow
import ReadData

def saveRawFilePath():
    #ReadData.Read()
    #spe_file = 
    #hdr_file = 
    return

class Main(QMainWindow, Ui_MainWindow):
    settings = QtCore.QSettings("config.ini",
                            QtCore.QSettings.Format.IniFormat)
    
    # in Windows: C:\...\... while in linux C:/.../...
    rawFile_path = "" # The abs path of the raw spe file
    rawsFile_path = "" # The abs path of the raw spes file
    rawJpg_path = ""

    HSI_length = 1300 # Default length value
    HSI_width = 480 # Default width value
    HSI_wl = 300 # Default wavelength value

    def __init__(self, QMainWindow):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        # ----------------------------Tab1-----------------------------
    
        # Part 1. Show the raw data
        # Import the single raw file
        self.importRawBtn.clicked.connect(self.importRaw)
        self.importRawBtn.setGeometry(50, 50, 200, 30)

        # Import the multiples raw files
        self.importRawBtn.clicked.connect(self.importRaws)

        # Read and show the raw file
        self.rgbViewBtn.clicked.connect(self.viewRgbFile)
        self.showHsiInfoBtn.clicked.connect(self.showHsiInfo)

        # Save the raw rgb file
        self.rgbSaveBtn.clicked.connect(self.rgbSave)
        
      


    def importRaw(self):
        file_dialog = QFileDialog()
        # Select file/files according to the flag 
        selected_directory = file_dialog.getExistingDirectory(self, "Select Directory")
        rawFile = ""
        if selected_directory:
            file_names = os.listdir(selected_directory)
            print(file_names)
            spe_file = file_names[0]
            hdr_file = file_names[1]
            rawFile = file_names[1]
        

        if rawFile:
            self.rawFile_path = os.path.join(selected_directory, rawFile)
            #self.rawFile_path = os.path.abspath(rawFile)
            self.rawFile_path = self.rawFile_path.replace("\\","/")
            self.rawPathlineEdit.setText(self.rawFile_path)
            self.rawJpg_path = self.rawFile_path
            '''
            hdr_file_path = 
            spe_file_path = 
            HSI_info = ReadData.ReadData(hdr_file_path,spe_file_path, 1)
            self.HSI_length = HSI_info[0]
            self.HSI_wl = HSI_info[1]
            self.HSI_width = HSI_info[2]
            '''
        else:
            return ""

    def importRaws(self):        
        return  

    
    def viewRgbFile(self):
        '''
        if self.rawFile_path:
            self.rawImgQlabel.setPixmap(QtGui.QPixmap(self.rawFile_path))
        '''
        image = cv2.imread(self.rawJpg_path)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", self.HSI_width , self.HSI_length) 

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showHsiInfo(self):
        self.lenShowBtn.setText(str(self.HSI_length)+" pix")
        self.widthShowBtn.setText(str(self.HSI_width)+" pix")
        self.wlShowBtn.setText(str(self.HSI_wl))
    

    # ----------------------------Tab2-----------------------------


    # ----------------------------Tab3-----------------------------


    # ----------------------------Tab4-----------------------------

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    md = Main(QMainWindow)
    md.show()
    sys.exit(app.exec_())
            



