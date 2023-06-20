
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsScene, QVBoxLayout, QGraphicsPixmapItem, QGraphicsView, QGraphicsRectItem, QGraphicsPathItem, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPainterPath
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from MainWindow import Ui_MainWindow
import ReadData as rd
import Preprocess as pre
import GetReflectance as gr
import Processing as pro

class Main(QMainWindow, Ui_MainWindow):
    settings = QtCore.QSettings("config.ini",
                            QtCore.QSettings.Format.IniFormat)
    
    ######----------------------------------------------------------------------------------------------------######
    #####----------------------------------Parameters definition start here------------------------------------#####
    ####--------------------------------------------------------------------------------------------------------####
    # ------------------------------------Tab1------------------------------------
    impFileNum = 0
    # in Windows: C:\...\... while in linux C:/.../...
    rawSpeFile_path = "" # The abs path of the raw spe file
    rawHdrFile_path = "" # The abs path of the raw hdr file
    rawsSpeFile_path = "" # The abs path of the raw spe files
    rawsHdrFile_path = "" # The abs path of the raw hdr files
    BRFSpeFile_path = "" # The abs path of the reference board spe file
    BRFHdrFile_path = "" # The abs path of the reference board hdr file


    # Data recording for selection rectangular
    scene = None
    selecting = False
    selection_rect = None
    selection_start = None
    selection_end = None

    BRF3_pos_range = [] # [BRF3%] [[3_x0,3_y0],[3_x1,3_y1]]
    BRF30_pos_range = [] # [BRF30%] [[30_x0,30_y0],[30_x1,30_y1]]

    # Data for single Hyperspectra image
    raw_HSI_info = []
    HSI_lines = 0 # Default length value
    HSI_samples = 0 # Default width value
    HSI_channels = 300 # Default wavelength value
    HSI = [[[]]] # 3-D HSI img
    HSI_wavelengths = [] # ranging from apporximately 400nm to 1000nm

    # Data for reference board image
    BRF_HSI_info = []

    # rbg Image generated by the three bands of HSI
    rgbImg = []

    # ------------------------------------Tab2------------------------------------
    # NDVI_matrix
    NDVI = []

    # Threshold value by set at the Tab2
    NDVI_TH = 0 # Threshold value of NDVI to seperate the plant from the background
    ampl_LowTH = 100  # Threshold value of amplititude of the hyperspectra to eliminate
    ampl_HighTH = 4000  # Threshold value of amplititude of the hyperspectra to eliminate

    BRFfile_paths = [] # ["3%BRF_filename", "30%BRF_filename"]

    # The proportion is initially set as 1
    cur_proportion = 1

    # class reflect
    reflect = None
    k = []
    b = []
    
    l1_rgbimg_path = "" # level 1 img rgb file path
    l2_rgbimg_path = "" # level 2 img rgb file path

    # ------------------------------------Tab3------------------------------------
    Hs_Para = ""
    Ptsths_Para = ""
    Ptsths_Para_Model = ""

    pro_data = None

    ###-------------------------------------------The End line---------------------------------------------------###


    ######----------------------------------------------------------------------------------------------------######
    #####-------------------------------------_init_ Function start here---------------------------------------#####
    ####--------------------------------------------------------------------------------------------------------####
    def __init__(self, QMainWindow):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.previousPage = None

        # ------------------------------------Tab1------------------------------------
        # Part 1. Raw Data Processing
        # Import the BRF HSI files
        self.impBRFImgBtn.clicked.connect(self.importBRFImg)
        
        # Mouse box selection for 3% board
        self.selectBox3Btn.clicked.connect(lambda: self.selectBox("3"))
        # Mouse box selection for 30% board
        self.selectBox30Btn.clicked.connect(lambda: self.selectBox("30"))

        # Get k and b of the reflectance equation
        self.importRftCaliFileBtn.clicked.connect(self.importRftCaliFile)
        self.RefCaliBtn.clicked.connect(self.RefCali)
        
        
        # Import the single raw HSI file
        self.impRawBtn.clicked.connect(self.importRaw)
        self.impRawBtn.setGeometry(50, 50, 200, 30)

        # Import the multiples raw HSI files
        self.impRawsBtn.clicked.connect(self.importRaws)


        # Read the raw file
        self.rgbGeneBtn.clicked.connect(lambda:self.getRgb("Gene"))
        # show the raw file
        self.rgbViewBtn.clicked.connect(lambda:self.getRgb("View"))
        # Save the raw rgb file
        self.rgbSaveBtn.clicked.connect(lambda:self.getRgb("Save"))

        # Read the raw BRF file
        self.BRFRawGeneBtn.clicked.connect(lambda:self.getBRFRgb("Gene"))
        # show the raw BRF file
        self.BRFRawViewBtn.clicked.connect(lambda:self.getBRFRgb("View"))
        # Save the raw BRF file
        self.BRFRawSaveBtn.clicked.connect(lambda:self.getBRFRgb("Save"))

        # Show the hsi information
        self.showHsiInfoBtn.clicked.connect(self.showHsiInfo)
        
        # Draw the hyperspectra curve
        self.HSCurveBtn.clicked.connect(self.HSCurveView)

        # ------------------------------------Tab2------------------------------------
        # Part 2. Data Pre-processing
        self.NDVI_TH = float(self.bgParaDb.currentText())
        self.ampl_LowTH = int(self.amplLowThDb.currentText())
        self.ampl_HighTH = int(self.amplHighThDb.currentText())
        # Handle Selection Changed
        self.bgParaDb.currentIndexChanged.connect(lambda: self.getPreProcessPara(1))
        self.amplLowThDb.currentIndexChanged.connect(lambda: self.getPreProcessPara(2))
        self.amplHighThDb.currentIndexChanged.connect(lambda: self.getPreProcessPara(3))

        # Level 1-2 pre-processing
        self.RmBgGeneBtn.clicked.connect(lambda: self.RmBg("Gene"))
        self.RmBgViewBtn.clicked.connect(lambda: self.RmBg("View"))
        self.RmBgSaveBtn.clicked.connect(lambda: self.RmBg("Save"))

        self.RmDbGeneBtn.clicked.connect(lambda: self.RmDb("Gene"))
        self.RmDbViewBtn.clicked.connect(lambda: self.RmDb("View"))
        self.RmDbSaveBtn.clicked.connect(lambda: self.RmDb("Save"))

        self.RefGeneBtn.clicked.connect(lambda: self.getReflect("Gene"))
        self.RefViewBtn.clicked.connect(lambda: self.getReflect("View"))
        self.RefSaveBtn.clicked.connect(lambda: self.getReflect("Save"))

        # Draw the reflectance curve
        self.RFCurveBtn.clicked.connect(self.RFCurveView)

        # ------------------------------------Tab3------------------------------------
        # Get the current text in the drab bar
        self.HS_Para = self.hsParaDb.currentText()
        self.Ptsths_Para = self.ptsthsParaDb.currentText()
        self.Ptsths_Para_Model = self.ptsthsParaModelDb.currentText()
        # Get the changed text in the drab bar
        self.hsParaDb.currentIndexChanged.connect(lambda: self.getProcessPara(1))
        self.ptsthsParaDb.currentIndexChanged.connect(lambda: self.getProcessPara(2))
        self.ptsthsParaModelDb.currentIndexChanged.connect(lambda: self.getProcessPara(3))

        self.hsParaGeneBtn.clicked.connect(lambda: self.getHsPara("Gene"))
        self.hsParaSaveBtn.clicked.connect(lambda: self.getHsPara("Save"))
        self.hsParaViewBtn.clicked.connect(lambda: self.getHsPara("View"))

        self.ptsthsGeneBtn.clicked.connect(lambda: self.getPtsthsPara("Gene"))
        self.ptsthsSaveBtn.clicked.connect(lambda: self.getPtsthsPara("Save"))
        self.ptsthsViewBtn.clicked.connect(lambda: self.getPtsthsPara("View"))

    ######----------------------------------------------------------------------------------------------------######
    #####-------------------------------------Helper Function start here---------------------------------------#####
    ####--------------------------------------------------------------------------------------------------------####
    # -------------------------------------Tab1-------------------------------------
    def importRaw(self):
        file_dialog = QFileDialog()
        selected_file, _ = file_dialog.getOpenFileName(QMainWindow(), '选择文件', '', '.spe(*.spe*)')
        if selected_file:
            self.rawSpeFile_path = selected_file
            self.rawSpeFile_path = self.rawSpeFile_path.replace("\\","/")
            self.rawHSIPathlineEdit.setText(self.rawSpeFile_path)
            
            self.rawHdrFile_path = self.rawSpeFile_path.replace(".spe",".hdr")
            self.impFileNum += 1

    def importRaws(self):  
        file_dialog = QFileDialog()
        selected_directory = file_dialog.getExistingDirectory(self, "选择文件夹")
        if selected_directory:
            file_names = os.listdir(selected_directory)
            #print(file_names)
        

    def importBRFImg(self):
        selected_file, _ = QFileDialog.getOpenFileName(QMainWindow(), '选择文件', '', '.spe(*.spe*)')
        if selected_file:
            self.BRFSpeFile_path = selected_file
            self.BRFSpeFile_path = self.BRFSpeFile_path.replace("\\","/")
            self.BRFPathlineEdit.setText(self.BRFSpeFile_path)
            
            self.BRFHdrFile_path = self.BRFSpeFile_path.replace(".spe",".hdr")

    def RefCali(self):
        self.reflect = gr.Reflectance(self.HSI_info, self.cur_proportion, [self.BRF3_pos_range, self.BRF30_pos_range], self.BRFfile_paths, [], [])
        # Get the k and b
        self.k, self.b = self.reflect.getReflectEquation()
        # Unlock the view and Save function
        QtWidgets.QMessageBox.about(self, "", "反射板校准已就绪")


    def getRgb(self, function):
        match function:
            case "Gene":
                self.HSI_info = rd.ReadData(self.rawHdrFile_path,self.rawSpeFile_path, 1)
                self.HSI_lines = self.HSI_info[0]
                self.HSI_channels = self.HSI_info[1]
                self.HSI_samples = self.HSI_info[2]
                self.HSI = self.HSI_info[3]
                self.HSI_wavelengths = self.HSI_info[4]
                # Unlock the view and Save function
                self.rgbViewBtn.setEnabled(True)
                self.rgbSaveBtn.setEnabled(True)
                self.showHsiInfoBtn.setEnabled(True)
                QtWidgets.QMessageBox.about(self, "", "高光谱原始数据处理成功")
            
            case "Save":
                if self.rawSpeFile_path != "":
                    self.rgbImg = rd.drawImg(self.HSI_info)
                    self.rgbImg.save("figures/test/raw" + str(self.impFileNum) + ".jpg")
                    QtWidgets.QMessageBox.about(self, "", "高光谱可视化数据保存成功")

            case "View":
                if self.rawSpeFile_path != "":                     
                    self.rawjpgFile_path = "figures/test/raw" + str(self.impFileNum) + ".jpg"
                    frame = QImage(self.rawjpgFile_path)
                    pix = QPixmap.fromImage(frame)
                    item = QGraphicsPixmapItem(pix)
                    # the rgb scene in Tab1
                    self.scene = QGraphicsScene()
                    self.scene.addItem(item)
                    self.hsiRawView.setScene(self.scene)
                    # Make the graph self-adaptive to the canvas
                    self.hsiRawView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

                    self.HSCurveBtn.setEnabled(True)


    def getBRFRgb(self, function):
        match function:
            case "Gene":
                self.HSI_info = rd.ReadData(self.BRFHdrFile_path,self.BRFSpeFile_path, 1)
                self.HSI_lines = self.HSI_info[0]
                self.HSI_channels = self.HSI_info[1]
                self.HSI_samples = self.HSI_info[2]
                self.HSI = self.HSI_info[3]
                self.HSI_wavelengths= self.HSI_info[4]
                # Unlock the view and Save function
                self.BRFRawViewBtn.setEnabled(True)
                self.BRFRawSaveBtn.setEnabled(True)

                self.showHsiInfoBtn.setEnabled(True)

                QtWidgets.QMessageBox.about(self, "", "高光谱反射板处理成功")

            case "Save":
                if self.BRFSpeFile_path != "":
                    self.rgbImg = rd.drawImg(self.HSI_info)
                    self.rgbImg.save("figures/test/raw" + str(self.impFileNum) + ".jpg")
                    QtWidgets.QMessageBox.about(self, "", "高光谱反射板可视化保存成功")

            case "View":
                if self.BRFSpeFile_path != "":                     
                    self.rawjpgFile_path = "figures/test/raw" + str(self.impFileNum) + ".jpg"
                    frame = QImage(self.rawjpgFile_path)
                    pix = QPixmap.fromImage(frame)
                    item = QGraphicsPixmapItem(pix)
                    # the rgb scene in Tab1
                    self.scene = QGraphicsScene()
                    self.scene.addItem(item)
                    self.hsiRawView.setScene(self.scene)

                    # Unlock
                    self.selectBox3Btn.setEnabled(True)
                    self.selectBox30Btn.setEnabled(True)
                    self.HSCurveBtn.setEnabled(True)

    def selectBox(self, brf_flag):
        self.view = hsiRawView(self.scene, brf_flag)
        #self.setCentralWidget(self.view)
        self.view.show()
        self.view.resize(600, 800)
        self.view.startSelection()

        
    def showHsiInfo(self):
        self.lenShowBtn.setText(str(self.HSI_lines)+" pix")
        self.widthShowBtn.setText(str(self.HSI_samples)+" pix")
        self.wlShowBtn.setText(str(self.HSI_channels)+" bands")
        
        self.wavesLayout = QVBoxLayout(self.wavesWidget)
        
        text = "图像具体波段"
        label = QLabel(text)
        label.setStyleSheet("border: none; font: 12pt 'Agency FB';") 
        self.wavesLayout.addWidget(label)

        for i in range(self.HSI_channels):
            text = "band " + str(i+1) + "------" + self.HSI_wavelengths[i] + " nm"
            label = QLabel(text)
            label.setStyleSheet("border: none; font: 12pt 'Times New Roman';") 
            self.wavesLayout.addWidget(label)
        self.WaveScrollArea.setWidgetResizable(True)
 
    # ------------------------------------Tab2------------------------------------
    def getPreProcessPara(self, index):
        combo_box = self.sender()
        match index:
            case 1:
                self.NDVI_TH = float(combo_box.currentText())
            case 2:
                self.ampl_LowTH = int(combo_box.currentText())
            case 3:
                self.ampl_HighTH = int(combo_box.currentText())

    # Remove the background by NDVI
    def RmBg(self, function):
        match function:        
            case "Gene":
                pre_data = pre.preprocess(self.HSI_info, self.NDVI_TH, self.ampl_LowTH, self.ampl_HighTH)
                
                level1 = pre_data.getLevel1()
                self.HSI_info = level1[0]
                self.cur_proportion = level1[2]
                self.NDVI = level1[3]
                # Unlock the view and Save function
                self.RmBgViewBtn.setEnabled(True)
                self.RmBgSaveBtn.setEnabled(True)
                QtWidgets.QMessageBox.about(self, "", "去除非植物部分背景成功")
            
            case "Save":
                l1_rgbimg = rd.drawImg(self.HSI_info)
                self.l1_rgbimg_path = "figures/test/pre_process/" + str(self.impFileNum) + "_level1.jpg"
                l1_rgbimg.save(self.l1_rgbimg_path)
                QtWidgets.QMessageBox.about(self, "", "可视化保存成功")
                '''
                fig, ax = plt.subplots(figsize=(6, 8))
                im = ax.imshow(self.NDVI, cmap='gray',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on NDVI", y=1.05)
                fig.colorbar(im)
                fig.savefig("figures/test/pre_process/" + str(self.impFileNum) + "_level1.jpg")
                QtWidgets.QMessageBox.about(self, "", "NDVI背景生成成功")
                '''
                
            case "View":
                frame = QImage(self.l1_rgbimg_path)
                pix = QPixmap.fromImage(frame)
                item = QGraphicsPixmapItem(pix)
                # the rgb scene in Tab1
                self.scene = QGraphicsScene()
                self.scene.addItem(item)
                self.hsidealtView.setScene(self.scene)
                # Make the graph self-adaptive to the canvas
                #self.hsidealtView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                '''
                fig, ax = plt.subplots(figsize=(6, 8))
                im = ax.imshow(self.NDVI, cmap='gray',interpolation='nearest')
                ax.set_title("Pseudo_Color Map of the Relative Values on NDVI", y=1.05)
                fig.colorbar(im)
                plt.show()
                '''
                

    
    # Remove the too bright and to dark img
    def RmDb(self, function):
            # To remove the shadow and the bright of the plot
        match function:
            case "Gene":
                pre_data = pre.preprocess(self.HSI_info, self.NDVI_TH, self.ampl_LowTH, self.ampl_HighTH)
                level2 = pre_data.getLevel2()
                self.HSI_info = level2[0]
                self.cur_proportion = level2[3]
                # Unlock the view and Save function
                self.RmDbViewBtn.setEnabled(True)
                self.RmDbSaveBtn.setEnabled(True)
                QtWidgets.QMessageBox.about(self, "", "去除过暗过曝成功")

            case "Save":
                l2_rgbImg = rd.drawImg(self.HSI_info)
                self.l2_rgbimg_path = "figures/test/pre_process/" + str(self.impFileNum) + "_level2.jpg"
                l2_rgbImg.save(self.l2_rgbimg_path)
                QtWidgets.QMessageBox.about(self, "", "可视化保存成功")

            case "View":
                frame = QImage(self.l2_rgbimg_path)
                pix = QPixmap.fromImage(frame)
                item = QGraphicsPixmapItem(pix)
                # the rgb scene in Tab1
                self.scene = QGraphicsScene()
                self.scene.addItem(item)
                self.hsidealtView.setScene(self.scene)
                


    # import the amplititude along diferent wavelengths of 3% and 30% BRF
    def importRftCaliFile(self):
        file_dialog = QFileDialog()
        selected_directory = file_dialog.getExistingDirectory(self, "选择文件夹")
        if selected_directory:
            BRFfile_names = os.listdir(selected_directory)
            BRFfile_names = [item.replace("\\","/") for item in BRFfile_names]
            selected_directory = selected_directory.replace("\\","/")
            self.BRFCaliPathlineEdit.setText(selected_directory)
            
            self.BRFfile_paths = [selected_directory + "/" + item for item in BRFfile_names]
 
    def getReflect(self, function):
        match function:        
            case "Gene":
                self.reflect = gr.Reflectance(self.HSI_info, self.cur_proportion, [self.BRF3_pos_range, self.BRF30_pos_range], self.BRFfile_paths, self.k, self.b)
                self.reflect.getReflect()
                # Unlock the view and Save function
                self.RefViewBtn.setEnabled(True)
                self.RefSaveBtn.setEnabled(True)
                QtWidgets.QMessageBox.about(self, "", "反射率校准处理成功")

            case "View":
                self.reflect.visualizeReflect(0)
                return

            case "Save":
                self.reflect.visualizeReflect(1)
                return
            
    def HSCurveView(self):
        self.view = HSCurve(self.scene)
        self.view.show()

    def RFCurveView(self):
        self.view = RFCurve(self.scene)
        self.view.show()


    # ----------------------------Tab3-----------------------------
    def getProcessPara(self, index):
        combo_box = self.sender()
        match index:
            case 1:
                self.Hs_Para = combo_box.currentText()
                print(self.Hs_Para)
            case 2:
                self.Ptsths_Para = combo_box.currentText()
            case 3:
                self.Ptsths_Para_Model = combo_box.currentText()

    def getHsPara(self, function):
        match function:
            case "Gene":
                reflect_info = [self.HSI_lines, self.HSI_channels, self.HSI_samples, self.reflect.ReflectMatrix, self.HSI_wavelengths, self.cur_proportion]
                self.pro_data = pro.process(reflect_info, self.Hs_Para, self.Ptsths_Para, self.Ptsths_Para_Model)
                self.pro_data.calcHsParas()

                # Unlock the view and Save function
                self.hsParaSaveBtn.setEnabled(True)
                self.hsParaViewBtn.setEnabled(True)
                QtWidgets.QMessageBox.about(self, "", "光谱指数计算成功")

            case "Save":
                self.pro_data.draw_pseudoColorImg("Save")
                QtWidgets.QMessageBox.about(self, "", "光谱指数计算结果保存成功")

            case "View":
                self.pro_data.draw_pseudoColorImg("View")

    def getPtsthsPara(self, function):
        match function:
            case "Gene":
                # Unlock the view and Save function
                self.ptsthsViewBtn.setEnabled(True)
                self.ptsthsSaveBtn.setEnabled(True)
                QtWidgets.QMessageBox.about(self, "", "光合表型参数计算成功")


            case "Save":
                QtWidgets.QMessageBox.about(self, "", "光合表型参数计算结果保存成功")


    # ----------------------------Tab4-----------------------------

class hsiRawView(QGraphicsView):
    def __init__(self, scene, brf_flag):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.selection_rect = None
        self.selecting = False
        self.BRF_flag = brf_flag

    def startSelection(self):
        self.selecting = True
        self.selection_rect = QGraphicsRectItem()
        if self.BRF_flag == "3":
            self.selection_rect.setPen(Qt.blue)
        if self.BRF_flag == "30":
            self.selection_rect.setPen(Qt.red)
        self.scene().addItem(self.selection_rect)

    def stopSelection(self):
        if self.selection_rect is not None:
            selected_items = self.scene().items(self.selection_rect.rect(), Qt.IntersectsItemShape)

            # print x and y
            rect = self.selection_rect.rect()

            if self.BRF_flag == "3":
                BRF3_x0 = int(rect.x())
                BRF3_y0 = int(rect.y())
                BRF3_x1 = int(BRF3_x0 + rect.width())
                BRF3_y1 = int(BRF3_y0 + rect.height())
                md.BRF3_pos_range = [[BRF3_x0,BRF3_y0],[BRF3_x1, BRF3_y1]]
                #print(md.BRF3_pos_range)
                
            
            elif self.BRF_flag == "30":
                BRF30_x0 = int(rect.x())
                BRF30_y0 = int(rect.y())
                BRF30_x1 = int(BRF30_x0 + rect.width())
                BRF30_y1 = int(BRF30_y0 + rect.height())
                md.BRF30_pos_range = [[BRF30_x0,BRF30_y0],[BRF30_x1, BRF30_y1]]
                #print(md.BRF30_pos_range)
                
            #self.scene().removeItem(self.selection_rect)
            self.selection_rect = None
        self.selecting = False


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.selecting:
            pos_in_view = event.pos()
            pos_in_scene = self.mapToScene(pos_in_view)
            self.selection_rect.setRect(QRectF(pos_in_scene, pos_in_scene))
            self.scene().addItem(self.selection_rect)

    def mouseMoveEvent(self, event):
        if self.selecting and self.selection_rect is not None:
            pos_in_view = event.pos()
            pos_in_scene = self.mapToScene(pos_in_view)
            rect = QRectF(self.selection_rect.rect().topLeft(), pos_in_scene)
            self.selection_rect.setRect(rect.normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.selecting:
            self.stopSelection()

class HSCurve(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setMouseTracking(True)  # Turn on the mouse track
        self.cursor_pos = QPointF(0, 0) 

        self.crosshair_item = QGraphicsPathItem()
        self.crosshair_item.setPen(QPen(Qt.blue))
        self.scene().addItem(self.crosshair_item)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateCrosshair()

    def mouseMoveEvent(self, event):
        self.cursor_pos = self.mapToScene(event.pos())
        #print(self.cursor_pos)
        self.updateCrosshair()

    def updateCrosshair(self):
        #view_width = self.viewport().width()
        #view_height = self.viewport().height()
        view_width = md.HSI_samples
        view_height = md.HSI_lines
        x = 0
        y = 0

        path = QPainterPath()
       
        if self.cursor_pos.x()>=0 and self.cursor_pos.x()<=view_width and self.cursor_pos.y()>=0 and self.cursor_pos.y()<=view_height:
            # Paint the cross cursor
            path.moveTo(self.cursor_pos.x(), 0)
            path.lineTo(self.cursor_pos.x(), view_height)
            path.moveTo(0, self.cursor_pos.y())
            path.lineTo(view_width, self.cursor_pos.y())
            
            self.crosshair_item.setPath(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x =  np.array(md.HSI_wavelengths)
            y = np.array(md.HSI[int(self.cursor_pos.y()),:,int(self.cursor_pos.x())])
            plt.xlabel("Wavelength(nm)")
            plt.ylabel("Hyperspectral Luminance")
            plt.plot(x, y, c='g', label='Curve_poly_Fit')
            plt.title("The Reflectance curve")

            plt.show()
    
    # delete the cross path after close the event
    def closeEvent(self, event):
        self.crosshair_item.setPath(QPainterPath())
    


class RFCurve(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setMouseTracking(True)  # Turn on the mouse track
        self.cursor_pos = QPointF(0, 0) 

        self.crosshair_item = QGraphicsPathItem()
        self.crosshair_item.setPen(QPen(Qt.blue))
        self.scene().addItem(self.crosshair_item)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateCrosshair()

    def mouseMoveEvent(self, event):
        self.cursor_pos = self.mapToScene(event.pos())
        #print(self.cursor_pos)
        self.updateCrosshair()

    def updateCrosshair(self):
        #view_width = self.viewport().width()
        #view_height = self.viewport().height()
        view_width = md.HSI_samples
        view_height = md.HSI_lines
        x = 0
        y = 0

        path = QPainterPath()
       
        if self.cursor_pos.x()>=0 and self.cursor_pos.x()<=view_width and self.cursor_pos.y()>=0 and self.cursor_pos.y()<=view_height:
            # Paint the cross cursor
            path.moveTo(self.cursor_pos.x(), 0)
            path.lineTo(self.cursor_pos.x(), view_height)
            path.moveTo(0, self.cursor_pos.y())
            path.lineTo(view_width, self.cursor_pos.y())
            
            self.crosshair_item.setPath(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Just show the HS with wavelength within 400nm - 990nm
            x =  np.array(md.HSI_wavelengths[2:-22])
            y = np.array(md.reflect.ReflectMatrix[int(self.cursor_pos.y()),2:-22,int(self.cursor_pos.x())])
            plt.xlabel("Wavelength(nm)")
            plt.ylabel("Reflectance")
    
            plt.plot(x,y,c='lightcoral',label='Curve_poly_Fit')
            plt.title("The Reflectance curve of the cross cursor point")
            plt.show()
    
    # delete the cross path after close the event
    def closeEvent(self, event):
        self.crosshair_item.setPath(QPainterPath())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    md = Main(QMainWindow)
    md.show()
    sys.exit(app.exec_())
            
