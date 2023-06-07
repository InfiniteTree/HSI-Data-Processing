# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1133, 825)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setStyleSheet("QTabBar::tab{width:200}\n"
"QTabBar::tab{height:30}\n"
"")
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(50, 50))
        self.tabWidget.setObjectName("tabWidget")
        self.Tab1 = QtWidgets.QWidget()
        self.Tab1.setMinimumSize(QtCore.QSize(1105, 0))
        self.Tab1.setBaseSize(QtCore.QSize(0, 0))
        self.Tab1.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.Tab1.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Tab1.setObjectName("Tab1")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.Tab1)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.frame = QtWidgets.QFrame(self.Tab1)
        self.frame.setStyleSheet("border: 2px solid black;\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setStyleSheet("border: 1px solid black;\n"
"font: 12pt \"Agency FB\";\n"
"")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_3 = QtWidgets.QFrame(self.frame_2)
        self.frame_3.setGeometry(QtCore.QRect(30, 180, 361, 181))
        self.frame_3.setStyleSheet("")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.showHsiInfoBtn = QtWidgets.QPushButton(self.frame_3)
        self.showHsiInfoBtn.setObjectName("showHsiInfoBtn")
        self.verticalLayout_2.addWidget(self.showHsiInfoBtn)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.frame_3)
        self.pushButton.setStyleSheet("Border: none")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        spacerItem1 = QtWidgets.QSpacerItem(28, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.lenShowBtn = QtWidgets.QLineEdit(self.frame_3)
        self.lenShowBtn.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.lenShowBtn.setObjectName("lenShowBtn")
        self.horizontalLayout.addWidget(self.lenShowBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_19 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_19.setStyleSheet("Border: none")
        self.pushButton_19.setObjectName("pushButton_19")
        self.horizontalLayout_2.addWidget(self.pushButton_19)
        spacerItem3 = QtWidgets.QSpacerItem(28, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.widthShowBtn = QtWidgets.QLineEdit(self.frame_3)
        self.widthShowBtn.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.widthShowBtn.setObjectName("widthShowBtn")
        self.horizontalLayout_2.addWidget(self.widthShowBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_20 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_20.setStyleSheet("Border: none")
        self.pushButton_20.setObjectName("pushButton_20")
        self.horizontalLayout_3.addWidget(self.pushButton_20)
        spacerItem5 = QtWidgets.QSpacerItem(28, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem5)
        self.wlShowBtn = QtWidgets.QLineEdit(self.frame_3)
        self.wlShowBtn.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.wlShowBtn.setObjectName("wlShowBtn")
        self.horizontalLayout_3.addWidget(self.wlShowBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_2)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 10, 511, 116))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.importRawBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.importRawBtn.setFont(font)
        self.importRawBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.importRawBtn.setStatusTip("")
        self.importRawBtn.setObjectName("importRawBtn")
        self.horizontalLayout_14.addWidget(self.importRawBtn)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem6)
        self.importRawsBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.importRawsBtn.setFont(font)
        self.importRawsBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.importRawsBtn.setObjectName("importRawsBtn")
        self.horizontalLayout_14.addWidget(self.importRawsBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_14)
        spacerItem7 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem7)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setStyleSheet("Border: none")
        self.label.setObjectName("label")
        self.horizontalLayout_15.addWidget(self.label)
        self.rawPathlineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.rawPathlineEdit.setObjectName("rawPathlineEdit")
        self.horizontalLayout_15.addWidget(self.rawPathlineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_15)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem8)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.rgbGeneBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.rgbGeneBtn.setObjectName("rgbGeneBtn")
        self.horizontalLayout_13.addWidget(self.rgbGeneBtn)
        self.rgbViewBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.rgbViewBtn.setStatusTip("")
        self.rgbViewBtn.setObjectName("rgbViewBtn")
        self.horizontalLayout_13.addWidget(self.rgbViewBtn)
        self.rgbSaveBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.rgbSaveBtn.setObjectName("rgbSaveBtn")
        self.horizontalLayout_13.addWidget(self.rgbSaveBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_13)
        self.rawImgQlabel = QtWidgets.QLabel(self.frame_2)
        self.rawImgQlabel.setGeometry(QtCore.QRect(550, 0, 525, 720))
        self.rawImgQlabel.setCursor(QtGui.QCursor(QtCore.Qt.BusyCursor))
        self.rawImgQlabel.setStyleSheet("border:1px solid black;")
        self.rawImgQlabel.setObjectName("rawImgQlabel")
        self.horizontalLayout_17.addWidget(self.frame_2)
        self.horizontalLayout_16.addWidget(self.frame)
        self.tabWidget.addTab(self.Tab1, "")
        self.Tab2 = QtWidgets.QWidget()
        self.Tab2.setObjectName("Tab2")
        self.groupBox = QtWidgets.QGroupBox(self.Tab2)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 421, 701))
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(20, 20, 381, 51))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.comboBox_5 = QtWidgets.QComboBox(self.horizontalLayoutWidget_4)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.horizontalLayout_4.addWidget(self.comboBox_5)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem9)
        self.comboBox_2 = QtWidgets.QComboBox(self.horizontalLayoutWidget_4)
        self.comboBox_2.setIconSize(QtCore.QSize(20, 20))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.horizontalLayout_4.addWidget(self.comboBox_2)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(20, 160, 381, 41))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.RemoveSD = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.RemoveSD.setFont(font)
        self.RemoveSD.setObjectName("RemoveSD")
        self.horizontalLayout_5.addWidget(self.RemoveSD)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem10)
        self.comboBox_3 = QtWidgets.QComboBox(self.horizontalLayoutWidget_5)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox_3)
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(10, 210, 391, 41))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.pushButton_9 = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.pushButton_9.setObjectName("pushButton_9")
        self.horizontalLayout_7.addWidget(self.pushButton_9)
        self.pushButton_21 = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.pushButton_21.setObjectName("pushButton_21")
        self.horizontalLayout_7.addWidget(self.pushButton_21)
        self.pushButton_10 = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.pushButton_10.setObjectName("pushButton_10")
        self.horizontalLayout_7.addWidget(self.pushButton_10)
        self.horizontalLayoutWidget_8 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_8.setGeometry(QtCore.QRect(20, 80, 381, 41))
        self.horizontalLayoutWidget_8.setObjectName("horizontalLayoutWidget_8")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_8)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.pushButton_16 = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        self.pushButton_16.setObjectName("pushButton_16")
        self.horizontalLayout_8.addWidget(self.pushButton_16)
        self.pushButton_22 = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        self.pushButton_22.setObjectName("pushButton_22")
        self.horizontalLayout_8.addWidget(self.pushButton_22)
        self.pushButton_17 = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        self.pushButton_17.setObjectName("pushButton_17")
        self.horizontalLayout_8.addWidget(self.pushButton_17)
        self.horizontalLayoutWidget_9 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_9.setGeometry(QtCore.QRect(20, 480, 371, 41))
        self.horizontalLayoutWidget_9.setObjectName("horizontalLayoutWidget_9")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_9)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButton_11 = QtWidgets.QPushButton(self.horizontalLayoutWidget_9)
        self.pushButton_11.setObjectName("pushButton_11")
        self.horizontalLayout_9.addWidget(self.pushButton_11)
        self.pushButton_23 = QtWidgets.QPushButton(self.horizontalLayoutWidget_9)
        self.pushButton_23.setObjectName("pushButton_23")
        self.horizontalLayout_9.addWidget(self.pushButton_23)
        self.pushButton_12 = QtWidgets.QPushButton(self.horizontalLayoutWidget_9)
        self.pushButton_12.setObjectName("pushButton_12")
        self.horizontalLayout_9.addWidget(self.pushButton_12)
        self.horizontalLayoutWidget_10 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_10.setGeometry(QtCore.QRect(20, 350, 381, 41))
        self.horizontalLayoutWidget_10.setObjectName("horizontalLayoutWidget_10")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_10)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.pushButton_15 = QtWidgets.QPushButton(self.horizontalLayoutWidget_10)
        self.pushButton_15.setObjectName("pushButton_15")
        self.horizontalLayout_10.addWidget(self.pushButton_15)
        self.pushButton_24 = QtWidgets.QPushButton(self.horizontalLayoutWidget_10)
        self.pushButton_24.setObjectName("pushButton_24")
        self.horizontalLayout_10.addWidget(self.pushButton_24)
        self.pushButton_25 = QtWidgets.QPushButton(self.horizontalLayoutWidget_10)
        self.pushButton_25.setObjectName("pushButton_25")
        self.horizontalLayout_10.addWidget(self.pushButton_25)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(20, 300, 381, 41))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.RemoveNT = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.RemoveNT.setFont(font)
        self.RemoveNT.setObjectName("RemoveNT")
        self.horizontalLayout_6.addWidget(self.RemoveNT)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem11)
        self.comboBox_4 = QtWidgets.QComboBox(self.horizontalLayoutWidget_6)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.horizontalLayout_6.addWidget(self.comboBox_4)
        self.horizontalLayoutWidget_11 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_11.setGeometry(QtCore.QRect(20, 420, 381, 41))
        self.horizontalLayoutWidget_11.setObjectName("horizontalLayoutWidget_11")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_11)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.GetReflectamce = QtWidgets.QPushButton(self.horizontalLayoutWidget_11)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.GetReflectamce.setFont(font)
        self.GetReflectamce.setObjectName("GetReflectamce")
        self.horizontalLayout_11.addWidget(self.GetReflectamce)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem12)
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget_11)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.horizontalLayout_11.addWidget(self.comboBox)
        self.horizontalLayoutWidget_12 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_12.setGeometry(QtCore.QRect(20, 590, 381, 51))
        self.horizontalLayoutWidget_12.setObjectName("horizontalLayoutWidget_12")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_12)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.radioButton = QtWidgets.QRadioButton(self.horizontalLayoutWidget_12)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout_12.addWidget(self.radioButton)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem13)
        self.pushButton_18 = QtWidgets.QPushButton(self.horizontalLayoutWidget_12)
        self.pushButton_18.setObjectName("pushButton_18")
        self.horizontalLayout_12.addWidget(self.pushButton_18)
        self.commandLinkButton = QtWidgets.QCommandLinkButton(self.Tab2)
        self.commandLinkButton.setGeometry(QtCore.QRect(1050, 310, 41, 48))
        self.commandLinkButton.setText("")
        self.commandLinkButton.setObjectName("commandLinkButton")
        self.groupBox_2 = QtWidgets.QGroupBox(self.Tab2)
        self.groupBox_2.setGeometry(QtCore.QRect(490, 0, 611, 731))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 581, 701))
        self.label_2.setStyleSheet("border:1px solid black;")
        self.label_2.setObjectName("label_2")
        self.commandLinkButton_2 = QtWidgets.QCommandLinkButton(self.groupBox_2)
        self.commandLinkButton_2.setGeometry(QtCore.QRect(40, 310, 41, 48))
        self.commandLinkButton_2.setBaseSize(QtCore.QSize(0, 0))
        self.commandLinkButton_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.commandLinkButton_2.setAutoFillBackground(False)
        self.commandLinkButton_2.setText("")
        self.commandLinkButton_2.setDescription("")
        self.commandLinkButton_2.setObjectName("commandLinkButton_2")
        self.tabWidget.addTab(self.Tab2, "")
        self.Tab3 = QtWidgets.QWidget()
        self.Tab3.setObjectName("Tab3")
        self.tabWidget.addTab(self.Tab3, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        self.verticalLayout_3.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionopen_file = QtWidgets.QAction(MainWindow)
        self.actionopen_file.setEnabled(True)
        self.actionopen_file.setObjectName("actionopen_file")
        self.actionOpen_File_Folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_File_Folder.setObjectName("actionOpen_File_Folder")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setToolTip(_translate("MainWindow", "<html><head/><body><p>1</p></body></html>"))
        self.showHsiInfoBtn.setText(_translate("MainWindow", "显示高光谱图像信息"))
        self.pushButton.setText(_translate("MainWindow", "图像长度:"))
        self.lenShowBtn.setText(_translate("MainWindow", "0        pix"))
        self.pushButton_19.setText(_translate("MainWindow", "图像宽度"))
        self.widthShowBtn.setText(_translate("MainWindow", "0        pix"))
        self.pushButton_20.setText(_translate("MainWindow", "图像波段"))
        self.wlShowBtn.setText(_translate("MainWindow", "NAN"))
        self.importRawBtn.setText(_translate("MainWindow", "导入单张原始照片"))
        self.importRawsBtn.setText(_translate("MainWindow", "导入多张原始照片"))
        self.label.setText(_translate("MainWindow", "文件路径："))
        self.rawPathlineEdit.setText(_translate("MainWindow", "高光谱数据文件路径"))
        self.rgbGeneBtn.setText(_translate("MainWindow", "生成"))
        self.rgbViewBtn.setText(_translate("MainWindow", "查看"))
        self.rgbSaveBtn.setText(_translate("MainWindow", "保存"))
        self.rawImgQlabel.setText(_translate("MainWindow", "Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Tab1), _translate("MainWindow", "原始数据"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "NDVI_背景移除"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "NAN"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "0.8"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "-1.0"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "-0.6"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "-0.2"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "0.2"))
        self.comboBox_2.setItemText(5, _translate("MainWindow", "0.6"))
        self.comboBox_2.setItemText(6, _translate("MainWindow", "1.0"))
        self.RemoveSD.setText(_translate("MainWindow", "暗背景矫正"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "200"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "400"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "600"))
        self.pushButton_9.setText(_translate("MainWindow", "生成"))
        self.pushButton_21.setText(_translate("MainWindow", "显示"))
        self.pushButton_10.setText(_translate("MainWindow", "保存"))
        self.pushButton_16.setText(_translate("MainWindow", "生成"))
        self.pushButton_22.setText(_translate("MainWindow", "显示"))
        self.pushButton_17.setText(_translate("MainWindow", "保存"))
        self.pushButton_11.setText(_translate("MainWindow", "生成"))
        self.pushButton_23.setText(_translate("MainWindow", "显示"))
        self.pushButton_12.setText(_translate("MainWindow", "保存"))
        self.pushButton_15.setText(_translate("MainWindow", "生成"))
        self.pushButton_24.setText(_translate("MainWindow", "显示"))
        self.pushButton_25.setText(_translate("MainWindow", "保存"))
        self.RemoveNT.setText(_translate("MainWindow", "过曝矫正"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "1000"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "2000"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "3000"))
        self.GetReflectamce.setText(_translate("MainWindow", "反射率计算"))
        self.comboBox.setItemText(0, _translate("MainWindow", "导入反射率校准板"))
        self.radioButton.setText(_translate("MainWindow", "使用默认值"))
        self.pushButton_18.setText(_translate("MainWindow", "一键生成"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_2.setText(_translate("MainWindow", "Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Tab2), _translate("MainWindow", "数据预处理"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Tab3), _translate("MainWindow", "数据分析"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "模型训练"))
        self.actionopen_file.setText(_translate("MainWindow", "Import File"))
        self.actionOpen_File_Folder.setText(_translate("MainWindow", "Open File Folder"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setStatusTip(_translate("MainWindow", "Save a file"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
