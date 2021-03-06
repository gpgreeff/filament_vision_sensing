# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainControl_placeholder.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(486, 633)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_interface = QtWidgets.QWidget()
        self.tab_interface.setObjectName("tab_interface")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_interface)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.tab_interface)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.text_vision = QtWidgets.QTextBrowser(self.frame_2)
        self.text_vision.setDocumentTitle("")
        self.text_vision.setObjectName("text_vision")
        self.gridLayout_3.addWidget(self.text_vision, 1, 0, 2, 1)
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.check_vision = QtWidgets.QPushButton(self.frame_2)
        self.check_vision.setEnabled(False)
        self.check_vision.setObjectName("check_vision")
        self.gridLayout_3.addWidget(self.check_vision, 2, 1, 1, 1)
        self.widget = QtWidgets.QWidget(self.frame_2)
        self.widget.setObjectName("widget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_3.addWidget(self.widget, 1, 1, 1, 1)
        self.verticalLayout_3.addWidget(self.frame_2)
        self.tabWidget.addTab(self.tab_interface, "")
        self.tab_data = QtWidgets.QWidget()
        self.tab_data.setObjectName("tab_data")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab_data)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame_plot = QtWidgets.QFrame(self.tab_data)
        self.frame_plot.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_plot.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_plot.setObjectName("frame_plot")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_plot)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.gridLayout_plot = QtWidgets.QGridLayout()
        self.gridLayout_plot.setObjectName("gridLayout_plot")
        self.verticalLayout_6.addLayout(self.gridLayout_plot)
        self.line_4 = QtWidgets.QFrame(self.frame_plot)
        self.line_4.setMinimumSize(QtCore.QSize(0, 40))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_6.addWidget(self.line_4)
        self.groupBox = QtWidgets.QGroupBox(self.frame_plot)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.recording_meas_only = QtWidgets.QRadioButton(self.groupBox)
        self.recording_meas_only.setObjectName("recording_meas_only")
        self.gridLayout.addWidget(self.recording_meas_only, 0, 0, 1, 1)
        self.recording_start = QtWidgets.QPushButton(self.groupBox)
        self.recording_start.setObjectName("recording_start")
        self.gridLayout.addWidget(self.recording_start, 0, 1, 1, 1)
        self.recording_video_and_meas = QtWidgets.QRadioButton(self.groupBox)
        self.recording_video_and_meas.setChecked(True)
        self.recording_video_and_meas.setObjectName("recording_video_and_meas")
        self.gridLayout.addWidget(self.recording_video_and_meas, 1, 0, 1, 1)
        self.recording_video_only = QtWidgets.QRadioButton(self.groupBox)
        self.recording_video_only.setObjectName("recording_video_only")
        self.gridLayout.addWidget(self.recording_video_only, 2, 0, 1, 1)
        self.recording_stop = QtWidgets.QPushButton(self.groupBox)
        self.recording_stop.setEnabled(False)
        self.recording_stop.setMinimumSize(QtCore.QSize(180, 0))
        self.recording_stop.setObjectName("recording_stop")
        self.gridLayout.addWidget(self.recording_stop, 1, 1, 2, 1)
        self.verticalLayout_6.addWidget(self.groupBox)
        self.line = QtWidgets.QFrame(self.frame_plot)
        self.line.setMinimumSize(QtCore.QSize(0, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_6.addWidget(self.line)
        self.process_video = QtWidgets.QPushButton(self.frame_plot)
        self.process_video.setObjectName("process_video")
        self.verticalLayout_6.addWidget(self.process_video)
        self.line_2 = QtWidgets.QFrame(self.frame_plot)
        self.line_2.setMinimumSize(QtCore.QSize(0, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_6.addWidget(self.line_2)
        self.text_main = QtWidgets.QTextBrowser(self.frame_plot)
        self.text_main.setObjectName("text_main")
        self.verticalLayout_6.addWidget(self.text_main)
        spacerItem = QtWidgets.QSpacerItem(316, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_6.addItem(spacerItem)
        self.verticalLayout_5.addWidget(self.frame_plot)
        self.tabWidget.addTab(self.tab_data, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 486, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Main Control"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">From Vision</span></p></body></html>"))
        self.check_vision.setText(_translate("MainWindow", "Vision Interface Check"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_interface), _translate("MainWindow", "Interface"))
        self.groupBox.setTitle(_translate("MainWindow", "New Video Recording"))
        self.recording_meas_only.setText(_translate("MainWindow", "Do not record video file, but measure"))
        self.recording_start.setText(_translate("MainWindow", "Start Recording"))
        self.recording_video_and_meas.setText(_translate("MainWindow", "Record video file and measure"))
        self.recording_video_only.setText(_translate("MainWindow", "Only record video, do not measure"))
        self.recording_stop.setText(_translate("MainWindow", "Stop Recording and Save Data"))
        self.process_video.setText(_translate("MainWindow", "Process Video"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_data), _translate("MainWindow", "Data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

