# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:47:36 2017


How to convert main.ui to main_ui.py file (a or b):
a)  from command line:
    pyuic5 -x main.ui -o main_ui.py
    pyuic5.bat:
    @"C:/Program Files/Anaconda3\python.exe" -m PyQt5.uic.pyuic %1 %2 %3 %4 %5 %6 %7 %8 %9
    
b)  form_class = uic.loadUiType("main.ui")[0]

"""
__version__ = "0.5"
__license__ = "MIT"

from ExtruderCam2 import VisionProcess

import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow

from MatplotlibWidget import MplCanvasWidget
from QtLoadSaveObjects import make_date_filename,append_texteditor
from UI.MainControl_placeholder_ui import Ui_MainWindow

#==============================================================================
#===============================================================================
class MainControlPlaceHolder(QMainWindow):
    """
    Placeholder for main control.
    Collects data and saves it with Pandas.
    
    Replace with Printer Interface Controler
    """
    def __init__(self,**kwargs):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        #init plot widget
        self.plot_hist_len = 1000 #pnts
        self.plot_init()
        
        
        #data collection
        self.vision_config = {} #header for data file
        self.video_filename = None
        self.data_folder = 'data'
        
        #http://stackoverflow.com/questions/10715965/
        #add-one-row-in-a-pandas-dataframe @ShikharDua
        #rows in the data, received per event each row item is a dict
        #this is put into dataframes, after meas completed
        #dataframes are then merged
        self.vision_row = [] 
#        self.tempereture_row = [] 
        
        
        #process control 
        self.update_count = 0
        self.vision_close_requested = False
        self.recording = False #busy or not
        self.video_mode = kwargs.get('video_mode',False)
        
        #vision process
        self.pipe_2_vision_in,self.pipe_2_vision_out = multiprocessing.Pipe()
        self.pipe_from_vision_in,self.pipe_from_vision_out = multiprocessing.Pipe()
        self.vision_process = VisionProcess(self.pipe_2_vision_out,
                                            self.pipe_from_vision_in,
                                            load_video_def = self.video_mode)
        
        #connections
        self.ui.recording_start.clicked.connect(self.recording_start_clicked)
        self.ui.recording_stop.clicked.connect(self.recording_stop_clicked)
        self.ui.process_video.clicked.connect(self.process_video_clicked)
        
        #UI settings
        self.ui.recording_start.setEnabled(not(self.video_mode))
        self.ui.process_video.setEnabled(self.video_mode)
        
        
        #main loop
        self.main_timer_rate = 15 #ms
        self.main_timer = QtCore.QTimer(self)
        self.main_timer.timeout.connect(self.main_loop)
        
        #START
        self.vision_process.start()
        self.main_timer.start(self.main_timer_rate)

    def main_loop(self):
        """
        """
        #check pipe
        short_count = 10
        while self.pipe_from_vision_out.poll() and short_count > 0:
            vision_data = self.pipe_from_vision_out.recv()
            short_count -= 1
            if isinstance(vision_data,dict):
                if vision_data.get('name') == 'frame_result':
                    #update/save new frame result
                    
                    vision_data.pop('name')
                    if self.recording:
                        self.vision_row.append(vision_data.copy())
                    
                    #plot data
                    self.plot_gear[:,1] = np.roll(self.plot_gear[:,1],-1)
                    self.plot_gear[-1,1] = vision_data['gear_speed']*vision_data['cal_factor']*self.vision_fps
                    
                    self.plot_fil[:,1] = np.roll(self.plot_fil[:,1],-1)
                    self.plot_fil[-1,1] = vision_data['filament_speed']*vision_data['cal_factor']*self.vision_fps

                    self.plot_width[:,1] = np.roll(self.plot_width[:,1],-1)
                    self.plot_width[-1,1] = vision_data['filament_width']*vision_data['cal_factor']

                    self.update_count += 1
                    
                    
                elif vision_data.get('name') == 'setup_details':
#                    filament = vision_data.get('user_filament_width',2.85)
                    self.vision_config = vision_data.get('config')
                    self.vision_fps = self.vision_config['camera']['cam_fps']
                
                elif vision_data.get('name') == 'video_filename':
#                    filament = vision_data.get('user_filament_width',2.85)
                    self.video_filename = vision_data.get('video_filename')
                
            else: #hopefully just a string
                append_texteditor(self.ui.text_vision,vision_data)
#                print(type(vision_data))
        
        #UPDATE plot
        if self.update_count > 1:
            self.update_count = 0
            update_data_y = [self.plot_gear[:,1],
                             self.plot_fil[:,1],
                             self.plot_width[:,1]]

            self.plot_data_widget.fast_update_y(update_data_y)

                
    def plot_init(self):
        """
        """
  
        self.plot_gear = np.zeros([self.plot_hist_len,2])
        self.plot_fil = np.zeros([self.plot_hist_len,2])
        self.plot_width = np.zeros([self.plot_hist_len,2])
        
        self.plot_gear[:,0] = np.arange(self.plot_hist_len)
        self.plot_fil[:,0] = np.arange(self.plot_hist_len)
        self.plot_width[:,0] = np.arange(self.plot_hist_len)
       

        
        gear_arg = [self.plot_gear[:,0],self.plot_gear[:,1]]
        gear_kwarg = {'c':'k','xmax' : self.plot_hist_len,'ymin' : 0,'ymax' : 4, 'label':'gear'}
        
        fil_arg = [self.plot_fil[:,0],self.plot_fil[:,1]]
        fil_kwarg = {'c':'r','xmax' : self.plot_hist_len,'ymin' : 0,'ymax' : 4, 'label':'filament'}

        width_arg = [self.plot_gear[:,0],self.plot_gear[:,1]]
        width_kwarg = {'c':'k','xmax' : self.plot_hist_len,'ymin' : 2,'ymax' : 4, 'label':'width'}
        
        
        initial_plot = [[0,gear_arg,gear_kwarg],
                        [0,fil_arg,fil_kwarg],
                        [1,width_arg,width_kwarg]]
        
        self.plot_data_widget = MplCanvasWidget(self, width=5, height=4, dpi=100,initial_data = initial_plot)
        
        self.plot_data_widget.axes[0].legend()
        self.plot_data_widget.axes[1].legend()
        plt.ion()
        plt.show()
        
        self.ui.gridLayout_plot.addWidget(self.plot_data_widget,0,0)

    def recording_start_clicked(self):
        """
        """        
        append_texteditor(self.ui.text_main,'Start recording')
        if self.ui.recording_meas_only.isChecked():
            self.pipe_2_vision_in.send({'measure':True})
        
        elif self.ui.recording_video_and_meas.isChecked():
            self.pipe_2_vision_in.send({'measure':True,'record_video':True})
            
        elif self.ui.recording_video_only.isChecked():
            self.pipe_2_vision_in.send({'measure':False,'record_video':True})
            
        self.ui.recording_stop.setEnabled(True)    
        self.ui.recording_start.setEnabled(False)
        self.recording = True

    def recording_stop_clicked(self):
        """
        """
        self.recording = False
        self.pipe_2_vision_in.send({'measure':False,'save_video':True})
        if not(self.ui.recording_video_only.isChecked()):
            self.save_results()
        self.ui.recording_stop.setEnabled(False)    
        self.ui.recording_start.setEnabled(True)
        append_texteditor(self.ui.text_main,'Stop recording')
        

    def save_results(self):
        """
        """
        append_texteditor(self.ui.text_main,'Saving')
        
        df_vision = pd.DataFrame(self.vision_row) 
        #df_other = pd.DataFrame(self.other_row) ...
        
#        df_temper = pd_convert_timestamp(df_temper)
        df_vision = pd_convert_timestamp(df_vision)
        
#        
#        #save result in single file
#        result = pd.concat([df_vision,df_temper,df_other...], axis=1)
        result = df_vision
        
        if self.video_filename is None:
            job_name = 'test'
            filename = make_date_filename(job_name,ext = 'csv',
                                           folder = self.data_folder)
        else:
            filename = self.video_filename.split('.avi')[0]
            filename += '.csv'
        
        with open(filename, 'w') as csv_file:
            write_header_metadata(csv_file,self.vision_config) #add header
            result.to_csv(csv_file) #write data
        
        append_texteditor(self.ui.text_main,'Saved results: {}'.format(filename))
    
    def process_video_clicked(self):
        """
        """
        append_texteditor(self.ui.text_main,'Start processing video')
        self.pipe_2_vision_in.send({'process_video':True})
        
        
    def closeEvent(self, event):
        """
        """
        if not(self.vision_close_requested):
            append_texteditor(self.ui.text_main,'Close event')
            self.pipe_2_vision_in.send('exit')
            self.vision_close_requested = True
            
        if self.vision_process.is_alive():
            QtCore.QTimer.singleShot(10, self.close)
            event.ignore()
        else:
            self.main_timer.stop()
            event.accept()
        
#==============================================================================
def pd_convert_timestamp(df,drop_duplicates = 'time_stamp'):
    """
    df- DataFrame, put time_stamp col as datetime index
    """
    if drop_duplicates is not None:
        
            df.drop_duplicates('time_stamp',inplace = True) 
        
        #prevent error on concat, temperature values seems to have double rows
    
    df['time_stamp'] = pd.DatetimeIndex(pd.to_datetime(df['time_stamp'],
                                        format = '%H:%M:%S.%f',
                                        exact = False))
    df.set_index('time_stamp',drop = True,inplace = True)

    return df
#==============================================================================
def write_header_metadata(filehandle,header,comment_char = '#'):
    """
    Header is a dictionary, for each item in the dict write a row to the filehandle
    with the following format: '#key: value'
    
    Args:
        filehandle: filehandle
        header(dict): meta data, such as configs, to put the start of the file
        comment_char(str)
    
    """
    lines = json.dumps(header, indent = 4, sort_keys=True)
    for line in lines.split('\n'):
        filehandle.write('#{}\n'.format(line))
#==============================================================================    
if __name__ == '__main__':
    if QtCore.QT_VERSION >= 0x50501:
        import traceback
        def excepthook(type_, value, traceback_):
            traceback.print_exception(type_, value, traceback_)
            QtCore.qFatal('')
        sys.excepthook = excepthook     
        
    app = QApplication(sys.argv)

    app_window = MainControlPlaceHolder()
    app_window.setWindowTitle("Main Controller Data View")
    app_window.show()
    sys.exit(app.exec_())

    