# -*- coding: utf-8 -*-
"""
ExtruderCam

Interface with camera or video stream, analyse the ROI - regions of interest -
in the image frame, and report results through a pipe/queue

User interface with PyQt.
    
Note:
    How to convert main.ui to main_ui.py file (a or b)
    a)  from command line:
        pyuic5 -x main.ui -o main_ui.py
        pyuic5.bat:
        @"C:/Program Files/Anaconda3\python.exe" -m PyQt5.uic.pyuic %1 %2 %3 %4 %5 %6 %7 %8 %9
        
    b)  form_class = uic.loadUiType("main.ui")[0]


Created on Wed Jan  4 15:23:32 2017


"""
__version__ = "2.0"
__license__ = "MIT"
__author__  ="GP Greeff"

from multiprocessing import Process#,Pipe
import cv2
import configparser
import os

#import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

from numpy import average as np_ave
from numpy import ndarray as np_ndarray

from UI.ExtruderCam2_ui import Ui_MainWindow
from ToolsCV import CameraLabel,converter,rotate_image,cv_colors,VideoStreamWorker
import ROIA
from QtLoadSaveObjects import load_file_name,get_files_like,set_combo_index
from QtLoadSaveObjects import make_config_dict,make_date_filename,append_texteditor
from TableTools import set_table_dictlist

class ExtruderCam(QtWidgets.QMainWindow):
    """
    
    Attributes:
        sig_abort_streamworkers (pyqtSignal): signal to worker threads to stop capturing frames
   
            
    Args:
        main_app (QApplication): used by worker threads to process thread events 
                                 i.e. check for sig_abort_streamworkers.
        pipe_in (multiprocessing.Pipe()): pipe in,  first end of first pipe
        pipe_out(multiprocessing.Pipe()): pipe out, second end of second pipe    
        
    """
    sig_abort_streamworkers = QtCore.pyqtSignal()
    
    def __init__(self,main_app,pipe_in = None,pipe_out = None,**kwargs):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.app_settings = kwargs
        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
        
        self.close_attempts = 0 #complicated exit process
        
        self.measure_active = False #when active, do not display cv dwg to increase speed
        self.process_active = True #run realtime in-processing
        self.new_frame_processed = False #update display on True
        self.update_areas = False #True, new/update areas and frame
        self.frame_id = 0    # id used in ROIA for update of points, 
                             # also the current video frame number
        self.prev_frame = -1
        self.convert_speed = 1.0 #cal factor/fps
        self.convert_width = 1.0 #cal factor
        self.time_passed_ms = (1/30.)*1000
#        self.missed_frames = 0

        #collect a few filament width estimations over time
        self.cal_collect_width = False  #control
        self.cal_width_history = []     #collected data
        self.cal_w_len = 60             #number to collect
                    
        #result reporting -send dict via pipe
        self.frame_result = {}
        self.frame_result['name'] = 'frame_result'
        self.frame_result['time_stamp'] = ''
        self.frame_result['gear_speed'] = 0 #pxl/frame
        self.frame_result['filament_speed'] = 0 #pxl/frame
        self.frame_result['filament_width'] = 0 #pxl
        self.frame_result['time_elasped'] = 0 #sec
        self.frame_result['cal_factor'] = 1.0 #mm/pxl
        self.frame_result['frame_id'] = 1.0 #
        
        #setup reporting -send dict via pipe
        self.setup_details = {}
        self.setup_details['name'] = 'setup_details'
        self.setup_details['user_filament_width'] = 2.85 #mm
        self.setup_details['config'] = {}
        
        #video recording reporting
        self.video_details = {}
        self.video_details['name'] = 'video_filename'
        self.video_details['video_filename'] = None
        self.data_folder = 'data'
        
        #load *.cfg
        self.load_configs()
        
        #timers
        self.current_time = QtCore.QTime()
        self.main_timer_rate = 50 #ms
        self.main_timer = QtCore.QTimer(self)
        self.main_timer.timeout.connect(self.main_task_loop)
        
        #source
        self.video_capture = None #from a video file
        self.stream = None #from camera
        self.frame_extruder = None
        
        #threading
        self.__workers_done = True
        self.__threads = []
        
        if kwargs.get('load_video_def',False):
            self.video_mode = True
            self.load_video_from_file(video_fullname = 'data\Test_2017_02_17__08_45_46.avi')
            self.ui.load_video.setEnabled(True)
            
        else:
            self.video_mode = False
            self.init_threads(main_app)
            
        
        #video recording settings
        self.recording = False #busy recording
        self.record_file = None 
        self.record_filename_prev = ''
        self.record_fps = self.stream_fps # fps for video file
            
        
        #UI camera display
        self.microscope_display = CameraLabel(self.ui.groupBox_video)
        self.microscope_display.setMaximumSize(QtCore.QSize(640, 480))
        self.microscope_display.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.microscope_display.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.microscope_display.setLineWidth(2)
        self.microscope_display.setScaledContents(True)
        self.microscope_display.setAlignment(QtCore.Qt.AlignCenter)
        self.microscope_display.setObjectName("microscope_display")
        self.ui.verticalLayout_2.addWidget(self.microscope_display)
        
        if self.frame_extruder is not None:
            self.microscope_display.pixmap = converter(self.frame_extruder)
        
        #init configs
        self.areas_setup()
        self.set_cal_values()
        
        #CONNECTIONS
        #UI ROI setup
        self.ui.img_rotation.valueChanged.connect(self.set_update_areas)
        
        self.ui.vline_left.valueChanged.connect(self.set_update_areas)
        self.ui.vline_right.valueChanged.connect(self.set_update_areas)
        
        self.ui.w_y2.valueChanged.connect(self.set_update_areas)
        self.ui.w_vline_left_border.valueChanged.connect(self.set_update_areas)
        self.ui.w_vline_right_border.valueChanged.connect(self.set_update_areas)
        
        self.ui.fil_roi_x.valueChanged.connect(self.set_update_areas)
        self.ui.fil_roi_y.valueChanged.connect(self.set_update_areas)
        self.ui.fil_roi_size.valueChanged.connect(self.set_update_areas)
        
        self.ui.gear_roi_x.valueChanged.connect(self.set_update_areas)
        self.ui.gear_roi_y.valueChanged.connect(self.set_update_areas)
        self.ui.gear_roi_size.valueChanged.connect(self.set_update_areas)
        
        self.ui.mat_colour.currentIndexChanged.connect(self.set_update_areas)
        
        self.ui.save_config.clicked.connect(self.save_config)
        
        #calibration
#        self.ui.filament_width.valueChanged.connect(self.updated_std_width)
        self.ui.calibrate.clicked.connect(self.start_cal)
  
        #video playback
        self.ui.load_video.clicked.connect(self.load_video_from_file)
        self.ui.frame_number.valueChanged.connect(self.new_frame_trackbar)
        self.ui.frame_next.clicked.connect(self.new_frame_next)
        self.ui.frame_prev.clicked.connect(self.new_frame_prev)
        
        #video recording
        self.ui.record_video.clicked.connect(self.record_video_start_stop)
        self.ui.record_save.clicked.connect(self.record_video_save)
        self.ui.record_start_new.clicked.connect(self.record_start_new)
        
        #---
        qr = self.frameGeometry()
        self.move(qr.topLeft())
        self.setWindowTitle('Vision')
        self.ui.report_results.setChecked(False)
        
        if self.video_mode:
            self.ui.tabWidget.setCurrentIndex(4)
        
        #DEBUG
        #profile
        self.busy_profile = kwargs.get('do_profiling',0) #0 or number of frames to profile
        
        self.ping_count = 0
        
        #START
        current_time_str = self.current_time.currentTime().toString('hh:mm:ss.zzz')
        append_texteditor(self.ui.text_pipe_out,'start time_stamp: {}'.format(current_time_str))
        
        #start threads if any
        for thread,worker in self.__threads:
            thread.start()  # this will emit 'started' and start thread's event loop

            
        self.main_timer.start(self.main_timer_rate)
        self.pipe_out.send('vision started')
        self.updated_std_width(self.ui.filament_width.value())

    def init_threads(self,main_app):
        """
        """
        self.__workers_done = False
        idx = 0 #more than one camera should be possible
        worker = VideoStreamWorker(idx,main_app,**self.setup_details['config']['camera'])
        thread = QtCore.QThread()
        thread.setObjectName('thread_' + str(idx))
        self.__threads.append((thread, worker))  # need to store worker too otherwise will be gc'd
        worker.moveToThread(thread)

        #get progress messages from worker
        worker.sig_new_frame.connect(self.process_new_frame)
        worker.sig_done.connect(self.on_worker_done)
        worker.sig_msg.connect(self.on_worker_info)
        worker.sig_error.connect(self.on_worker_info)

        #control worker
        self.sig_abort_streamworkers.connect(worker.abort)
        
        #init/info vals
        self.stream_fps = worker.fps
        self.frame_extruder = worker.frame
        self.update_cam_table(worker)
        
        #get ready to start worker
        thread.started.connect(worker.work)
        
            
        
        
    @QtCore.pyqtSlot(int,int,float,np_ndarray)
    def process_new_frame(self, thread_id:int,frame_id: int, time_step: float,frame:np_ndarray):
        """
        Called when camera processed a new frame,
        or by main loop during video file review
        
        """   
        self.frame_extruder = frame
        self.time_passed_ms = time_step
        self.frame_id = frame_id
        
        if self.recording:
            self.record_file.write(self.frame_extruder)
            
        #re-init all areas from scratch, with new settings
        if self.update_areas:
            self.areas_setup() 
            self.update_areas = False

        if self.process_active:
            self.pre_process_whole()
            self.process_areas()
            
        self.new_frame_processed = True
        

    @QtCore.pyqtSlot(int,str)
    def on_worker_info(self, worker_id,worker_msg):
        self.pipe_out.send('Threading: {}'.format(worker_msg))

    @QtCore.pyqtSlot(int)
    def on_worker_done(self, worker_id):
        self.pipe_out.send('Threading: Worker {} DONE'.format(worker_id))
        self.__workers_done = True  #for more than one count down until all are done          

    @QtCore.pyqtSlot()
    def abort_workers(self):
        self.sig_abort_streamworkers.emit()
        self.pipe_out.send('Threading: asking each worker to abort')
        for thread, worker in self.__threads:  # note nice unpacking by Python, avoids indexing
            thread.quit()  # this will quit **as soon as thread event loop unblocks**
            thread.wait()  # <- so you need to wait for it to *actually* quit

        # even though threads have exited, there may still be messages on the main thread's
        # queue (messages that threads emitted before the abort):
        self.pipe_out.send('Threading: All threads exited')
        

              
    def main_task_loop(self):
        """
        """
        #replay recorded video
        if self.video_mode: 
            if self.video_new_frame():
                self.process_new_frame(0,self.frame_id,self.time_passed_ms,self.frame_extruder)
                
        #update display
        if self.new_frame_processed:
            self.new_frame_processed = False
            self.post_process()
            
        elif not(self.video_mode):
            self.ping_count += 1
            if self.ping_count > 30:
                self.pipe_out.send('ping {}'.format(self.ping_count ))
                self.ping_count = 0
                date_time =  QtCore.QDateTime.currentDateTime()
                msg = date_time.toString()
                self.pipe_out.send(msg)
                self.new_frame_processed = True
                

        #poll comms        
        if self.pipe_in.poll():
            new_task = self.pipe_in.recv()
            append_texteditor(self.ui.text_pipe_in,'New task {}'.format(new_task))
            if isinstance(new_task,dict):
                    #process video
                    if new_task.get('process_video',False):
                        self.ui.report_results.setChecked(True)
                        self.ui.play_video.setChecked(True)
                    
                    #process camera
                    else:
                        measure = new_task.get('measure',False)
                        if measure:
                            self.updated_std_width()
                            
                        if new_task.get('record_video',False):
                            self.ui.record_video.setChecked(True)
                            self.record_video_start_stop(new_state = True)
                        
                        if new_task.get('save_video',False):
                            self.record_video_save()
                        
                        self.ui.report_results.setChecked(measure)
                        self.measure_active = measure and not(self.video_mode)
                        self.process_active = not(self.recording and not(self.ui.report_results.isChecked()))
                    
                    
            else:
                #string command or something else
                if new_task == 'exit':
                    self.close()
                elif new_task == 'init_dino_cam':
                    #try re-init cam
#                    self.init_dino_cam()
                    pass
        
        #only execute for limeted number of frames  
        if self.busy_profile > 0:
            self.busy_profile -= 1
            if self.busy_profile <= 0:
                self.close()
                    

    def closeEvent(self, event):
        """
        Check if exit required, then save configs and close connections and 
        other widgets. Important to release caputre.
        
        """
#        quit_msg = 'Quit?'
#        reply = QtGui.QMessageBox.question(self, 'Exit',quit_msg,
#                                           QtGui.QMessageBox.Yes, 
#                                           QtGui.QMessageBox.No)
#        

        if True: #reply == QtGui.QMessageBox.Yes:
            #end threads
            if not(self.__workers_done):
                if self.close_attempts == 0:
                    self.abort_workers() #stop thread, release capture


            #if threads are donw, stop main timer
            if self.__workers_done or self.close_attempts > 500:
                self.main_timer.stop()
                self.save_config()
                event.accept()
            else:
                self.close_attempts += 1
                QtCore.QTimer.singleShot(10, self.close)
                event.ignore() 
            
        else:
            event.ignore()        


    def update_config(self):
        """
        """
        #set
        section = self.set_config_name
        self.config.set(section,'width_thresh_inv',self.ui.width_thresh_inv.isChecked())
        
        self.config.set(section,'vline_right',self.ui.vline_right.value())
        self.config.set(section,'vline_left',self.ui.vline_left.value())
        self.config.set(section,'img_rotation',self.ui.img_rotation.value())
        
        self.config.set(section,'w_y2',self.ui.w_y2.value())
        self.config.set(section,'w_vline_left_border',self.ui.w_vline_left_border.value())
        self.config.set(section,'w_vline_right_border',self.ui.w_vline_right_border.value())
        
        self.config.set(section,'gear_roi_x',self.ui.gear_roi_x.value())
        self.config.set(section,'gear_roi_y',self.ui.gear_roi_y.value())
        self.config.set(section,'gear_roi_size',self.ui.gear_roi_size.value())
        
        self.config.set(section,'fil_roi_y',self.ui.fil_roi_x.value())
        self.config.set(section,'fil_roi_y',self.ui.fil_roi_y.value())
        self.config.set(section,'fil_roi_size',self.ui.fil_roi_size.value())
        
        self.config.set(section,'filament_width',self.ui.filament_width.value())
        
        self.config.set(section,'cal_factor',self.ui.cal_factor.value())
        date_time = self.ui.cal_date.dateTime().toString()
        self.config.set(section,'cal_date',date_time)
        
        self.config.set(section,'mat_colour',self.ui.mat_colour.currentText())
        self.config.set(section,'mat_date_id',self.ui.mat_date_id.text())
        self.config.set(section,'mat_manufacturer',self.ui.mat_manufacturer.text())
        self.config.set(section,'mat_type',self.ui.mat_type.currentText())
        
        self.setup_details['config'] = make_config_dict(self.config)
        #camera settings only set/change in config itself.
        #fps etc.


    def save_config(self,config_filename = None):
        """
        """
        self.update_config()
        #write
        config_filename = config_filename or self.config_filename
        with open(config_filename, 'w') as configfile:
            self.config.write(configfile) 
        #
        append_texteditor(self.ui.text_pipe_out,'Config updated: {}'.format(config_filename))
#        logging.info('config file updated: {}'.format(self.config_filename))
        
    def load_configs(self,add_config_filename = None):
        """
        """
        self.expected_speed = 2 #mm/s
        
        default_config_filename = 'config_vision.cfg'
        self.set_config_name = 'vision'
        
        self.config = configparser.RawConfigParser()
        self.config.optionxform = str
        self.config_filename = default_config_filename
        
        read_result = self.config.read(self.config_filename )
        append_texteditor(self.ui.text_pipe_out,'config loaded: {}'.format(read_result))
        if add_config_filename is not None:
            self.config_filename = add_config_filename
            read_result = self.config.read(self.config_filename)
            append_texteditor(self.ui.text_pipe_out,'add config loaded: {}'.format(read_result))
            
       

        #image processing
        self.ui.width_thresh_inv.setChecked(self.config.getboolean(self.set_config_name,'width_thresh_inv'))
        
        #vertical alingment line - user sets it to filament left and right edge
        self.ui.vline_left.setValue(self.config.getint(self.set_config_name,'vline_left'))
        self.ui.vline_right.setValue(self.config.getint(self.set_config_name,'vline_right'))
        
        #whole image rotation
        self.ui.img_rotation.setValue(self.config.getfloat(self.set_config_name,'img_rotation'))
        
        #ROI - filament width estimation 
        self.ui.w_y2.setValue(self.config.getint(self.set_config_name,'w_y2'))
        self.ui.w_vline_left_border.setValue(self.config.getint(self.set_config_name,'w_vline_left_border'))
        self.ui.w_vline_right_border.setValue(self.config.getint(self.set_config_name,'w_vline_right_border'))
        
        #ROI - filament speed estimation 
        self.ui.fil_roi_x.setValue(self.config.getint(self.set_config_name,'fil_roi_x'))
        self.ui.fil_roi_y.setValue(self.config.getint(self.set_config_name,'fil_roi_y'))
        self.ui.fil_roi_size.setValue(self.config.getint(self.set_config_name,'fil_roi_size'))
        
        #ROI GEAR
        self.ui.gear_roi_x.setValue(self.config.getint(self.set_config_name,'gear_roi_x'))
        self.ui.gear_roi_y.setValue(self.config.getint(self.set_config_name,'gear_roi_y'))
        self.ui.gear_roi_size.setValue(self.config.getint(self.set_config_name,'gear_roi_size'))
        
        #CAL
        self.ui.filament_width.setValue(self.config.getfloat(self.set_config_name,'filament_width'))
        self.ui.cal_factor.setValue(self.config.getfloat(self.set_config_name,'cal_factor'))
        self.ui.calfactor_inv.setValue(1/self.config.getfloat(self.set_config_name,'cal_factor'))
        date_time = self.config.get(self.set_config_name,'cal_date')
        self.ui.cal_date.setDateTime(QtCore.QDateTime.fromString(date_time))
        
        #material
        set_combo_index(self.ui.mat_colour,self.config.get(self.set_config_name,'mat_colour'))
        set_combo_index(self.ui.mat_type,self.config.get(self.set_config_name,'mat_colour'))
        self.ui.mat_date_id.setText(self.config.get(self.set_config_name,'mat_date_id'))
        self.ui.mat_manufacturer.setText(self.config.get(self.set_config_name,'mat_manufacturer'))
        
        #pre-process
        self.rot_apply_flag = round(self.ui.img_rotation.value(),1) != 0
        
        #camera
        self.ui.cam_model.setText(self.config.get('camera','cam_model'))
        self.ui.cam_number.setValue(self.config.getint('camera','cam_number'))
        self.ui.cam_fps.setValue(self.config.getint('camera','cam_fps'))
        
        self.ui.fps.setValue(self.config.getint('camera','cam_fps'))
                                   
        #for writing to configs to results file
        self.setup_details['config'] = make_config_dict(self.config)
        
        
        
    def update_cam_table(self,stream):
        """
        """
        set_table_dictlist(self.ui.cam_table,
                           stream.cam_cv_dict,
                           col_order = ['id','CV','value'],
                           header_upper = True)
        
    def updated_std_width(self,new_value = None):
        """
        """
        if new_value is not None:
            self.setup_details['user_filament_width'] = new_value
#        append_texteditor(self.ui.text_pipe_in,'update config')
        self.update_config()
        self.setup_details['config'] = make_config_dict(self.config)
        self.pipe_out.send(self.setup_details)
        
        
    def set_update_areas(self):
        """
        """
        self.update_areas = True
        

    def areas_setup(self):
        """
        """
        self.update_config()
        
        #set if any details must be shown of specific ROI
        self.show_gear = self.video_mode and self.app_settings.get('show_gear',False)
        self.show_width = self.video_mode and self.app_settings.get('show_width',False)
#        self.show_post = False
        self.show_filament_speed =  self.video_mode and self.app_settings.get('show_filament_speed',False)
        
        collect_data_all = False
        
        self.areas = {}
        
        if self.frame_extruder is None:
            print('camera not connected')
            return False

        #----------------------------------------------------------------------
        #FILAMENT WIDTH ESTIMATION - PRE-GEAR AREA
        vline_left = self.ui.vline_left.value()
        vline_right = self.ui.vline_right.value()
        w_vline_left_border =  self.ui.w_vline_left_border.value()
        w_vline_right_border =  self.ui.w_vline_right_border.value()
        
        x_start = vline_left - w_vline_left_border#20
        y_start = self.ui.w_y2.value() - 40
        x_end = vline_right + w_vline_right_border#15
        method_roi_pnts = (x_start,y_start, x_end, self.ui.w_y2.value())
        
        method_kwargs = {}
        method_kwargs.update(self.setup_details['config']['vision'])
        method_kwargs.update(self.setup_details['config']['camera'])
        method_kwargs['width_default'] = vline_right - vline_left
        method_kwargs['w_vline_left_border'] = w_vline_left_border
        method_kwargs['w_vline_right_border'] = w_vline_right_border
        method_kwargs['draw_on'] = self.show_width
        method_kwargs['plot_on'] = self.show_width
        
        
        

        width_roi = ROIA.FilamentWidthMethod(self.frame_extruder,method_roi_pnts,
                                             name_id = 'width_1',
                                             collect_on = collect_data_all,
                                             **method_kwargs)
        
        
        width_roi.roi_colour = cv_colors['yellow']
        
        #----------------------------------------------------------------------
        #FILAMENT SPEED - POST GEAR
        x_start = self.ui.fil_roi_x.value() #left top pnt
        y_start = self.ui.fil_roi_y.value()   #left top pnt
        x_end = x_start + self.ui.fil_roi_size.value()
        y_end = y_start + self.ui.fil_roi_size.value()*2
        method_roi_pnts = (x_start,y_start,x_end,y_end)
        
        method_kwargs = {}
        method_kwargs.update(self.setup_details['config']['vision'])
        method_kwargs.update(self.setup_details['config']['camera'])
        method_kwargs['draw_on'] = self.show_filament_speed
        method_kwargs['plot_on'] = self.show_filament_speed
        
        filspeed_roi = ROIA.SmallAreaDenseMethod(self.frame_extruder,method_roi_pnts,
                                             name_id = 'FsDM',
                                             collect_on = collect_data_all,
                                             **method_kwargs)
                              
        
        filspeed_roi.roi_colour = cv_colors['purple']

        #----------------------------------------------------------------------
        #GEAR speed
        x_start = self.ui.gear_roi_x.value() #left top pnt
        y_start = self.ui.gear_roi_y .value()  #left top pnt
        x_end = x_start + self.ui.gear_roi_size.value()
        y_end = y_start + self.ui.gear_roi_size.value()*2
        method_roi_pnts = (x_start,y_start,x_end,y_end)

        method_kwargs = {}
        method_kwargs.update(self.setup_details['config']['vision'])
        method_kwargs.update(self.setup_details['config']['camera'])
        
        method_kwargs['draw_on'] = self.show_gear
        method_kwargs['plot_on'] = self.show_gear

        gear_roi = ROIA.SmallAreaDenseMethod(self.frame_extruder,method_roi_pnts,
                                             name_id = 'GsDM',
                                             collect_on = collect_data_all,
                                             **method_kwargs)
                
                
        
        gear_roi.roi_colour = cv_colors['orange']

        #----------------------------------------------------------------------
        self.areas['width'] = width_roi
        self.areas['filament_speed'] = filspeed_roi
        self.areas['gear'] = gear_roi
        


    def start_cal(self):
        """
        start the calibration process
        
        """
        self.cal_collect_width = True  #start collecting in post_process
        self.cal_width_history = []    #reset
        self.ui.tabWidget.setEnabled(False)

    def set_calfactor(self):
        """
        Measure filament width, set spinbox to value, 
        then call by clicking calibrate button.
        
        """
        prev_cal = self.ui.cal_factor.value()
        user_w_pxl = self.ui.vline_right.value() - self.ui.vline_left.value()
        width_pxl = np_ave(self.cal_width_history) # self.areas['width'].width_value
        new_cal = round(self.ui.filament_width.value()/float(width_pxl),4) #mm/pxl
        self.ui.cal_factor.setValue(new_cal)
        
        append_texteditor(self.ui.text_pipe_out,'Calibrate: user_W: {} vs detected: {:.4f} pxl'.format(user_w_pxl,width_pxl))
        append_texteditor(self.ui.text_pipe_out,'Calibrate: prev cal_factor:  {:.4f} mm/pxl'.format(prev_cal))
        
        #update vals
        self.set_cal_values()
        
        #set cal date
        date_time =  QtCore.QDateTime.currentDateTime()
        self.ui.cal_date.setDateTime(date_time)
        
        #return to main tab
        self.ui.tabWidget.setCurrentIndex(0)

    
    def set_cal_values(self):
        """
        """
        self.convert_width = self.ui.cal_factor.value()
        self.convert_speed = self.convert_width * self.stream_fps
        self.ui.calfactor_inv.setValue(1/self.convert_width )
        self.frame_result['cal_factor'] = self.convert_width
        #done
        append_texteditor(self.ui.text_pipe_out,'Calibrate: set cal_factor:  {:.4f} mm/pxl'.format(self.ui.cal_factor.value()))
        append_texteditor(self.ui.text_pipe_out,'Calibrate: set convert_speed:  {:.4f} mm/pxl frame/s'.format(self.convert_speed))
                

    def pre_process_whole(self):
        """
        """
        if self.rot_apply_flag:
            try:
                self.frame_extruder = cv2.warpAffine(self.frame_extruder, 
                                                    self.rot_mat, self.rot_wh,
                                                    flags = self.rot_flags)
            except AttributeError:
                self.determine_rot_params()
                                                
    def determine_rot_params(self):
        """
        only determine once newly/first selected angle
        """               
        angle = self.ui.img_rotation.value()
        self.rot_apply_flag = round(angle,1) != 0
        self.frame_extruder,self.rot_mat,self.rot_wh,self.rot_flags  = rotate_image(self.frame_extruder,angle = angle,
                                                                                     return_param = True)

    def process_areas(self,store_values = False):
        """
        """
        for area in self.areas.values():
           
            roi = self.frame_extruder[area.roi_y1:area.roi_y2,area.roi_x1:area.roi_x2]
            area.apply(roi,self.frame_id)
            
#            if area.id_type != 'gear':
#                area.apply_method(roi,self.frame_id,store_values)
            
        if self.ui.report_results.isChecked():
            self.frame_result['time_stamp'] = self.current_time.currentTime().toString('hh:mm:ss.zzz')
            self.frame_result['gear_speed'] = self.areas['gear'].speed_step_current #pxl/frame
            self.frame_result['filament_speed'] = self.areas['filament_speed'].speed_step_current #pxl/frame
            self.frame_result['filament_width'] = self.areas['width'].width_value #pxl
            self.frame_result['time_elasped'] = self.time_passed_ms #msec
            self.frame_result['cal_factor'] = self.convert_width #mm/pxl
            self.frame_result['frame_id'] = self.frame_id #
            self.pipe_out.send(self.frame_result)
            
 

    def post_process(self):
        """
        draw ROI rectangles and some text on frame
        """
        #----------------------------------------------------------
        #draw ROI extruder
#        colour = (0,0,255)
        if not(self.measure_active):
            for area in self.areas.values():
                cv2.rectangle(self.frame_extruder, area.roi_pnt1, area.roi_pnt2,
                              area.roi_colour, 1)
            
            #filament width - manual lines
            cv2.line(self.frame_extruder,(self.ui.vline_right.value(),0),
                     (self.ui.vline_right.value(),self.frame_extruder.shape[0]),cv_colors['green'])
            
            cv2.line(self.frame_extruder,(self.ui.vline_left.value(),0),
                     (self.ui.vline_left.value(),self.frame_extruder.shape[0]),cv_colors['blue'])
    #        cv2.line(self.frame_extruder,(self.vline_left,25),
    #                 (self.vline_right,25),(255,0,0))
            
            #display detected width        
    #        if self.state == 0: #setup mode
            x_offset = self.areas['width'].roi_pnt1[0]
            x_left = int(round(self.areas['width'].width_left_pnt + x_offset))
            x_right = int(round(self.areas['width'].width_right_pnt + x_offset))
            y1 = self.areas['width'].roi_pnt1[1]
            y2 = self.areas['width'].roi_pnt2[1]
            
            cv2.line(self.frame_extruder,(x_left,y1),
                 (x_left,y2),cv_colors['red'])
                 
            cv2.line(self.frame_extruder,(x_right,y1),
                 (x_right,y2),cv_colors['red'])
                     
            
            if self.cal_collect_width:
                self.cal_width_history.append(self.areas['width'].width_value)
                if len(self.cal_width_history) > self.cal_w_len:
                    self.ui.tabWidget.setEnabled(True)
                    self.cal_collect_width = False
                    self.set_calfactor()
                else:
                    text = 'calibrating: {}/{} - {}'.format(len(self.cal_width_history),self.cal_w_len,self.areas['width'].width_value)
                    self.pipe_out.send(text)
                    
                    
    
    #        fps_str = 'FPS: {:.1f}'.format(1./self.time_step)
    ##        fps_str = 'FPS: {:.3f}'.format(1./np.average(self.time_steps[-15:]))
    #        draw_str(self.frame_extruder,(50,40),fps_str)
    
           #gear 
#           cv2.circle(self.frame_extruder, (self.ui.x_gear_c.value(), self.ui.y_gear_c.value()), 
#           self.areas['gear'].cr_od, cv_colors['cyan'])
            
            #cx and cx        
#            cv2.line(self.frame_extruder,
#                     (self.ui.x_gear_c.value(),0),
#                     (self.ui.x_gear_c.value(),self.frame_extruder.shape[0]),
#                     cv_colors['cyan'])
#            
#            cv2.line(self.frame_extruder,
#                     (0,self.ui.y_gear_c.value()),
#                     (self.frame_extruder.shape[1],self.ui.y_gear_c.value()),
#                     cv_colors['cyan'])
#            
#            #gear mask
#            cv2.circle(self.frame_extruder, (self.ui.x_gear_c.value(), self.ui.y_gear_c.value()), 
#                       self.areas['gear'].cr_max, cv_colors['red'])
#            cv2.circle(self.frame_extruder, (self.ui.x_gear_c.value(), self.ui.y_gear_c.value()), 
#                       self.areas['gear'].cr_min, cv_colors['red'])
        #----
        
        self.ui.gear_speed.setValue(self.areas['gear'].speed_step_current*self.convert_speed)
        self.ui.filament_speed.setValue(self.areas['filament_speed'].speed_step_current*self.convert_speed)
        self.ui.measured_width.setValue(self.areas['width'].width_value*self.convert_width)
#        self.ui.missed_frames.setValue(self.missed_frames)
        
        if self.time_passed_ms != 0:
            self.ui.fps.setValue(1/self.time_passed_ms*1000.)
            
        
        
        #SHOW
        self.microscope_display.pixmap = converter(self.frame_extruder)
        self.update()

    def load_video_from_file(self,video_fullname = None):
        """
        Called by pb clicked:
            disconnect current camera if connected
            enalbe controls
            setup frame numbers and slider.
            
        Once video is running, camera view cannot be restored. 
        Only through restarting app.
        """
        self.main_timer.stop()
        
        if self.stream is not None:
            self.stream.stop() #stop thread, release capture
            self.stream.new_frame.disconnect()
        
        if video_fullname is None:
            video_fullname = load_file_name(self,type_filter  = 'Video File (*.AVI)')
        
        if video_fullname is not None:
            append_texteditor(self.ui.text_pipe_out,'load video: {}'.format(video_fullname))
            
            self.ui.frame_next.setEnabled(True)
            self.ui.frame_prev.setEnabled(True)
            self.ui.frame_number.setEnabled(True)
            self.ui.play_video.setEnabled(True)
            
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
                
            self.video_capture = cv2.VideoCapture(video_fullname)
            retval, frame = self.video_capture.read()
            self.frame_extruder = frame
          
            filename = os.path.split(video_fullname)[-1]
            self.video_filename = os.path.splitext(filename)[0]
            if retval:
                
                self.ui.fps.setValue(self.video_capture.get(cv2.CAP_PROP_FPS))
                self.stream_fps = self.ui.fps.value()
                self.time_passed_ms = (1/self.ui.fps.value())*1000
                self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                append_texteditor(self.ui.text_pipe_out,'next_video shape: {}, num frames {}'.format(frame.shape,self.num_frames))
                
                self.ui.frame_number.setMaximum(self.num_frames)
                #try get config
                file_search = os.path.splitext(video_fullname)[0]
                video_config_filename = get_files_like(folder = None,
                                                       filename_contains =file_search,
                                                       ext = 'cfg')
                try:
                    video_config_filename = video_config_filename[0]
                    append_texteditor(self.ui.text_pipe_out,'load video config: {}'.format(video_config_filename))
                    append_texteditor(self.ui.text_pipe_out,'video config: {}'.format(video_config_filename))
                    self.load_configs(video_config_filename)
                    self.areas_setup()
                    self.set_cal_values()
                    
                    
                except IndexError:
                    append_texteditor(self.ui.text_pipe_out,'no config file found: {}'.format(video_config_filename)) #
            else:
                append_texteditor(self.ui.text_pipe_out,'frame is None: {}'.format(video_fullname))
                
                
        #---- end if video_filename is not None   
        
        self.main_timer.start(self.main_timer_rate)


    def video_new_frame(self):
        """
        get new frame
        """
        if self.video_capture is not None:
            cont_mode = self.ui.play_video.isChecked()
            if cont_mode:
                self.new_frame_next()
            if self.prev_frame != self.frame_id or self.update_areas:
                self.prev_frame = self.frame_id
                self.frame_changed = False
                ret, self.frame_extruder = self.video_capture.read()
                if self.frame_extruder is not None:
                    return True
                else:
                    pass

        return False                                                                                  

    def new_frame_trackbar(self,new_frame_id):
        """
        """
        self.frame_id = new_frame_id
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)        

    def new_frame_next(self):
        """
        """
        if self.frame_id + 1 < self.num_frames:
            new_frame = self.frame_id + 1
        else:
            new_frame = self.frame_id
            self.ui.play_video.setChecked(False) #end of file
#        self.frame_id = self.frame_id + 1 if self.frame_id + 1 < self.num_frames else self.frame_id
#        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
        
        self.ui.frame_number.setValue(new_frame)
        
                                  

    def new_frame_prev(self):
        """
        """
        new_frame= self.frame_id -1 if self.frame_id - 1 >= 0 else 0
        self.ui.frame_number.setValue(new_frame)
#        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
    
    def record_video_start_stop(self,new_state = False):
        """
        Video recording, called from checkbox click
        
        Start, continue or pause video recording.
        """   
        
        if new_state and not(self.video_mode):# start recording
            ui_filename = self.ui.record_filename.text()
            if ui_filename == '' or self.record_filename_prev == ui_filename:
                #create new filename
                job_name = 'Test' 
                self.record_filename = make_date_filename(job_name,ext = 'avi',
                                                          folder = self.data_folder)
            else:
                self.record_filename = ui_filename
                if self.record_filename.find('.avi') < 0:
                    self.record_filename = self.record_filename + '.avi'
                    
            
            self.video_details['video_filename'] = self.record_filename
            self.pipe_out.send(self.video_details)                  
                              
            height, width, layers =  self.frame_extruder.shape
            
    #        fourcc = cv2.cv.CV_FOURCC(*'XVID')  # 
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.record_file = cv2.VideoWriter(self.record_filename,fourcc, 
                                               self.record_fps, 
                                              (width,height))
            
            self.ui.record_state.setText('Recording')
            self.ui.record_filename.setText(self.record_filename)
            self.ui.record_start_new.setEnabled(True)
            self.ui.record_save.setEnabled(True)
            self.record_filename_prev = self.record_filename 
            self.recording = True
            

        elif not(self.video_mode): #pause
            self.recording = False
            self.ui.record_state.setText('Paused')
        else:
            self.recording = False
    
    def record_video_save(self):
        """
        stop and save
        """
        self.recording = False
        self.ui.record_video.setChecked(False)
        if self.record_file is not None:
            #release video file - save
            self.record_file.release()
            self.ui.record_state.setText('Saved')
            self.ui.record_start_new.setEnabled(False)
            self.record_file = None
            append_texteditor(self.ui.text_pipe_out,'Video saved: {}'.format(self.record_filename))
            #save config with the same namve
            configname = self.record_filename.split('.avi')[0]
            configname += '.cfg'
            self.save_config(configname)
            
            
    
    def record_start_new(self):
        """
        """
        self.record_video_save()
        self.ui.record_video.setChecked(True)
        self.record_video_start_stop(True)

#==============================================================================
#==============================================================================
#==============================================================================
class VisionProcess(Process):
    """
    """
    def __init__(self,pipe_2_vision_out,pipe_from_vision_in,**kwargs):
        super().__init__()
        self.pipe_2_vision_out,self.pipe_from_vision_in = pipe_2_vision_out,pipe_from_vision_in
        self.app_settings = kwargs

    def run(self):
        """
        start application
        """
        self.app = QtWidgets.QApplication([])
        self.vision = ExtruderCam(self.app,self.pipe_2_vision_out,self.pipe_from_vision_in,**self.app_settings)    
        self.vision.show()
        self.app.exec_()
#==============================================================================
#==============================================================================
#==============================================================================
class EmptyPipe():
    def send(self,text):
        print(text)
    def poll(self):
        return False
    def recv(self):
        return ''
      
#==============================================================================
#==============================================================================
#==============================================================================    
    
def main():
    import sys
#    print('main')
#    print(QtCore.QT_VERSION)
    if QtCore.QT_VERSION >= 0x50501:
        import traceback
        def excepthook(type_, value, traceback_):
            traceback.print_exception(type_, value, traceback_)
            QtCore.qFatal('')
        sys.excepthook = excepthook
   
    pipe_2_vision_out = EmptyPipe()
    pipe_from_vision_in = EmptyPipe()
    
    app = QtWidgets.QApplication(sys.argv)

    mc = ExtruderCam(app,
                     pipe_2_vision_out,
                     pipe_from_vision_in,
                     do_profiling = 0,
                     load_video_def = False,
                     show_gear = False)
    mc.show()
    sys.exit(app.exec_())    

    
        
#===============================================================================    
if __name__ == '__main__':
    main()


    
    