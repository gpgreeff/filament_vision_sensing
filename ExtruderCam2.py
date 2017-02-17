# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:23:32 2017

@author: Greeff



How to convert main.ui to main_ui.py file (a or b):
a)  from command line:
    pyuic5 -x main.ui -o main_ui.py
    pyuic5.bat:
    @"C:/Program Files/Anaconda3\python.exe" -m PyQt5.uic.pyuic %1 %2 %3 %4 %5 %6 %7 %8 %9
    
b)  form_class = uic.loadUiType("main.ui")[0]

"""
__version__ = "0.5"
__license__ = "MIT"

from multiprocessing import Process,Pipe
import cv2
import configparser
import os

#import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets


from UI.ExtruderCam2_ui import Ui_MainWindow
from ToolsCV import CameraLabel,converter,rotate_image,cv_colors,VideoStreamWorker
from ROIA import ROIA
from QtLoadSaveObjects import load_file_name,get_files_like,set_combo_index
from QtLoadSaveObjects import make_config_dict,make_date_filename,append_texteditor
from TableTools import set_table_dictlist



'''
-------------------------------------------------------------------------------
CamInterface
-------------------------------------------------------------------------------
''' 
class ExtruderCam(QtWidgets.QMainWindow):
    '''
    '''
    def __init__(self,pipe_in = None,pipe_out = None,**kwargs):
        '''
        '''
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
        self.missed_frames = 0
                              
        
        
        #result reporting
        self.frame_result = {}
        self.frame_result['name'] = 'frame_result'
        self.frame_result['time_stamp'] = ''
        self.frame_result['gear_speed'] = 0 #pxl/frame
        self.frame_result['filament_speed'] = 0 #pxl/frame
        self.frame_result['filament_width'] = 0 #pxl
        self.frame_result['time_elasped'] = 0 #sec
        self.frame_result['cal_factor'] = 1.0 #mm/pxl
        self.frame_result['frame_id'] = 1.0 #
        
        #setup reporting
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
        
        #TIMERS
        self.current_time = QtCore.QTime()
        self.main_timer_rate = 50 #ms
        self.main_timer = QtCore.QTimer(self)
        self.main_timer.timeout.connect(self.main_task_loop)
        
        #source
        self.video_capture = None #from a video file
        self.stream = None #from camera
        self.frame_extruder = None
        
        if kwargs.get('load_video_def',False):
            self.video_mode = True
            self.load_video_from_file(video_fullname = 'data\Test_2017_02_17__08_45_46.avi')
            
        else:
            self.video_mode = False
#            self.stream_to_pipe_in,self.stream_to_pipe_out = Pipe()
#            self.stream_from_pipe_in,self.stream_from_pipe_out = Pipe()
#            self.stream = VideoStreamWorker(self.stream_to_pipe_out,self.stream_from_pipe_in,**self.setup_details['config']['camera'])
            self.stream = VideoStreamWorker(**self.setup_details['config']['camera'])
            self.stream_fps = self.stream.fps
            self.frame_extruder = self.stream.read_frame()
            self.update_cam_table()
        
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
        #camera setup
        self.ui.img_rotation.valueChanged.connect(self.set_update_areas)
        self.ui.vline_left.valueChanged.connect(self.set_update_areas)
        self.ui.vline_right.valueChanged.connect(self.set_update_areas)
        self.ui.x_gear_c.valueChanged.connect(self.set_update_areas)
        self.ui.x_gear_teeth.valueChanged.connect(self.set_update_areas)
        self.ui.y_gear_c.valueChanged.connect(self.set_update_areas)
        self.ui.y_post_start.valueChanged.connect(self.set_update_areas)
        self.ui.y_pre_end.valueChanged.connect(self.set_update_areas)
        self.ui.mat_colour.currentIndexChanged.connect(self.set_update_areas)
        self.ui.od_resize_factor.valueChanged.connect(self.set_update_areas)
        
        self.ui.save_config.clicked.connect(self.save_config)
        
        #calibration
#        self.ui.filament_width.valueChanged.connect(self.updated_std_width)
        self.ui.calibrate.clicked.connect(self.set_calfactor)
  
        #video playback
        self.ui.load_video.clicked.connect(self.load_video_from_file)
        self.ui.frame_number.valueChanged.connect(self.new_frame_trackbar)
        self.ui.frame_next.clicked.connect(self.new_frame_next)
        self.ui.frame_prev.clicked.connect(self.new_frame_prev)
        
        #video recording
        self.ui.record_video.clicked.connect(self.record_video_start_stop)
        self.ui.record_save.clicked.connect(self.record_video_save)
        self.ui.record_start_new.clicked.connect(self.record_start_new)
        
        #frame capture IO
        if self.stream is not None:
#            self.stream.new_frame.connect(self.process_new_frame)
            self.stream.new_frame_rdy.connect(self.new_frame_rdy)
        
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
        
        #START
        current_time_str = self.current_time.currentTime().toString('hh:mm:ss.zzz')
        append_texteditor(self.ui.text_pipe_out,'start time_stamp: {}'.format(current_time_str))
        if self.stream is not None:
            self.stream.start()
#            self.stream_to_pipe_in.send(1)
            
        self.main_timer.start(self.main_timer_rate)
        self.pipe_out.send('vision started')
        self.updated_std_width(self.ui.filament_width.value())


#------------------------------------------------------------------------------
    def new_frame_rdy(self):
        '''
        new frame on pipe
        '''
#        print('new_frame_rdy rcvd')
        new_data = self.stream.get_frame_with_rate()
#        print('new_data',len(new_data))
        self.frame_extruder,self.time_passed_ms,self.frame_id,self.missed_frames = new_data
        self.process_new_frame()
        self.stream.frame_processed = True
#        self.stream_to_pipe_in.send(1)
        
#------------------------------------------------------------------------------
    def process_new_frame(self,new_frame = None):
        '''
        Called when camera processed a new frame,
        or by main loop during video file review
        
        '''    
        if new_frame is not None:
            self.frame_extruder = new_frame[0]
            self.time_passed_ms = new_frame[1]
            self.frame_id = new_frame[2] or self.frame_id
        
        if self.recording:
            self.record_file.write(self.frame_extruder)
            
        
        if self.update_areas:
            self.areas_setup()
            self.update_areas = False
#            print('update_areas')
#            
        if self.process_active:
            self.pre_process_whole()
            self.process_areas()
        self.new_frame_processed = True
        
#------------------------------------------------------------------------------        
    def main_task_loop(self):
        '''
        '''
        #replay recorded video
        if self.video_mode: 
            if self.video_new_frame():
                self.process_new_frame([self.frame_extruder,self.time_passed_ms,None])
                
        #update display
        if self.new_frame_processed:
            self.new_frame_processed = False
            self.show_setup_roi()
       

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
                    
#------------------------------------------------------------------------------
    def closeEvent(self, event):
        '''
        Check if exit required, then save configs and close connections and 
        other widgets. Important to release caputre.
        
        '''
#        quit_msg = 'Quit?'
#        reply = QtGui.QMessageBox.question(self, 'Exit',quit_msg,
#                                           QtGui.QMessageBox.Yes, 
#                                           QtGui.QMessageBox.No)
#        
#       #reply == QtGui.QMessageBox.Yes:
    
        if True:
            if self.stream is not None:
                if self.close_attempts == 0:
                    self.stream.stop() #stop thread, release capture
#                    self.stream_from_pipe_in.close()
#                    self.stream_to_pipe_in.close()

                #thread seems to need some time to stop-
                thread_done = not(self.stream.isRunning()) and self.stream.isFinished()
    #            print(thread_done,self.close_attempts)
            else:
                thread_done = True
                
            if thread_done or self.close_attempts > 500:
                self.main_timer.stop()
                self.save_config()
                event.accept()
            else:
                self.close_attempts += 1
                QtCore.QTimer.singleShot(10, self.close)
                event.ignore() 
            
        else:
            event.ignore()        

#------------------------------------------------------------------------------
    def update_config(self):
        '''
        '''
        #set
        section = self.set_config_name
        self.config.set(section,'width_thresh_inv',self.ui.width_thresh_inv.isChecked())
        self.config.set(section,'vline_right',self.ui.vline_right.value())
        self.config.set(section,'vline_left',self.ui.vline_left.value())
        self.config.set(section,'angle_extruder',self.ui.img_rotation.value())
        self.config.set(section,'y_pre_end',self.ui.y_pre_end.value())
        self.config.set(section,'y_post_start',self.ui.y_post_start.value())
        self.config.set(section,'x_gear_c',self.ui.x_gear_c.value())
        self.config.set(section,'y_gear_c',self.ui.y_gear_c.value())
        self.config.set(section,'od_resize_factor',self.ui.od_resize_factor.value())
        self.config.set(section,'x_gear_teeth',self.ui.x_gear_teeth.value())
        self.config.set(section,'filament_width',self.ui.filament_width.value())
        self.config.set(section,'w_ave_range',self.ui.w_ave_range.value())
        self.config.set(section,'cal_factor',self.ui.cal_factor.value())
        date_time = self.ui.cal_date.dateTime().toString()
        self.config.set(section,'cal_date',date_time)
        self.config.set(section,'mat_colour',self.ui.mat_colour.currentText())
        self.config.set(section,'mat_date_id',self.ui.mat_date_id.text())
        self.config.set(section,'mat_manufacturer',self.ui.mat_manufacturer.text())
        self.config.set(section,'mat_type',self.ui.mat_type.currentText())
        
        #camera settings only set/change in config itself.
        #fps etc.

#------------------------------------------------------------------------------
    def save_config(self,config_filename = None):
        '''
        '''
        self.update_config()
        #write
        config_filename = config_filename or self.config_filename
        with open(config_filename, 'w') as configfile:
            self.config.write(configfile) 
        #
        append_texteditor(self.ui.text_pipe_out,'Config updated: {}'.format(config_filename))
#        logging.info('config file updated: {}'.format(self.config_filename))
#------------------------------------------------------------------------------        
    def load_configs(self,add_config_filename = None):
        '''
        '''
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
            
       

        #--
        self.pitch_multiplier = self.config.getfloat(self.set_config_name,'pitch_multiplier') #number of gear pitches in post y direction
        self.def_cal_factor = self.config.getfloat(self.set_config_name,'def_cal_factor') #
        

        #--
        self.ui.w_ave_range.setValue(self.config.getint(self.set_config_name,'w_ave_range'))
        self.ui.width_thresh_inv.setChecked(self.config.getboolean(self.set_config_name,'width_thresh_inv'))
        
        self.ui.vline_left.setValue(self.config.getint(self.set_config_name,'vline_left'))
        self.ui.vline_right.setValue(self.config.getint(self.set_config_name,'vline_right'))
        
        self.ui.img_rotation.setValue(self.config.getfloat(self.set_config_name,'img_rotation'))
        

        #ROI POST AND PRE
        # end of 'pre', just left & above bearing
        self.ui.y_pre_end.setValue(self.config.getint(self.set_config_name,'y_pre_end'))
        # start of 'post', just right & below of gear
        self.ui.y_post_start.setValue(self.config.getint(self.set_config_name,'y_post_start'))
        

        #ROI GEAR
        self.ui.x_gear_c.setValue(self.config.getint(self.set_config_name,'x_gear_c'))
        self.ui.y_gear_c.setValue(self.config.getint(self.set_config_name,'y_gear_c'))
        self.ui.x_gear_teeth.setValue(self.config.getint(self.set_config_name,'x_gear_teeth'))
        
        self.ui.od_resize_factor.setValue(self.config.getfloat(self.set_config_name,'od_resize_factor'))
        
        
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
        
        
#------------------------------------------------------------------------------        
    def update_cam_table(self):
        '''
        '''
        set_table_dictlist(self.ui.cam_table,
                           self.stream.cam_cv_dict,
                           col_order = ['id','CV','value'],
                           header_upper = True)
#------------------------------------------------------------------------------        
    def updated_std_width(self,new_value = None):
        '''
        '''
        if new_value is not None:
            self.setup_details['user_filament_width'] = new_value
#        append_texteditor(self.ui.text_pipe_in,'update config')
        self.update_config()
        self.setup_details['config'] = make_config_dict(self.config)
        self.pipe_out.send(self.setup_details)
        
#------------------------------------------------------------------------------        
    def set_update_areas(self):
        '''
        '''
        self.update_areas = True
        
#------------------------------------------------------------------------------
    def areas_setup(self):
        '''
        '''
        
            
        #set if any details must be shown of specific ROI
        self.show_gear = self.video_mode and self.app_settings.get('show_gear',False)
        self.show_width = self.video_mode and self.app_settings.get('show_width',False)
#        self.show_post = False
        self.show_filament_speed =  self.video_mode and self.app_settings.get('show_filament_speed',False)
        self.areas = {}
        
        if self.frame_extruder is None:
            print('camera not connected')
            return False
        #----------------------------------------------------------------------
        #CAL

        #----------------------------------------------------------------------
        #PRE GEAR AREA- get filament width
        width_default = self.ui.vline_right.value() - self.ui.vline_left.value()
        roi_width_pnts = (self.ui.vline_left.value() - 20,self.ui.y_pre_end.value() - 40, 
                   self.ui.vline_right.value() + 15, self.ui.y_pre_end.value())
        x1,y1,x2,y2 = roi_width_pnts
        roi = self.frame_extruder[y1:y2,x1:x2]
        width = ROIA('width',roi,
                     roi_pnts = roi_width_pnts,
                     plot_on = self.show_width,
                     width_default = width_default,
                     material = self.ui.mat_colour.currentText().lower())
        
        width.roi_colour = cv_colors['yellow']
        
        #----------------------------------------------------------------------
        #POST GEAR profile
#        pitch = np.pi*8./24
#        pitch_pxl = pitch/self.def_cal_factor
#        profile_h = int(round(pitch_pxl*self.pitch_multiplier))
#        self.y_post_end = self.y_post_start+profile_h
#        roi_post = (self.vline_left - 15,self.y_post_start, 
#                    self.vline_right + 20, self.y_post_end)
#        post = ROIA(2,'post',roi_pnts = roi_post,plot_on = self.show_post,
#                        cal_x = self.cal_factor,w_ave_range = self.w_ave_range,
#                        width_thresh_inv  = self.width_thresh_inv)
#        post.ave_method_use_green = True
#        post.roi_colour = (0,255,255)
        
        
        #----------------------------------------------------------------------
        #POST GEAR - filament speed
        self.y_post_end = 480-10
        roi_filament_speed = (self.ui.vline_left.value() - self.ui.x_gear_teeth.value(),self.ui.y_post_start.value(), 
                              self.ui.vline_right.value() + 10, self.y_post_end)
        x1,y1,x2,y2 = roi_filament_speed
        roi = self.frame_extruder[y1:y2,x1:x2]
        filament_speed = ROIA('filament_speed',roi,
                              roi_pnts = roi_filament_speed,
                              plot_on = self.show_filament_speed,
                              material = self.ui.mat_colour.currentText().lower())
        
        filament_speed.roi_colour = cv_colors['purple']

        #----------------------------------------------------------------------
        #GEAR
        self.gear_cr_od = int(round(self.ui.od_resize_factor.value() * 4.0 / self.def_cal_factor)) #8 mm gear radius
        diameter_border = 5 # 10
#        x_start = max(0,self.x_gear_c - self.gear_cr_od - diameter_border)
        x_start = max(0,self.ui.x_gear_c.value()) #left top pnt
        relative_cx = 0 #self.x_gear_c
        y_start = max(0,self.ui.y_gear_c.value() - self.gear_cr_od - diameter_border) #left top pnt
        
        x_end = min(640,self.ui.x_gear_c.value() + self.gear_cr_od + diameter_border) #right bottom pnt
        y_end = min(480,self.ui.y_gear_c.value() + self.gear_cr_od + diameter_border) #right bottom pnt
        
#        x_start = int(round((x_start + x_end)/2)) - 20
#        x_end -= 20             
#        y_end = int(round((y_start + y_end)/2)) #- 80

#        x_start = 0
#        x_end -= 20
#        y_start = self.ui.y_gear_c.value() + self.gear_cr_od - 80
#        y_end =   self.y_post_end - 30                                      
        
        roi_gear = (x_start, y_start,x_end,y_end)
        x1,y1,x2,y2 = roi_gear
        roi = self.frame_extruder[y1:y2,x1:x2]
        gear = ROIA('gear',roi,
                    roi_pnts = roi_gear,plot_on = self.show_gear,
                    cx = relative_cx,
                    cy = self.ui.y_gear_c.value(), 
                    cr_od = self.gear_cr_od)
        
        gear.roi_colour = cv_colors['orange']

        #----------------------------------------------------------------------
        spike_threshold = round(self.expected_speed/(self.ui.cal_factor.value() * self.stream_fps) * 2.,1)
        filament_speed.speed_spike_threshold  = spike_threshold
        
        
        #----------------------------------------------------------------------
        self.areas['width'] = width
#        self.areas['post'] = post
        self.areas['filament_speed'] = filament_speed
        self.areas['gear'] = gear
        

#------------------------------------------------------------------------------    
    def set_cal_values(self):
        '''
        '''
        self.convert_width = self.ui.cal_factor.value()
        self.convert_speed = self.convert_width * self.stream_fps
        self.ui.calfactor_inv.setValue(1/self.convert_width )
        self.frame_result['cal_factor'] = self.convert_width
        #done
        append_texteditor(self.ui.text_pipe_out,'Calibrate: set cal_factor:  {:.4f} mm/pxl'.format(self.ui.cal_factor.value()))
        append_texteditor(self.ui.text_pipe_out,'Calibrate: set convert_speed:  {:.4f} mm/pxl'.format(self.convert_speed))
        
#------------------------------------------------------------------------------
    def set_calfactor(self):
        '''
        Measure filament width, set spinbox to value, 
        then call by clicking calibrate button.
        
        '''
        prev_cal = self.ui.cal_factor.value()
        user_w_pxl = self.ui.vline_right.value() - self.ui.vline_left.value()
        width_pxl = self.areas['width'].width_value
        new_cal = round(self.ui.filament_width.value()/float(width_pxl),4) #mm/pxl
        self.ui.cal_factor.setValue(new_cal)
        
        append_texteditor(self.ui.text_pipe_out,'Calibrate: user_W: {} vs detected: {}'.format(user_w_pxl,width_pxl))
        append_texteditor(self.ui.text_pipe_out,'Calibrate: prev cal_factor:  {:.4f} mm/pxl'.format(prev_cal))
        
        self.set_cal_values()
        self.ui.tabWidget.setCurrentIndex(0)
        
#------------------------------------------------------------------------------
    def pre_process_whole(self):
        '''
        '''
        if self.rot_apply_flag:
            try:
                self.frame_extruder = cv2.warpAffine(self.frame_extruder, 
                                                    self.rot_mat, self.rot_wh,
                                                    flags = self.rot_flags)
            except AttributeError:
                self.determine_rot_params()
#------------------------------------------------------------------------------                                                
    def determine_rot_params(self):
        '''
        only determine once newly/first selected angle
        '''               
        angle = self.ui.img_rotation.value()
        self.rot_apply_flag = round(angle,1) != 0
        self.frame_extruder,self.rot_mat,self.rot_wh,self.rot_flags  = rotate_image(self.frame_extruder,angle = angle,
                                                                                     return_param = True)
#------------------------------------------------------------------------------
    def process_areas(self,store_values = False):
        '''
        '''
        for area in self.areas.values():
            x1,y1,x2,y2 = area.roi_pnts
            roi = self.frame_extruder[y1:y2,x1:x2]
            area.apply_method(roi,self.frame_id,store_values)
#            if area.id_type != 'gear':
#                area.apply_method(roi,self.frame_id,store_values)
            
        if self.ui.report_results.isChecked():
            self.frame_result['time_stamp'] = self.current_time.currentTime().toString('hh:mm:ss.zzz')
            self.frame_result['gear_speed'] = self.areas['gear'].gear_speed #pxl/frame
            self.frame_result['filament_speed'] = self.areas['filament_speed'].filament_speed #pxl/frame
            self.frame_result['filament_width'] = self.areas['width'].width_value #pxl
            self.frame_result['time_elasped'] = self.time_passed_ms #msec
            self.frame_result['cal_factor'] = self.convert_width #mm/pxl
            self.frame_result['frame_id'] = self.frame_id #
            self.pipe_out.send(self.frame_result)
            
 
#------------------------------------------------------------------------------
    def show_setup_roi(self):
        '''
        draw ROI rectangles and some text on frame
        '''
        #----------------------------------------------------------
        #draw ROI extruder
#        colour = (0,0,255)
        if not(self.measure_active):
            for area in self.areas.values():
                cv2.rectangle(self.frame_extruder, area.roi_pnt1, area.roi_pnt2,
                              area.roi_colour, 1)
                         
            #gear OD
            cv2.circle(self.frame_extruder, (self.ui.x_gear_c.value(), self.ui.y_gear_c.value()), 
                       self.gear_cr_od, cv_colors['cyan'])
            
            #filament width manual lines
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
                     
            
            
    
    #        fps_str = 'FPS: {:.1f}'.format(1./self.time_step)
    ##        fps_str = 'FPS: {:.3f}'.format(1./np.average(self.time_steps[-15:]))
    #        draw_str(self.frame_extruder,(50,40),fps_str)
            
            #cx and cx        
            cv2.line(self.frame_extruder,
                     (self.ui.x_gear_c.value(),0),
                     (self.ui.x_gear_c.value(),self.frame_extruder.shape[0]),
                     cv_colors['cyan'])
            
            cv2.line(self.frame_extruder,
                     (0,self.ui.y_gear_c.value()),
                     (self.frame_extruder.shape[1],self.ui.y_gear_c.value()),
                     cv_colors['cyan'])
            
            #gear mask
            cv2.circle(self.frame_extruder, (self.ui.x_gear_c.value(), self.ui.y_gear_c.value()), 
                       self.areas['gear'].cr_max, cv_colors['red'])
            cv2.circle(self.frame_extruder, (self.ui.x_gear_c.value(), self.ui.y_gear_c.value()), 
                       self.areas['gear'].cr_min, cv_colors['red'])
        #-    
        self.ui.gear_speed.setValue(self.areas['gear'].gear_speed*self.convert_speed)
        self.ui.filament_speed.setValue(self.areas['filament_speed'].filament_speed*self.convert_speed)
        self.ui.measured_width.setValue(self.areas['width'].width_value*self.convert_width)
        self.ui.missed_frames.setValue(self.missed_frames)
        
        if self.time_passed_ms != 0:
            self.ui.fps.setValue(1/self.time_passed_ms*1000.)
            
        
        
        #SHOW
        self.microscope_display.pixmap = converter(self.frame_extruder)
        self.update()
#------------------------------------------------------------------------------
    def load_video_from_file(self,video_fullname = None):
        '''
        Called by pb clicked:
            disconnect current camera if connected
            enalbe controls
            setup frame numbers and slider.
            
        Once video is running, camera view cannot be restored. 
        Only through restarting app.
        '''
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

#------------------------------------------------------------------------------
    def video_new_frame(self):
        '''
        get new frame
        '''
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
#------------------------------------------------------------------------------
    def new_frame_trackbar(self,new_frame_id):
        '''
        '''
        self.frame_id = new_frame_id
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)        
#------------------------------------------------------------------------------
    def new_frame_next(self):
        '''
        '''
        if self.frame_id + 1 < self.num_frames:
            new_frame = self.frame_id + 1
        else:
            new_frame = self.frame_id
            self.ui.play_video.setChecked(False) #end of file
#        self.frame_id = self.frame_id + 1 if self.frame_id + 1 < self.num_frames else self.frame_id
#        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
        
        self.ui.frame_number.setValue(new_frame)
        
                                  
#------------------------------------------------------------------------------
    def new_frame_prev(self):
        '''
        '''
        new_frame= self.frame_id -1 if self.frame_id - 1 >= 0 else 0
        self.ui.frame_number.setValue(new_frame)
#        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
#------------------------------------------------------------------------------    
    def record_video_start_stop(self,new_state = False):
        '''
        Video recording, called from checkbox click
        
        Start, continue or pause video recording.
        '''   
        
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
            self.record_filename_prev = self.record_filename 
            self.recording = True
            

        elif not(self.video_mode): #pause
            self.recording = False
            self.ui.record_state.setText('Paused')
        else:
            self.recording = False
#------------------------------------------------------------------------------    
    def record_video_save(self):
        '''
        stop and save
        '''
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
            
            
#------------------------------------------------------------------------------    
    def record_start_new(self):
        '''
        '''
        self.record_video_save()
        self.ui.record_video.setChecked(True)
        self.record_video_start_stop(True)
#------------------------------------------------------------------------------
#==============================================================================
#==============================================================================
#==============================================================================
class VisionProcess(Process):
    '''
    '''
    def __init__(self,pipe_2_vision_out,pipe_from_vision_in,**kwargs):
        '''
        '''
        super().__init__()
        self.pipe_2_vision_out,self.pipe_from_vision_in = pipe_2_vision_out,pipe_from_vision_in
        self.app_settings = kwargs
#-------------------------------------------------------------------------------
    def run(self):
        '''
        start application
        '''
        self.app = QtWidgets.QApplication([])
        self.vision = ExtruderCam(self.pipe_2_vision_out,self.pipe_from_vision_in,**self.app_settings)    
        self.vision.show()
        self.app.exec_()
#==============================================================================
class EmptyPipe():
    def send(self,text):
        print(text)
    def poll(self):
        return False
    def recv(self):
        return ''
      
#==============================================================================    

    
def main():
    import sys#,traceback
#    print('main')
#    print(QtCore.QT_VERSION)
#    if QtCore.QT_VERSION >= 0x50501:
#        def excepthook(type_, value, traceback_):
#            traceback.print_exception(type_, value, traceback_)
#            QtCore.qFatal('')
#        sys.excepthook = excepthook
   
    pipe_2_vision_out = EmptyPipe()
    pipe_from_vision_in = EmptyPipe()
    
    app = QtWidgets.QApplication(sys.argv)

    mc = ExtruderCam(pipe_2_vision_out,
                     pipe_from_vision_in,
                     do_profiling = 0,
                     load_video_def = False,
                     show_gear = False)
    mc.show()
    sys.exit(app.exec_())    
    

    
         
    
        
#===============================================================================    
if __name__ == '__main__':
    main()


    
    