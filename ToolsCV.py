# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:30:20 2016

@author: Greeff
"""

import cv2
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import pyqtSignal #video stream worker
import time
import subprocess #dino interface control
'''
===============================================================================
CV2 and DRAWING METHODS
===============================================================================
'''
#BGR = [0,0,0] 
red = (0,0,255)#
green = (0,255,0)
blue = (255,0,0)
white = (255,255,255)
black = (0,0,0)
orange1= (0,170,255)
yellow = (0,255,255)
cyan = (255,255,0)
purple = (255,0,255)
gray = (125,125,125)
cv_colors = {'red':red,'green':green,'blue':blue,'white':white,
             'black':black,'orange':orange1,'yellow':yellow,'cyan':cyan,
             'purple':purple,'gray':gray}
#------------------------------------------------------------------------------             
def draw_str(dst, xxx_todo_changeme, s):
    (x, y) = xxx_todo_changeme
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

                
#------------------------------------------------------------------------------               
def rotate_image(image, angle,other_centre = None, return_param = False):
    '''
    http://stackoverflow.com/questions/9041681/
    opencv-python-rotate-image-by-x-degrees-around-specific-point
    '''
    if image.ndim ==3:
        h,w,l = image.shape
    else:
        h,w = image.shape
        
    if other_centre is None:
        image_center = (w/2,h/2)
    else:
        image_center = other_centre
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, (w,h),flags=cv2.INTER_LINEAR)
    
    if return_param:
        w_h = (w,h)
        return result,rot_mat,w_h,cv2.INTER_LINEAR
        
    return result
#===============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
def converter(frame,pixmap = True,colour_frame = True):
    '''
    converty opencv BGR numpy  array to q-img or to pixmap (if true)
    #flopalm.com/post/.../convert-python-opencv-image-numpy-array-pyqt
    '''
    if colour_frame:
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)#if len(frame.shape) == 2:

    qimg = QtGui.QImage(rgb.data,rgb.shape[1],rgb.shape[0],QtGui.QImage.Format_RGB888)
    #QtGui.QImage(frame.tostring(), frame.width, frame.height, QtGui.QImage.Format_RGB888).rgbSwapped()    
    if not pixmap:
        return qimg  
    else:
        return QtGui.QPixmap.fromImage(qimg)
#==============================================================================
#==============================================================================
#==============================================================================
class CameraLabel(QtWidgets.QLabel):
#    new_y_pos = QtCore.pyqtSignal(object) #new y pos available
    def __init__(self,parent = None,**kwargs):
        '''
        Note, centre is now relative to label size, and not pixmap. The pixmap 
        is scaled to fit label, then the lines are drawn ontop.
        '''
        QtWidgets.QLabel.__init__(self)
        self.display_active = True
        self.alignment_y = self.height()/2
        self.alignment_x = self.width()/2
        self.half_y = self.alignment_y
#        self.draw_line = True
#        self.pixmap = QtGui.QPixmap()
        
        #line width
        self.line_width = 0.10 #%pxls, half
        self.line_width_h = 10
        self.line_width_start_y = 10
        self.line_width_end_y = 10
        
        self.size_hint_width = kwargs.get('size_hint_width',320)
        self.size_hint_height = kwargs.get('size_hint_height',240)
        
        
        #pen setup
#        self.align_h_pen = QtGui.QPen(QtCore.Qt.green, 2, QtCore.Qt.DashLine)#, cap, join
#        self.align_v_pen = QtGui.QPen(QtCore.Qt.green, 2, QtCore.Qt.SolidLine)#, cap, join
#        self.line_width_pen  = QtGui.QPen(QtCore.Qt.green, 2, QtCore.Qt.SolidLine)
        
#        self.align_brush = 1
#-------------------------------------------------------------------------------
    def set_draw_line(self,new_bool_val):
        self.draw_line = new_bool_val
#-------------------------------------------------------------------------------    
    def resizeEvent(self,event):
        super(CameraLabel, self).resizeEvent(event)
        self.alignment_y = self.height()/2
        self.alignment_x = self.width()/2
        self.mouse_press_x = self.alignment_x 
        self.half_y = self.height()/2 - 10
        self.line_width_start_y = self.half_y - self.line_width_h
        self.line_width_end_y = self.half_y + self.line_width_h

#-------------------------------------------------------------------------------
    def paintEvent(self, event):
        rect = self.rect()
        painter = QtGui.QPainter(self)
        
        try:
            painter.drawPixmap(0,0,self.width(),self.height(), self.pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            
            painter.drawRect(rect)
        except TypeError: #arguments did not match any overloaded call - pixmap incorrect
            pass
        
#-------------------------------------------------------------------------------
    def minimumSizeHint(self):
        return QtCore.QSize(self.size_hint_width, self.size_hint_height)

    def sizeHint(self):
        return QtCore.QSize(self.size_hint_width, self.size_hint_height)
#-------------------------------------------------------------------------------
#==============================================================================
#==============================================================================
#==============================================================================
class VideoStreamWorker(QtCore.QThread):
    '''
    Using threading to reduce video capture latency
    
    Based on the following and update to Qt5:
        http://www.pyimagesearch.com
        https://gist.github.com/TimSC/5670099
        https://wiki.python.org/moin/PyQt/Threading,_Signals_and_Slots
        https://nikolak.com/pyqt-threading-tutorial/
        
    '''
    new_frame_rdy = pyqtSignal()
    new_frame = pyqtSignal([list])
    
    def __init__(self,**cam_settings):
        '''
        cam_number:
            camera number
        
        dino_cam:
            use dino_cam initialsation routine
            
        
        '''
        super(VideoStreamWorker, self).__init__() 
        
        self.capture = None
        self.frame = None
        self.fps = 30
        self.frame_id = 0
        self.cam_settings = cam_settings
        
        #pipe method
#        self.pipe_in,self.pipe_out = pipe_in,pipe_out
        self.frame_processed = True
        self.frames_missed = 0
        
        
        if cam_settings.get('cam_dino',False):
            self.capture= init_dino_cam(**cam_settings)
        else:
            self.capture = cv2.VideoCapture(cam_settings.get('cam_number',0))
        
        
        
        
        if self.capture is not None:
            self.fps =  int(round(self.capture.get(5)))
            self.cam_cv_dict = cam_get_dict(self.capture,cam_settings.get('cam_get_vals',''))
            (self.grabbed, self.frame) = self.capture.read()
            
            self.exiting = False
            
            self.fps_time = QtCore.QTime() #timer
            self.fps_time.start()
        else:
            self.exiting = True
#        t.daemon = True
        
#-------------------------------------------------------------------------------    
    def run(self):
        '''
        keep looping infinitely until the thread is stopped
        '''
        while not self.exiting:
            (self.grabbed, self.frame) = self.capture.read()
            if self.grabbed:

                #emit frame method
                #self.frame_id = self.frame_id + 1 if self.frame_id <= 10000 else 0
                #self.new_frame.emit([self.frame,self.fps_time.elapsed(),self.frame_id])
                #self.fps_time.start()
                
                #pipe method
#                if self.pipe_in.poll():
#                    new_task = self.pipe_in.recv()
#                    print('new_task',new_task)
#                    if new_task == 1:
                if self.frame_processed:
                    self.new_frame_rdy.emit()
                    self.frame_processed = False
                    
                    
                elif not(self.exiting):
                    self.frames_missed += 1
#                self.pipe_out.send([self.frame,self.fps_time.elapsed(),self.frame_id,self.frames_missed])
                
                
#                print('new_frame_rdy emit')
#                else:
#                    self.frames_missed += 1
                    
                
                
            else:
                print('no frame!')
        
#        self.quit()
#        print('quiting!')
#        self.setTerminationEnabled(True)
#        self.terminate()
        
                
#-------------------------------------------------------------------------------    
    def get_frame_with_rate(self):
        '''
        return the frame most recently read
        '''
        time_step = self.fps_time.elapsed()
        self.frame_id = self.frame_id + 1 if self.frame_id <= 999 else 0
        self.fps_time.start()
        return [self.frame,time_step,self.frame_id,self.frames_missed]

#-------------------------------------------------------------------------------    
    def read_frame(self):
        '''
        return the frame most recently read
        '''
        return self.frame
#-------------------------------------------------------------------------------    
    def stop(self):
        '''
        indicate that the thread should be stopped
        '''
        self.exiting = True
        if self.capture is not None:
            self.capture.release()
        
        
#-------------------------------------------------------------------------------
    def __del__(self):
    
        self.exiting = True
        if self.capture is not None:
            self.capture.release()
        self.quit()
        self.wait()
#==============================================================================
def init_dino_cam(**cam_settings):
    '''
    cam model: Dino-Lite Pro - AM413T
    
    '''
    #exeternal commands - cam specific
    folder = r'C:\Program Files (x86)\Dino-Lite DOS Control\Samples\DN_DS_Ctrl.exe '
    flicker_reduction_on =  folder + 'AntiFlicker On 50'
    cam_led_on =  folder + 'LED on'
    cam_ae_on  =  folder + 'AE on -CAM0'
    cam_ae_off =  folder + 'AE off -CAM0'
    
#    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#    capture.set(cv2.CAP_PROP_FPS, fps)
    
    capture = None
    capture = cv2.VideoCapture(cam_settings.get('cam_number',0))
    ret, frame = capture.read()
    tries_left = 10
    
    while frame is None and tries_left > 0:
        ret, frame = capture.read()
        tries_left -= 1

    if frame is not None:
        capture.set(3,cam_settings.get('cam_width',640)) #set width
        capture.set(4,cam_settings.get('cam_height',480)) #set height
        capture.set(5,cam_settings.get('cam_fps',30))  #set fps
#        capture.set(15, 0.1)
        
        subprocess.call(flicker_reduction_on)
        subprocess.call(cam_led_on)
        subprocess.call(cam_ae_on)
        time.sleep(cam_settings.get('cam_ae_delay',1)) #delay seconds
        subprocess.call(cam_ae_off)
        
    else:
        capture = None
    
    return capture
#==============================================================================
def CV_cam_setttings_names():
    '''
    '''
    settings = []
    settings.append('CV_CAP_PROP_POS_MSEC, Current position of the video file in milliseconds.')
    settings.append('CV_CAP_PROP_POS_FRAMES, 0-based index of the frame to be decoded/captured next.')
    settings.append('CV_CAP_PROP_POS_AVI_RATIO, Relative position of the video file')
    settings.append('CV_CAP_PROP_FRAME_WIDTH, Width of the frames in the video stream.')
    settings.append('CV_CAP_PROP_FRAME_HEIGHT, Height of the frames in the video stream.')
    settings.append('CV_CAP_PROP_FPS, Frame rate.')
    settings.append('CV_CAP_PROP_FOURCC, 4-character code of codec.')
    settings.append('CV_CAP_PROP_FRAME_COUNT, Number of frames in the video file.')
    settings.append('CV_CAP_PROP_FORMAT, Format of the Mat objects returned by retrieve() .')
    settings.append('CV_CAP_PROP_MODE, Backend-specific value indicating the current capture mode.')
    settings.append('CV_CAP_PROP_BRIGHTNESS, Brightness of the image (only for cameras).')
    settings.append('CV_CAP_PROP_CONTRAST, Contrast of the image (only for cameras).')
    settings.append('CV_CAP_PROP_SATURATION, Saturation of the image (only for cameras).')
    settings.append('CV_CAP_PROP_HUE, Hue of the image (only for cameras).')
    settings.append('CV_CAP_PROP_GAIN, Gain of the image (only for cameras).')
    settings.append('CV_CAP_PROP_EXPOSURE, Exposure (only for cameras).')
    settings.append('CV_CAP_PROP_CONVERT_RGB, Boolean flags indicating whether images should be converted to RGB.')
    settings.append('CV_CAP_PROP_WHITE_BALANCE, Currently unsupported')
    settings.append('CV_CAP_PROP_RECTIFICATION, Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)')
    
    return settings
#==============================================================================
def cam_get_dict(capture,get_indices = 'all'):
    '''
    '''
    settings = CV_cam_setttings_names()
    cam_setup = []
    if get_indices == 'all':
        get_indices = range(19)
    else:
        get_indices = [int(get_index) for get_index in get_indices.split(',')]

    for index in get_indices:
        cam_setup_row = {}
        cam_setup_row['value'] = capture.get(index)
        cam_setup_row['CV'] = settings[index].split(',')[0]
        cam_setup_row['description'] = settings[index].split(',')[1]
        cam_setup_row['id'] = index
        cam_setup.append(cam_setup_row)
    return cam_setup
#==============================================================================