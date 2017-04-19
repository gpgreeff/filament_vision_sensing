# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:30:20 2016

"""
__version__ = "1.0"
__license__ = "MIT"
__author__  ="GP Greeff"

import cv2
from numpy import ndarray as np_ndarray
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore



from PyQt5.QtCore import pyqtSignal #video stream worker
import time
import subprocess #dino interface control

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

#def draw_str(dst, xxx_todo_changeme, s):
#    (x, y) = xxx_todo_changeme
#    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
#    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

                
def rotate_image(image, angle,other_centre = None, return_param = False):
    """
    http://stackoverflow.com/questions/9041681/
    opencv-python-rotate-image-by-x-degrees-around-specific-point
    """
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


def converter(frame,pixmap = True,colour_frame = True):
    """
    Convert opencv BGR numpy  array to q-img or to pixmap (if true)
    
    flopalm.com/post/.../convert-python-opencv-image-numpy-array-pyqt
    """
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

class CameraLabel(QtWidgets.QLabel):
    """
    Note:
        centre is relative to label size, and not pixmap. 
        The pixmap is scaled to fit label, then the lines are drawn ontop.
    
    """
#    new_y_pos = QtCore.pyqtSignal(object) #new y pos available
    def __init__(self,parent = None,**kwargs):

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

    def set_draw_line(self,new_bool_val):
        self.draw_line = new_bool_val
    
    def resizeEvent(self,event):
        super(CameraLabel, self).resizeEvent(event)
        self.alignment_y = self.height()/2
        self.alignment_x = self.width()/2
        self.mouse_press_x = self.alignment_x 
        self.half_y = self.height()/2 - 10
        self.line_width_start_y = self.half_y - self.line_width_h
        self.line_width_end_y = self.half_y + self.line_width_h


    def paintEvent(self, event):
        rect = self.rect()
        painter = QtGui.QPainter(self)
        
        try:
            painter.drawPixmap(0,0,self.width(),self.height(), self.pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            
            painter.drawRect(rect)
        except TypeError: #arguments did not match any overloaded call - pixmap incorrect
            pass
        

    def minimumSizeHint(self):
        return QtCore.QSize(self.size_hint_width, self.size_hint_height)

    def sizeHint(self):
        return QtCore.QSize(self.size_hint_width, self.size_hint_height)


class VideoStreamWorker(QtCore.QObject):
    """
    Using threading to reduce video capture latency.
    
    Setup to handle more than one thread worker, but each camera can only be accessed by one worker.
    
    Camera is selected with cam_settings['cam_number']
    
    Main source:
        http://stackoverflow.com/questions/41526832/pyqt5-qthread-signal-not-working-gui-freeze   
        
    Additional info:
        http://www.pyimagesearch.com
        https://gist.github.com/TimSC/5670099
        https://wiki.python.org/moin/PyQt/Threading,_Signals_and_Slots
        https://nikolak.com/pyqt-threading-tutorial/
    
    
        
    dino_cam:
        use dino_cam initialsation routine

    Main thread interface:
        
        #control worker: (to worker)
        sig_abort_workers = pyqtSignal()
        self.sig_abort_workers.connect(worker_thread.abort)
        
        #data from worker
        @pyqtSlot(int, int, float, np_ndarray)
        def on_new_frame(self, worker_id:int,frame_id: int, time_step: float,frame:np_ndarray)
        
        etc.
        
        
    """
    sig_new_frame  = pyqtSignal(int, int,float, np_ndarray) # worker id, frame id, time_step, frame
    sig_done  = pyqtSignal(int)      # worker id: emitted at end of work()
    sig_msg   = pyqtSignal(int,str)  # worker id, message to be shown to user
    sig_error = pyqtSignal(int,str)  # worker id, error type and add info

    def __init__(self, id: int,main_app,**cam_settings):
        super().__init__()
        self.__id = id
        self.__abort = False
        self.app = main_app
        self.capture = None
        self.frame = None
        self.frame_id = 0
        
        self.fps = cam_settings.get('cam_fps',30)
        self.cam_settings = cam_settings
        
        if cam_settings.get('cam_dino',False):
            self.capture= init_dino_cam(**cam_settings)
        else:
            self.capture = cv2.VideoCapture(cam_settings.get('cam_number',0))
            if self.capture is not None:
                self.capture.set(5,self.fps)  #set fps
        
        if self.capture is not None:
            self.fps =  int(round(self.capture.get(5)))
            self.cam_cv_dict = cam_get_dict(self.capture,cam_settings.get('cam_get_vals','4,5,6,11,12,13,14,16'))
            self.grabbed, self.frame = self.capture.read()

            self.fps_time = QtCore.QTime() #timer
            self.fps_time.start()
            
            self.capturing_frames = True
            self.sig_msg.emit(self.__id,'Capture ready')
        else:
            self.capturing_frames = False
            self.sig_error.emit(self.__id,'Capture init fail')
        

    @QtCore.pyqtSlot()
    def work(self):
        """
        wait for new frames until __abort is called
        """
        thread_name = QtCore.QThread.currentThread().objectName()
        thread_id = int(QtCore.QThread.currentThreadId())  # cast to int() is necessary
        self.sig_msg.emit(self.__id,'Running worker #{} from thread "{}" (#{})'.format(self.__id, thread_name, thread_id))

        while self.capturing_frames:
            self.grabbed, self.frame = self.capture.read()
            if self.grabbed:
                time_step = self.fps_time.elapsed()
                self.sig_new_frame.emit(self.__id,self.frame_id,time_step,self.frame)
                self.fps_time.start()
                self.frame_id = self.frame_id + 1 if self.frame_id + 1 < 999 else 0
            else:
               self.sig_error.emit(self.__id,'Capture read fail, after frame: {}'.format(self.frame_id))
            
            # check if we need to abort the loop; need to process events to receive signals;
            self.app.processEvents()  # this could cause change to self.__abort
            
            if self.__abort:
                # note that "step" value will not necessarily be same for every thread
                self.sig_msg.emit(self.__id,'Stopping work at frame {}'.format(self.frame_id))
                self.capturing_frames = False
                break

        self.sig_done.emit(self.__id)
        if self.capture is not None:
            self.capture.release()

    def abort(self):
        """
        called to stop thread, by main thread with for example:
            self.sig_abort_workers.emit()
        """
        self.sig_msg.emit(self.__id,'Worker #{} notified to abort'.format(self.__id))
        self.__abort = True
        


def init_dino_cam(**cam_settings):
    """
    cam model: Dino-Lite Pro - AM413T
    
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #capture.set(cv2.CAP_PROP_FPS, fps)
    
    """
    #exeternal commands - cam specific
    folder = r'C:\Program Files (x86)\Dino-Lite DOS Control\Samples\DN_DS_Ctrl.exe '
    flicker_reduction_on =  folder + 'AntiFlicker On 50'
    cam_led_on =  folder + 'LED on'
    cam_ae_on  =  folder + 'AE on -CAM0'
    cam_ae_off =  folder + 'AE off -CAM0'
    
    capture = None
    capture = cv2.VideoCapture(cam_settings.get('cam_number',0))
    read_success, frame = capture.read()
    tries_left = 10
    
    while not(read_success) and tries_left > 0:
        read_success, frame = capture.read()
        tries_left -= 1

    if read_success:
        capture.set(3,cam_settings.get('cam_width',640)) #set width
        capture.set(4,cam_settings.get('cam_height',480))#set height
        capture.set(5,cam_settings.get('cam_fps',30))    #set fps

        
        subprocess.call(flicker_reduction_on)
        subprocess.call(cam_led_on)
        subprocess.call(cam_ae_on)
        time.sleep(cam_settings.get('cam_ae_delay',1)) #delay seconds
        subprocess.call(cam_ae_off)
        
    else:
        capture = None
    
    return capture

def CV_cam_setttings_names():
    """
    """
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

def cam_get_dict(capture,get_indices = 'all'):
    """
    4,5,6,11,12,13,14,16
    """
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
