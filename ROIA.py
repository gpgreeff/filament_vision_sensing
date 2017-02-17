# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 13:37:47 2016

@author: Greeff



"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt as math_sqrt

np.seterr(invalid='raise')


#==============================================================================
def reject_outliers(data, m = 2.,return_option = 0,default_value = 0):
    '''
    http://stackoverflow.com/questions/11686720/
    is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    
    data:   np.array
    
    m:      threshold
    
    return_option:
    
        0 - values w/o outliers
        
        1 - average of values w/o outliers
        
        2 - mean of values w/o outliers
        
        3 - non-outlier indices
        
        4 - [option 0, option 2, option 3]
    
    '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    indices = np.array(s < m,dtype = int)
#    indices = indices.astype(int)
#    
    try:
        new_set = data[indices]

    except IndexError:
        if default_value is not None:
            return default_value
        new_set = data[-1]
#        print('Warning IndexError - len(data) == 1')

    if return_option == 0:
        return new_set
        
    if return_option == 1:
        return np.average(new_set)
        
    elif return_option == 2:
       return np.mean(new_set)
      
    elif return_option == 3:
       return indices
      
    elif return_option == 4:
       return [np.std(new_set),np.mean(new_set)] 
#==============================================================================
class ROIA(object):
    '''
    ROIA: Region of Interest Analyse.

    Analyse differnt ROI with different methods:
    
        -width detection
            -delta colour method
            -colour mask-contour fill method
            -canny method
            -roi threshold method
        
        -speed
            -gear method - tracking
            -filament method - tracking
            -filament mehtod - phase shift
        
        -colour mask
            -hsv roi setup
            -apply mask
    '''
    def __init__(self,id_type,roi, **kwargs):
        '''
        id_type - type of detection:
            -'width'
            -'gear'
            -'filament_speed'
        
        roi: np array 
        
        plot_on = False - debug control, investigate specific frame
            
        roi_pnts = (x1,y1,x2,y2)
        
        
        w_ave_range = 15
        w_detection_half_width
        w_vline_left_border
        w_vline_right_border
        w_edge_offset
        
        
        width_default = 150
            
        thresh_inv = True
        material = 'white'
        '''
        self.id_type = id_type 
        self.roi = roi
        self.configs = kwargs
        
        #view/debug control
        self.update_plot = False
        self.plot_on = kwargs.get('plot_on',False)
        self.first_analysis = True
        
        if self.plot_on:
             plt.ion()
             plt.show()

        #general processing

#        self.blur_method = 1
        
        material = self.configs.get('material','white')
        
        self.thresh_inv = self.configs.get('thresh_inv',False) #delta ave method - invert min/max
        #----------------------------------------------------------------------
        if self.id_type == 'width': 
            #width calculation
 
            #sub_pixel method
            self.w_zf = 4 #zoomfactor
            self.w_detection_half_width = self.configs.get('w_detection_half_width',10)
            self.w_vline_left_border = self.configs.get('w_vline_left_border',20)
            self.w_vline_right_border = self.configs.get('w_vline_right_border',15)
            
            #delta colour method
            self.w_edge_offset = self.configs.get('w_edge_offset',3) #edge effect clearance
            self.w_ave_range = self.configs.get('w_ave_range',15) #+-range from ave center line where profile must be
            
            self.apply_method = self.apply_method_width
            
            #material specific
            if material == 'model':
                self.w_ave_method_use_green = False 
                self.thresh_inv  = True
            else:
                self.w_ave_method_use_green = True #delta ave method - use blue-greeen instead of gray
            
            if self.plot_on:
                fig, axes = plt.subplots(2,2,num = 'filament width')
                self.axes_ave_cols_left = axes[0,0]


        #----------------------------------------------------------------------
        elif self.id_type == 'filament_speed':
            self.speed_compare = []
            self.tracks = []
            self.filament_speed = 0
            
            #TRACKING
            self.feature_params = dict(maxCorners = 50,
                                       qualityLevel = 0.3,
                                       minDistance = 2,
                                       blockSize = 3 )
            
            self.track_pnt_mask_radius = 2
            
            lk_params_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            
    #        self.lk_params = dict( winSize  = (30, 30),
    #                              maxLevel = 2,
    #                              criteria = lk_params_criteria)
                                  
            self.lk_params = dict( winSize  = (15, 15),
                                  maxLevel = 2,
                                  criteria = lk_params_criteria)
            self.track_len = 2
            self.detect_interval = 2
            
            self.filament_sp_min_step = 0.75
            self.ave_method_use_red = False
            
            
            if material == 'green':
                self.filament_sp_filter_blue = True
                self.filament_sp_use_right = False
                
            elif material == 'white':
                self.filament_sp_filter_blue = False
                self.filament_sp_use_right = False
    
            elif material == 'clear':
                self.filament_sp_use_right = True
                self.filament_sp_filter_blue = False
            
            elif material == 'red':
                self.filament_sp_use_right = True
                self.filament_sp_filter_blue = True
                
            elif material == 'model':
                self.filament_sp_use_right = True
                self.filament_sp_filter_blue = True
                self.ave_method_use_red = True
                
                
            #correlation speed
            self.speed_max_corr = 15.
            
            if self.plot_on:
                self.apply_method = self.flow_track_filament_with_plot
                self.num_used_tracks = []
                self.num_not_vertical_x_vs_y_frame = []
                self.num_y_neg_frame = []
            else:
                self.apply_method = self.apply_method_filament_speed
        #----------------------------------------------------------------------
        elif self.id_type == 'gear':
            #gear
            
            self.diameter =  self.configs.get('diameter',8.)    #mm
            self.radius_mm = self.diameter/2   #mm
            
            self.cr_od = self.configs.get('cr_od',227)  #pxl
            self.cx =  self.configs.get('cx',21)        #pxl
            self.cy = self.configs.get('cy',240)        #pxl
            
            self.gear_use_mask =  self.configs.get('gear_use_mask',True) 
          
            self.cr_max = self.cr_od + 10
            self.cr_min = self.cr_od - 20
            

            self.gear_sp_min_R = self.configs.get('gear_sp_min_R',2.) 
            self.gear_sp_min_h_sqrd = self.configs.get('gear_sp_min_h_sqrd',2)#self.gear_sp_min_R**2
            self.gear_sp_min_angle = self.configs.get('gear_sp_min_angle',0.005) #0.015
            self.gear_sp_max_angle = self.configs.get('gear_sp_max_angle',0.088*10) #0.088
            
            #vals
            self.gear_speed = 0
            self.speed_compare = []
            self.tracks = []

            #TRACKING
            self.feature_params = dict(maxCorners = 50,
                                       qualityLevel = 0.3,
                                       minDistance = 2,
                                       blockSize = 3 )
            
            self.track_pnt_mask_radius = 2
            
            lk_params_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            
    #        self.lk_params = dict( winSize  = (30, 30),
    #                              maxLevel = 2,
    #                              criteria = lk_params_criteria)
                                  
            self.lk_params = dict( winSize  = (15, 15),
                                  maxLevel = 2,
                                  criteria = lk_params_criteria)
            
            self.track_len = self.configs.get('gear_track_len',2)
            self.detect_interval = self.configs.get('gear_detect_interval',2)
            
            if self.plot_on:
                self.apply_method = self.apply_method_gear_with_plot
                
                x1,y1,x2,y2 = self.configs.get('roi_pnts',(437, 320, 510,400))
                roi_height = y2 - y1
                roi_width = x2 - x1
                #cv windows
                start_wx = 800
                cv2.namedWindow('gear_roi_in')
                cv2.moveWindow('gear_roi_in',start_wx,0)
                
                cv2.namedWindow('gear_gray_pre')
                cv2.moveWindow('gear_gray_pre',start_wx,roi_height +40)
                
                if self.gear_use_mask:
                    cv2.namedWindow('gear_mask')
                    cv2.moveWindow('gear_mask',start_wx+roi_width + 40,0)
                
                cv2.namedWindow('gear_thresh')
                cv2.moveWindow('gear_thresh',start_wx+roi_width + 40,roi_height +40)

               
                
            else:
                self.apply_method = self.apply_method_gear

        #----------------------------------------------------------------------
        #general inits
        if self.thresh_inv:
            self.thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        else:
            self.thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
            
        #roi setup
        self.roi_init(self.configs.get('roi_pnts',(437, 320, 510,400)))
        
        #zero arrays
        self.meas_vals_init()   
        
        
            

#------------------------------------------------------------------------------
    def meas_vals_init(self):
        '''
        '''
        #profile
#        self.profile_vals = np.array([])
        
        #width procesing
        if self.id_type == 'width':
            self.width_average = [] #average of w_vals_f per frame - track delta width
            self.width_value = self.configs.get('width_default',150)  #current value for filament width
            self.width_left_pnt = 0
            self.width_right_pnt = self.width_value
            
        else:
            #speed
            self.speed = [] #gear track speed or filament speed, in pxl per frame
            
#        self.speed_r = []
#        self.speed_f = np.array([0]) #filter speed
#        self.speed_beta = [] #correlation method
#        self.speed_w = []
#        self.speed_column = [] #column method

#------------------------------------------------------------------------------
    def roi_init(self,roi_pnts):
        '''
        '''
        self.roi_pnts = roi_pnts
        x1,y1,x2,y2 = self.roi_pnts
        self.roi_pnt1 = (x1,y1)
        self.roi_pnt2 = (x2,y2)
        self.roi_height = y2 - y1
        self.roi_width= x2 - x1
#        self.x_points = np.arange(self.roi_width)
#        self.roi_y_start = self.roi_pnt1[1]
#        self.roi_y_end = self.roi_pnt2[1]

        #gear
        if self.id_type == 'gear':
            self.cy_flipped = (self.roi_pnts[3] - self.roi_pnts[1]) - self.cy
            self.gear_mask = None
            self.new_pnts_mask  = None
            
        #filament speed    
        elif self.id_type == 'filament_speed':
            self.new_pnts_mask  = None
        
        #width
        elif self.id_type == 'width':
            self.w_roi_middle = int((self.roi_width)/2)
            self.w_roi_middle_left = int(self.w_roi_middle/2)
            self.w_roi_middle_right = int(self.w_roi_middle + self.w_roi_middle/2)
            
            
            
            self.w_left_pnt = self.w_roi_middle_left
            self.w_right_pnt = self.w_roi_middle_right
            
            self.w_vline_left_roi = self.w_vline_left_border
            self.w_vline_right_roi = x2- x1 - self.w_vline_right_border
#            print(self.w_vline_left_roi,self.w_detection_half_width)
            roi_left = self.roi[:,self.w_vline_left_roi - self.w_detection_half_width:self.w_vline_left_roi + self.w_detection_half_width]
            roi_right = self.roi[:,self.w_vline_right_roi - self.w_detection_half_width:self.w_vline_right_roi + self.w_detection_half_width]
        
            self.w_vline_left_roix = roi_left.shape[1] - self.w_detection_half_width
            self.w_vline_right_roix = self.w_detection_half_width
            
            self.w_zoom_left = (roi_left.shape[1]*self.w_zf,  roi_left.shape[0]*self.w_zf)
            self.w_zoom_right= (roi_right.shape[1]*self.w_zf, roi_right.shape[0]*self.w_zf)
        
#------------------------------------------------------------------------------
    def apply_method_width(self,roi,frame_id,store_values):
        '''
        estimate the filament width, assume perfect roundness
        '''
        #pre
        self.roi = cv2.GaussianBlur(roi, (3, 3),0)
        self.gray = cv2.cvtColor(self.roi,cv2.COLOR_BGR2GRAY)

        #process
#        self.delta_colour_method_ave()
        self.width_subpixel_method()
        
        if store_values:
            self.width_average.append(self.width_value)
        

#------------------------------------------------------------------------------
    def apply_method_filament_speed(self,roi,frame_id,store_values):
        '''
        '''
        self.frame_id = frame_id
        self.roi = cv2.GaussianBlur(roi, (3, 3),0)
        
        roi_test = self.roi.copy()
        if self.filament_sp_filter_blue:
            roi_test[:,:,1:] = 0
            
        self.gray = cv2.cvtColor(roi_test,cv2.COLOR_BGR2GRAY)
        left_start = 0  #max(0,x_left_ave-self.w_ave_range )
        left_end = 30   #x_left_ave + self.w_ave_range
        
        ret_val,thresh_roi_1 = cv2.threshold(self.gray[:,left_start:left_end],0,255,self.thresh_type)

        self.gray[:,left_start:left_end] = thresh_roi_1
        
        if self.filament_sp_use_right:
            right_start = -10
            ret_val,thresh_roi_2 = cv2.threshold(self.gray[:,right_start:],0,255,self.thresh_type)
            self.gray[:,right_start:] = thresh_roi_2
            
            self.gray[:,left_end:right_start] = 0
                
        else:    
            self.gray[:,left_end:] = 0
        
        #process
#       speed_corr = self.speed_w_phase_corr(thresh_roi_1)
        self.filament_speed = self.flow_track_filament(self.gray)
#       self.column_speed(thresh_roi_1)            
        
        if store_values:
#           self.speed_beta.append(speed_corr)
            self.speed.append(self.filament_speed)
        

#------------------------------------------------------------------------------
    def apply_method_post(self,roi,frame_id,store_values):
        '''
        '''
        self.frame_id = frame_id
        self.roi = cv2.GaussianBlur(roi, (3, 3),0)
        self.gray = cv2.cvtColor(self.roi,cv2.COLOR_BGR2GRAY)

        #cvt/init
        self.result_img = np.zeros_like(self.gray)

        #process
        self.width_ave_range_method()
            
#       if store_values:
#------------------------------------------------------------------------------
    def apply_method_gear(self,roi,frame_id,store_values):
        '''
        '''
        self.frame_id = frame_id
        roi_test = cv2.GaussianBlur(roi, (3, 3),0)
        #cvt
#        roi_test = self.roi.copy()
        roi_test[:,:,1:] = 0

        self.gray = cv2.cvtColor(roi_test,cv2.COLOR_BGR2GRAY)
        
        if self.gear_use_mask:
            #mask gray
            try:
                self.gray[self.gear_mask_indices] = 0
            except AttributeError:
                self.gear_mask_generate(self.cx,self.cy,self.cr_max,self.cr_min)
                self.gray[self.gear_mask_indices] = 0
    
        thresh = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 1)
#        
#        ret_val,thresh = cv2.threshold(self.gray,0,255,self.thresh_type)
        
        if self.gear_use_mask:
            thresh[self.gear_mask_indices] = 0
        thresh = cv2.GaussianBlur(thresh, (3, 3),0)
        
        #track points
        self.gear_speed =  self.flow_track_gear(thresh)
        
        if store_values:
            self.speed.append(self.gear_speed)


#------------------------------------------------------------------------------
    def apply_method_gear_with_plot(self,roi,frame_id,store_values):
        '''
        '''
        cv2.imshow('gear_roi_in',roi)
        #pre-process
        self.frame_id = frame_id
        roi_test = cv2.GaussianBlur(roi, (3, 3),0)
        
        #cvt
#        roi_test = self.roi.copy()
        roi_test[:,:,1:] = 0

        self.gray = cv2.cvtColor(roi_test,cv2.COLOR_BGR2GRAY)
        
        #mask gray
        if self.gear_use_mask:
            try:
                self.gray[self.gear_mask_indices] = 0
            except AttributeError:
                self.gear_mask_generate(self.cx,self.cy,self.cr_max,self.cr_min)
                self.gray[self.gear_mask_indices] = 0
        
        cv2.imshow('gear_gray_pre',self.gray)
        
#        thresh = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                       cv2.THRESH_BINARY_INV, 11, 1)
        
        ret_val,thresh = cv2.threshold(self.gray,0,255,self.thresh_type)
        
        if self.gear_use_mask:
            thresh[self.gear_mask_indices] = 0
        thresh = cv2.GaussianBlur(thresh, (3, 3),0)
        
        cv2.imshow('gear_thresh',thresh)
        
        #track points
        self.gear_speed =  self.flow_track_gear(thresh)
#        self.gear_speed =  self.flow_track_gear_with_plot(thresh)
        
#        if store_values:
        self.speed.append(self.gear_speed)

        
        #----------------------------------------------------------------------
        #show
#        gear_vis = cv2.cvtColor(self.gray.copy(),cv2.COLOR_GRAY2BGR)
        cv2.imshow('gear_self.gray',roi_test)
#        cv2.imshow('gear_vis_centre',gear_vis)
        
        #plot
        if self.first_analysis:
            self.fig, self.axes = plt.subplots(2,2,num = 'gear speed')
            self.axes_gear_speed = self.axes[0,0]
            
            fig_manager = plt.get_current_fig_manager()
            window_geometry = fig_manager.window.geometry()
            x,y,dx,dy = window_geometry.getRect()
            fig_manager.window.setGeometry(640, 480, dx, dy)

               
        
        self.axes_gear_speed.cla()
        self.axes_gear_speed.plot(np.array(self.speed)*0.0212 * 30)
#        self.fig.clf()
#        self.axes_gear_speed.plot(self.speed)
#        self.axes_gear_speed_artist.set_xdata(range(len(self.speed)))
#        self.axes_gear_speed_artist.set_ydata(self.speed)
#        self.axes_gear_speed.axis(xmax = len(self.speed),ymax = max(self.speed))
        
        #----------------------------------------------------------------------
        #mask gray


#        thresh[self.gear_mask_indices] = 0
#        thresh = cv2.GaussianBlur(thresh, (3, 3),0)
#        
#        #track points
        self.first_analysis = False
        
#------------------------------------------------------------------------------
    def gear_mask_generate(self,cx,cy,cr_max,cr_min):
        '''
        '''
        self.gear_mask = np.zeros_like(self.gray,dtype = np.uint8)
        cv2.circle(self.gear_mask,(cx,cy),cr_max,(255),-1)
        cv2.circle(self.gear_mask,(cx,cy),cr_min,0,-1)
        
        self.gear_mask_indices = self.gear_mask == 0
        
        if self.plot_on:
            cv2.imshow('gear_mask',self.gear_mask)
#------------------------------------------------------------------------------
    def flow_track_gear(self,frame_gray):
        '''

        '''
        self.speed_xy_per_track = []
        #----------------------------------------------------------------------
        if self.tracks:
            self.track_points_update(frame_gray)
        #----------------------------------------------------------------------
        #update speed
        if len(self.speed_xy_per_track) > 1:
            gear_speed = reject_outliers(np.array(self.speed_xy_per_track),
                                         return_option = 1)
            gear_speed = math_sqrt(gear_speed)
            
        else:
            gear_speed = 0
        #----------------------------------------------------------------------
        #get new points
        self.get_new_track_points(frame_gray)

        return gear_speed
#------------------------------------------------------------------------------
    def flow_track_gear_with_plot(self,frame_gray):
        '''

        '''
#        print 'gear_flow ',self.frame_id
        #----------------------------------------------------------------------
        if self.plot_on:
#            plt.figure('xy')
#            plt.cla()
            plt.figure('Track View')
            for i in range(1,4,1):
                plt.subplot(3,1,i)
                plt.cla()
                plt.grid(True)
                plt.axhline(0,c='k')

            plt.subplot(3,1,1)
            plt.title('x and y steps')
            
            plt.subplot(3,1,2)
            plt.title('radius')
            plt.axhline(self.gear_sp_min_R,c='r')
            
            plt.subplot(3,1,3)
            plt.title('theta steps')
            plt.axhline(self.gear_sp_min_angle,c='r')
            plt.axhline(self.gear_sp_max_angle,c='r')

            num_reject_min_R = 0
            num_too_small_angle = 0
            if self.first_analysis:
                self.num_tracks_per_frame = []
                self.num_reject_min_R_frame = []
                self.num_too_small_angle_frame = []

        #----------------------------------------------------------------------
        speed_w_per_track = np.array([])
        speed_xy_per_track = np.array([])
        self.speed_xy_per_track = []
        #----------------------------------------------------------------------
        if self.tracks:
            self.track_points_update(frame_gray)
            
            
            array_track = np.asarray(self.tracks)
            if self.plot_on:
                print('num of tracks: ',np.shape(array_track),len(self.tracks))
                self.num_tracks_per_frame.append(array_track.shape[0])

            for counter,track in enumerate(array_track):
#                if counter >= 150:
#                    break
                b = np.asarray(track)

                #last 2 point only method
                last_points = b[-2:,:]

                #xy vector step method
                x_step = np.diff(last_points[:,0])
                y_step = np.diff(last_points[:,1])
                R = np.sqrt(x_step**2 + y_step**2)

                #--------------------------------------------------------------
                if self.plot_on:
#                    plt.figure(fig_name)
#                    plt.subplot(plots,1,1,aspect='equal')
#
#                    plt.scatter(last_points[0,0],last_points[0,1],c='b')
#                    plt.scatter(last_points[1,0],last_points[1,1],c='r')
#
#                    for point in last_points:
#                        plt.plot([self.cx,point[0]],[self.cy,point[1]],'g')
#
#                    plt.plot([last_points[0][0],last_points[1][0]],
#                             [last_points[0][1],last_points[1][1]],'c')

                    plt.figure('Track View')
                    plt.subplot(3,1,1)
                    plt.plot(counter,x_step,'bo',label = 'x_step')
                    plt.plot(counter,y_step,'ko',label = 'y_step')
                    plt.subplot(3,1,2)
                    plt.plot(counter,R,'ko',label = 'radius')


                #--------------------------------------------------------------
                #reject less than min
                R_indices = R > self.gear_sp_min_R
                R = R[R_indices]

                if not R:
                    if self.plot_on:
                        num_reject_min_R += 1
                    continue

                #translate gear centre (and points) to frame origin
                b_x = last_points[:,0] - self.cx
                b_y = last_points[:,1] - self.cy_flipped

                #radius for all points in track
#                r_theta = np.sqrt(b_x**2 + b_y**2)

                #calculate angles
                theta_all = np.arctan2(b_y,b_x)

                #step radians
                theta_steps = np.diff(theta_all)
                theta_steps = theta_steps[R_indices]
                #reject angles with too large/small radian step-change
                if self.plot_on:
                    plt.subplot(3,1,3)
                    plt.plot(counter,theta_steps,'bo',label = 'theta_step')


                t_indices = (np.abs(theta_steps) >= self.gear_sp_min_angle)*(np.abs(theta_steps) < self.gear_sp_max_angle)
                if not(np.any(t_indices)):
                    if self.plot_on:
                        num_too_small_angle += 1 #or too big
                    continue

                theta_steps = theta_steps[t_indices]

                R = R[t_indices]
                sign = 1- 2*int(theta_steps[-1] < 0) # if neg then -1 else +1
                if sign > 0:
                    sign_indices = theta_steps > 0
                else:
                    sign_indices = theta_steps < 0

                R = sign*R[sign_indices]
                speed_xy_per_track = np.append(speed_xy_per_track,R)
                theta_steps = theta_steps[sign_indices]

#                print 'theta_steps °:',np.rad2deg(theta_steps)
                if self.plot_on:
                    speed_w_per_track = np.append(speed_w_per_track,theta_steps*self.cr_od)



        #----------------------------------------------------------------------
        
        #update speed
        if self.speed_xy_per_track:
            gear_speed = reject_outliers(np.array(self.speed_xy_per_track),
                                         return_option = 1)
        else:
            gear_speed = 0
#            gear_speed = np.mean(self.speed_xy_per_track)
            
        if speed_xy_per_track:
            gear_speed_cmp = reject_outliers(speed_xy_per_track,return_option = 1)
        else:
            gear_speed_cmp = 0
        self.speed_compare.append(gear_speed_cmp)
        
        #get new points
        self.get_new_track_points(frame_gray)

        #plot
        if self.plot_on:
            self.num_reject_min_R_frame.append(num_reject_min_R)
            self.num_too_small_angle_frame.append(num_too_small_angle)

            plt.figure('all frames')
            plots = 3
            plt.subplot(plots,1,1)
            plt.title('mean final speed per frame')
            plt.cla()
            plt.plot(self.speed,'b')
            plt.plot(self.speed_compare,'r')
            plt.axhline(0,c='k')

            plt.subplot(plots,1,2)
            plt.cla()
            plt.title('individual track speeds - current frame')
            plt.plot(speed_xy_per_track,'r')
            plt.plot(speed_w_per_track,'gray')
            plt.plot(self.speed_xy_per_track,'b')
            plt.axhline(0,c='k')
            if speed_xy_per_track:
                plt.axhline(np.mean(speed_xy_per_track),c = 'r')
                print('mean r speed: ',np.mean(speed_xy_per_track))

            if speed_w_per_track:
                plt.axhline(np.mean(speed_w_per_track),c = 'gray')
                print('mean r speed: ',np.mean(speed_w_per_track))

#            print '-------------------------'
            plt.subplot(plots,1,3)
            plt.cla()
            plt.title('tracks per frame')
            plt.plot(self.num_tracks_per_frame, 'o-', label = 'Tracks')
            plt.plot(self.num_reject_min_R_frame,'o-', label = 'min R')
            plt.plot(self.num_too_small_angle_frame,'o-', label = 'Angle Outside')
            plt.legend()


            plt.tight_layout()
            #------------------------------------------------------------------
            cv2.namedWindow('gear_vis')
            cv2.moveWindow('gear_vis',650,0)
            vis = cv2.resize(frame_gray,(self.roi_width*1,self.roi_height*1))
            cv2.imshow('gear_vis',vis)

#        return vis
        return gear_speed
#------------------------------------------------------------------------------
    def gear_speed_append(self,x,y,prev_pnt):
        '''
        determine speed (step) for a new track pnt added to an existing track
        
        x, y new pnt coordinates
        
        prev_pnt = [x,y]
        
        reject pnts with too small h (h = dx**2 + dy**2), 
        ie pnts that did not move
        
        CW positive rotation
        
        '''
        dx = x - prev_pnt[0]
        dy = y - prev_pnt[1]
        h = dx*dx + dy*dy
        
        if h > self.gear_sp_min_h_sqrd:
#            angle_new = np.arctan2(y - self.cy_flipped,x - self.cx)
#            angle_prev = np.arctan2(prev_pnt[1] - self.cy_flipped,prev_pnt[0]- self.cx)
            x0 = prev_pnt[0]- self.cx
            y0 = prev_pnt[1] - self.cy_flipped
#            det = dx*y0 - x0*dy
            if dx*y0 < x0*dy:
#                self.speed_xy_per_track.append(np.sqrt(h))
#                self.speed_xy_per_track.append(math_sqrt(h))
                self.speed_xy_per_track.append(h) #sqrt rt later
                
                
#------------------------------------------------------------------------------
    def filament_speed_append(self,x,y,prev_pnt):
        '''
        determine speed (step) for a new track pnt added to an existing track
        
        x, y new pnt coordinates
        
        prev_pnt = [x,y]
        
        remove displacements:
            upward (negative)
            horizontal:
                dx == 0
                dx/dy > 0.9
                
                
        '''
        dy = y - prev_pnt[1]
        
        if dy > self.filament_sp_min_step: #i.e downward and bigger than
            dx = x - prev_pnt[0]
            #vertical test
            if round(dx,2) == 0.0 or abs(dy/dx) > 0.9:
                self.speed_xy_per_track.append(dy)  
#                h = dx**2 + dy**2
#                self.speed_xy_per_track.append(np.sqrt(h))     

#------------------------------------------------------------------------------
    def flow_track_filament_with_plot(self,roi,frame_id,store_values):
        '''
        '''
        self.frame_id = frame_id
        self.roi = cv2.GaussianBlur(roi, (3, 3),0)
        
        roi_test = self.roi.copy()
        if self.filament_sp_filter_blue:
            roi_test[:,:,1:] = 0
            
        self.gray = cv2.cvtColor(roi_test,cv2.COLOR_BGR2GRAY)
        left_start = 0  #max(0,x_left_ave-self.w_ave_range )
        left_end = 30   #x_left_ave + self.w_ave_range
        
        ret_val,thresh_roi_1 = cv2.threshold(self.gray[:,left_start:left_end],0,255,self.thresh_type)

        self.gray[:,left_start:left_end] = thresh_roi_1
        
        if self.filament_sp_use_right:
            right_start = -10
            ret_val,thresh_roi_2 = cv2.threshold(self.gray[:,right_start:],0,255,self.thresh_type)
            self.gray[:,right_start:] = thresh_roi_2
            
            self.gray[:,left_end:right_start] = 0
                
        else:    
            self.gray[:,left_end:] = 0
        frame_gray = self.gray
        #----------------------------------------------------------------------
        speed_y_per_track = np.array([])
        self.speed_xy_per_track = []
        self.num_not_vertical_x_vs_y = 0
        num_y_neg = 0
        
        #----------------------------------------------------------------------
        if self.plot_on:
            print('---------------------------------')
            print('flow_track_filament',self.frame_id)
            plot_update = True #self.frame_id % 50 == 0
            if plot_update:
                num_plots = 4
                plt.figure('track_view')
                for i in range(num_plots):
                    plt.subplot(num_plots,1,i+1)
                    plt.cla()
                    plt.grid(True)
                plt.subplot(num_plots,1,1)
                plt.axhline(0,c='k')
                plt.axhline(self.filament_sp_min_step,c='r')

            if self.first_analysis:
                self.num_tracks_per_frame = []
                self.num_not_vertical_x_vs_y_frame = []
#                self.num_not_vertical_x_abs_frame = []
                self.num_y_neg_frame = []
#                self.spike_indices = []
                self.local_frame_index = 0
                self.num_used_tracks = []

        #----------------------------------------------------------------------
        if self.tracks:
            self.track_points_update(frame_gray)
            array_track = np.asarray(self.tracks)
            if self.plot_on:
                print('flow_track_filament, tracks shape: ', np.shape(array_track), self.frame_id)
                self.num_tracks_per_frame.append(array_track.shape[0])

            for counter,track_list in enumerate(array_track):
                track = np.asarray(track_list)
                #last 2 point only method
                track = track[-2:,:]
                y_step = np.diff(track[:,1])[-1]
                x_step = np.diff(track[:,0])[-1]
                
                #for each track, plot the last step
                if self.plot_on and plot_update:
                    plt.figure('track_view')
                    plt.subplot(num_plots,1,1)
                    plt.plot(counter,y_step,'bo-')
                    plt.plot(counter,x_step,'ko-')
                
                #now test and not use to detemine speed if test fail
                #direction and min size test
                if round(y_step,2) <= self.filament_sp_min_step:
                    num_y_neg += 1
                    if self.plot_on and plot_update:
                        plt.figure('track_view')
                        plt.subplot(num_plots,1,1)
                        plt.axvline(counter,c='k')
                        
                    continue
                    
                #vertical test
                if round(x_step,2) == 0.0:
                    vertical = True
                else:
                    vertical = round(np.abs(y_step)/np.abs(x_step),1) > 0.9
                    
                if not(vertical):
#                    print 'not vertical x > y',vertical_indicies
                    self.num_not_vertical_x_vs_y += 1
                    if self.plot_on and plot_update:
                        plt.figure('track_view')
                        plt.subplot(num_plots,1,1)
                        plt.axvline(counter,c='c')
                    continue


                speed_y_per_track = np.append(speed_y_per_track,y_step) #pxl

                if self.plot_on and plot_update:
                    plt.subplot(num_plots,1,1)
                    plt.plot(counter,y_step,'ro',label = 'Ydiff')
                    plt.title('STEP XY')
                    plt.grid(True)
                    
#

        #----------------------------------------------------------------------
        #update speed
        if speed_y_per_track.size:# or no_movement_check:
            speed_y_per_track_ave = reject_outliers(speed_y_per_track,return_option = 1)
            
        else:
            speed_y_per_track_ave = 0

        
        #update speed
        if self.speed_xy_per_track:
            filament_speed = reject_outliers(np.array(self.speed_xy_per_track),
                                         return_option = 1)
        else:
            filament_speed = 0
        
        self.speed_compare.append(speed_y_per_track_ave)
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        #UPDATE POINTS
        self.get_new_track_points(frame_gray)

        #----------------------------------------------------------------------
        #DEBUG PLOT
        if self.plot_on:
            self.num_used_tracks.append(len(speed_y_per_track))
            print('speed_y_per_track: ',np.shape(speed_y_per_track))
            plt.figure('track_view')
            num_plots = 4
            #-----------------------------------------------------------------
            plt.subplot(num_plots,1,1)
            plt.axhline(speed_y_per_track_ave,c='g')
            
            plt.subplot(num_plots,1,2)
            plt.cla()
            plt.plot(speed_y_per_track,'bo',label = 'track_method')
            plt.axhline(0,c='k')
            plt.axhline(speed_y_per_track_ave,c='g')
            plt.axhline(self.filament_sp_min_step,c='r')
#            plt.axhline(self.speed[-1],c='r')
            plt.title('Speeds for all Tracks for Current Frame')
            plt.grid(True)
#            plt.legend(loc='best', fancybox=True, framealpha=0.5)

            #-----------------------------------------------------------------
            plt.subplot(num_plots,1,3)
            plt.cla()
            
#            plt.plot(self.speed_beta,label = 'correlation method')
            plt.plot(self.speed_compare,label = 'track method a')
            plt.plot(self.speed,label = 'track method b')
#            plt.plot(self.corr_speed_b,label = 'correlation method_b')
            
            

            plt.axhline(0,c='k')
#            for no_mov in self.no_mov_check_indices:
#                plt.plot(no_mov,0,'ko')
#            for spike in self.spike_indices:
#                plt.axvline(spike,c='k')


            plt.title('Speed per Frame')
            plt.grid(True)
            plt.legend()
            #-----------------------------------------------------------------
            plt.subplot(num_plots,1,4)
            plt.cla()

            self.num_not_vertical_x_vs_y_frame.append(self.num_not_vertical_x_vs_y)
            self.num_y_neg_frame.append(num_y_neg)

            plt.plot(self.num_used_tracks,'o-',label = 'used tracks')
            plt.plot(self.num_tracks_per_frame,label = 'tracks')
            plt.plot(self.num_not_vertical_x_vs_y_frame,label = 'vertical:x vs y')
            plt.plot(self.num_y_neg_frame,label = 'y negative')

            plt.axhline(0,c='k')
            plt.title('Tracks per Frame')
            plt.grid(True)
            plt.legend(loc = 'upper left')

            plt.tight_layout()
            #-----------------------------------------------------------------

            cv2.namedWindow('filament_speed')
            cv2.moveWindow('filament_speed',650,0)
            vis = cv2.resize(frame_gray,(self.roi_width*1,self.roi_height*1))
            cv2.imshow('filament_speed',vis)

            cv2.namedWindow('filament_roi')
            cv2.moveWindow('filament_roi',1000,0)
            vis = cv2.resize(self.roi,(self.roi_width*1,self.roi_height*1))
            cv2.imshow('filament_roi',vis)

            self.local_frame_index += 1
            #-----------------------------------------------------------------
            print()

        self.filament_speed = filament_speed
        if store_values:
#           self.speed_beta.append(speed_corr)
            self.speed.append(self.filament_speed)
        self.first_analysis = False
#------------------------------------------------------------------------------
    def flow_track_filament(self,frame_gray):
        '''
        '''
        self.speed_xy_per_track = []
       
        #----------------------------------------------------------------------
        if self.tracks:
            self.track_points_update(frame_gray)

        #----------------------------------------------------------------------
        #update speed
        if self.speed_xy_per_track:
            filament_speed = reject_outliers(np.array(self.speed_xy_per_track),
                                             return_option = 1)
        else:
            filament_speed = 0
        
        #----------------------------------------------------------------------
        #UPDATE POINTS
        self.get_new_track_points(frame_gray)


        return filament_speed
#------------------------------------------------------------------------------
    def speed_w_phase_corr(self,frame_gray):
        '''
        '''
        new_a = np.float32(frame_gray)
        if self.first_analysis:
            self.prev_a = new_a
            phase_corr_xy = [0,0]
            new_speed_y = 0
#            self.speed_f[0] = 0
            
#            self.prev_b = [new_a,new_a,new_a]
#            self.corr_speed_b = [0]
            
        else:
            phase_corr_xy = cv2.phaseCorrelate(self.prev_a,new_a)
            self.prev_a = new_a
            new_speed_y = abs(phase_corr_xy[1])
            if new_speed_y >= self.speed_max_corr:
#                print 'Warning: Speed_w_phase_corr - speed clipped {:.3f}; frame {}'.format(new_speed_y,self.frame_id)
                if self.speed:
                    new_speed_y = self.speed[-1]
                else:
                    new_speed_y = 0

        
        return new_speed_y


#------------------------------------------------------------------------------
    def get_new_track_points(self,frame_gray):
        '''
        '''
        if self.frame_id % self.detect_interval == 0 or self.first_analysis:
            try:
                mask = self.new_pnts_mask.copy()
            except AttributeError:    
                self.new_pnts_mask = np.zeros_like(frame_gray) + 255
                mask = self.new_pnts_mask.copy()
                
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), self.track_pnt_mask_radius, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
            
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])
#                    if self.plot_on:
#                         cv2.circle(self.roi, (x, y), 2, (0, 255, 0), -1)
        #update prev
        self.prev_gray = frame_gray
#------------------------------------------------------------------------------
    def track_points_update(self,frame_gray):#,vis):
        '''
        '''
        img0, img1 = self.prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params) #reverse
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > self.track_len:
                del tr[0]
            new_tracks.append(tr)
            try:
                prev_pnt = tr[-2]
                if self.id_type == 'gear':
                    self.gear_speed_append(x,y,prev_pnt)
                else: #filament
                    self.filament_speed_append(x,y,prev_pnt)
            except IndexError:
                pass #first point in track
#            if self.plot_on:
#                cv2.circle(vis, (x, y), 2, (0, 0,255), -1)
        self.tracks = new_tracks

#        if self.plot_on:
#            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
             #all points in a track
#            for track in self.tracks:
#                for x,y in track:
#                    cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)

#            cv2.imshow('track_pnts',vis)

#------------------------------------------------------------------------------
    def width_ave_range_method(self,return_thresh = False):
        '''
        assume-
            filament is going straight down
            right is a straigt line, profile only on left edge,
            left profile within average left centre line +- range value


        '''
        x_left_ave,x_right_ave = self.delta_colour_method_ave(return_only_left_right = True)

        #START
        if self.plot_on:
            vis = self.roi.copy()

        left_start = max(0,x_left_ave-self.w_ave_range )
        left_end = x_left_ave + self.w_ave_range
        res_val,thresh = cv2.threshold(self.gray[:,left_start:left_end],0,255,self.thresh_type)

        #------------------------------------------------------------------
        x_left_points = thresh.argmax(1)
        self.w_vals = x_right_ave - x_left_points + left_start
        #------------------------------------------------------------------


        if self.plot_on:
            print('----------------------------------------------------------')
            print('width_ave_range_method')
            print(x_left_points.shape,thresh.shape)
            print(x_left_ave,x_right_ave)
#            cv2.polylines(self.result_img, [points], 1, (255))
#            vis = vis[:,left_start:left_end]

            cv2.line(vis,(x_left_ave,0),(x_left_ave,self.roi_height),(0,0,0))
            cv2.line(vis,(left_start,0),(left_start,self.roi_height),(255,0,255))
            cv2.line(vis,(left_end,0),(left_end,self.roi_height),(255,0,255))
            cv2.line(vis,(x_right_ave,0),(x_right_ave,self.roi_height),(0,0,0))

            for y_index,x_left_point in enumerate(x_left_points):
                vis[y_index,x_left_point + left_start] = (0,0,255)

            cv2.imshow('thresh_area',thresh)
            cv2.imshow('vis', cv2.resize(vis,(vis.shape[1]*2,vis.shape[0]*2)))

            if False:
                plt.figure('width_ave_range_method')

                plt.cla()
                plt.grid(True)
                num_plots = 1
                plt.subplot(num_plots,1,1)
                plt.plot(self.w_vals,label = 'w_vals')
                plt.plot(self.w_vals_f,label = 'w_vals_f')
    #            plt.plot(self.w_vals_x,label = 'w_vals_x')
                plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
                plt.tight_layout()

    #            plt.figure('spectrum')
#                plot_spectrum(self.w_vals_f,'width_ave_range_method-spectrum')
        if return_thresh:
            return thresh

#------------------------------------------------------------------------------
    def width_subpixel_method(self):
        '''
        The filament width ROI has 2 sub-ROIs - left and right.
        
        Detect edge in each sub-ROI. 
        Subpixel resolution achieved by zooming image.
        
        '''
        #SUB ROI AREA
        gray_left = self.gray[:,self.w_vline_left_roi-self.w_detection_half_width:self.w_vline_left_roi + self.w_detection_half_width]
        gray_right = self.gray[:,self.w_vline_right_roi - self.w_detection_half_width:self.w_vline_right_roi + self.w_detection_half_width]
        
        #ZOOM
        gray_left = cv2.resize(gray_left,self.w_zoom_left)
        gray_right = cv2.resize(gray_right,self.w_zoom_right)
        
        #ESTIMATE
        edge_left = self.width_est_edge_zoomed(gray_left,left_edge = True)
        edge_right = self.width_est_edge_zoomed(gray_right,left_edge = False)
        
        edge_left_pnt = edge_left/self.w_zf + self.w_vline_left_roi - self.w_detection_half_width
        edge_right_pnt = edge_right/self.w_zf + self.w_vline_right_roi - self.w_detection_half_width
        
        
        self.width_value = edge_right_pnt - edge_left_pnt
        self.width_left_pnt = edge_left_pnt
        self.width_right_pnt = edge_right_pnt
        
        #PLOT 
        if self.plot_on:
            self.width_subpixel_plot(gray_left,gray_right)
#------------------------------------------------------------------------------
    def width_subpixel_plot(self,gray_left,gray_right):
        '''
        '''
        col_ave_left_gray = np.average(gray_left,0)
#        col_ave_right_gray = np.average(gray_right,0)
        
#        diff_colave_left_gray = np.diff(col_ave_left_gray)
#        diff_colave_right_gray = np.diff(col_ave_right_gray)
        
#        max_left = diff_colave_left_gray.argmax()
#        max_right = diff_colave_left_gray.argmin()
        
        self.axes_ave_cols_left.plot(col_ave_left_gray[:],'gray')
        #width and edges
#        w_sides.plot(w_left,'r',w_right,'b')
#        w_val.plot(width)
#        
#        #LEFT
#        ave_over_cols_left.plot(col_ave_left_gray[:],'gray')
#        ave_over_cols_left.axvline(vline_left_roix*zf,c = 'b',linestyle = '--',label = 'vline')
#        
#        
#        
#        diff_over_avecol_left.plot(diff_colave_left_gray,'gray')
#        diff_over_avecol_left.axvline(vline_left_roix*zf,c = 'b',linestyle = '--',label = 'vline')
#        diff_over_avecol_left.axvline(max_left,c = 'k',linestyle = '--',label = 'min/max')
#        diff_over_avecol_left.axvline(edge_left,c = 'r',linestyle = '--',label = 'sub_edge')
#        
#        
#        
#        #RIGHT
#        ave_over_cols_right.plot(col_ave_right_gray[:],'gray')
#        ave_over_cols_right.axvline(vline_right_roix*zf,c = 'b',linestyle = '--',label = 'vline')
#        
#        
#        
#        
#        diff_over_avecol_right.plot(diff_colave_right_gray,'gray')
#        diff_over_avecol_right.axvline(vline_right_roix*zf,c = 'b',linestyle = '--',label = 'vline')
#        diff_over_avecol_right.axvline(max_right,c = 'k',linestyle = '--',label = 'min/max')
#        diff_over_avecol_right.axvline(edge_right,c = 'r',linestyle = '--',label = 'sub_edge')
#        
#        
#        
#        #ANNOTATE
#        ave_over_cols_left.set_title('ave_over_cols_left')
#        diff_over_avecol_left.set_title('diff_over_avecol_left')
#        ave_over_cols_right.set_title('ave_over_cols_right')
#        diff_over_avecol_right.set_title('diff_over_avecol_right')
#        diff_over_avecol_left.legend()
#        diff_over_avecol_left.legend()
#        ave_over_cols_right.legend()
#        diff_over_avecol_right.legend()

#------------------------------------------------------------------------------
    def width_est_edge_zoomed(self,gray_roi,left_edge = True):
        '''
        gray_roi, roi over expected edge area, must be grayscale.
        
        '''
        #ESTIMATE
        ave_over_columns = np.average(gray_roi,0) #for gray scale ROI
        diff_ave = np.diff(ave_over_columns) #first derivative
        
        #arg max/min of first derivative
        if left_edge:
            turn_pnt = diff_ave.argmax()
        else:
            turn_pnt = diff_ave.argmin()
        
        #arg max average value
        max_ave = diff_ave[turn_pnt-self.w_zf:turn_pnt+self.w_zf]
        if len(max_ave) > 0:
            max_ave_centre = np.average(max_ave, weights=max_ave)
            edge_pos = turn_pnt-self.w_zf + max_ave_centre
        else:
            edge_pos = turn_pnt
        
        return edge_pos        
#------------------------------------------------------------------------------
'''
'''

if __name__ == '__main__':
#    ROIA()
    print('main')
#    main()
