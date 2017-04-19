# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 13:37:47 2016


"""
__version__ = "1.0"
__license__ = "MIT"
__author__  ="GP Greeff"

import numpy as np
import cv2
import matplotlib.pyplot as plt
#from math import sqrt as math_sqrt

from QtLoadSaveObjects import make_config_dict
np.seterr(invalid='raise')


#==============================================================================
class SmallAreaDenseMethod():
    """
    Apply Dense Optical Flow Method to the a small area ROI.
        
    Args:
        frame (np.array): captured image to init method. 
        roi_pnts (tuple): Region of Interest - (x1,y1,x2,y2), coords relative to main frame
        
    
    """
    def __init__(self,frame,roi_pnts,**kwargs):
        #self info
        self.name_id = kwargs.get('name_id','SDM_0')
        
        #debug control
        self.plot_on = kwargs.get('plot_on',False) # plots data
        self.draw_on = kwargs.get('draw_on',False) #draws addionational screens
        self.collect_on = kwargs.get('collect_on',False) #collects data in list
        self.collect_on = self.collect_on or self.plot_on or self.draw_on
        
        #plotting placeholder, for debuging/testing
        self.plot_axes = None
        
        #detection parameters
        self.min_flow_speed = kwargs.get('min_flow_speed',0.1) #pxl/frame
        self.flow_rounding_decimals = kwargs.get('flow_rounding_decimals',3) #pxl/frame
                                    
        #START INIT
        #roi region,, size parameters and disk_mask - recalculate on change of ROI/params etc.
        self.update_roi_params(roi_pnts,**kwargs)

        #init first frames
        frame_roi = frame[self.roi_y1:self.roi_y2,self.roi_x1:self.roi_x2]
        self.pre_process(frame_roi) #determine frame_gray
        self.prev_gray = self.frame_gray.copy()
        self.flow = None
        
        #data collection
        self.speed_step_current = 0
        self.time_step_current = 0
        
        #addtional data for review, used with collect_on/plot_on
        self.speed_step = [] #average speed in pxl/frame
        self.speed_mms = [] #speed converted to mm/s
        self.time_step = [] #processing time step as measured
        self.num_rejected = []
        self.num_total = []
        self.eval_vline = 8
        self.eval_hist = []

        
        #set apply method
        w_timestep = kwargs.get('w_timestep',False)
        if w_timestep:
            self.apply = self.apply_w_timestep
        else:
            self.apply = self.apply_wo_timestep
            
#        print()
#        print('cal_factor: {}'.format(self.cal_factor))
#        print('fps: {}'.format(self.fps))
#        print('speed_conversion: {}'.format(self.speed_conversion))


    def update_roi_params(self,roi_pnts,**kwargs):
        """
        Update roi parameters and dependent vars
        """
        self.roi_x1,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        self.roi_pnt1 = (self.roi_x1,self.roi_y1)
        self.roi_pnt2 = (self.roi_x2,self.roi_y2)
        self.roi_height = self.roi_y2 - self.roi_y1
        self.roi_width  = self.roi_x2 - self.roi_x1
        
        #speed conversion
        self.fps = kwargs.get('cam_fps',15)
        self.cal_factor = kwargs.get('cal_factor',0.0212) 
        self.speed_conversion = self.cal_factor*self.fps
        
        #draw
        step = 4
        self.y_ix,self.x_ix = np.mgrid[step/2:self.roi_height :step,step/2:self.roi_width:step].reshape(2,-1)
        self.y_ix = self.y_ix.astype(np.int)
        self.x_ix = self.x_ix.astype(np.int)
        
        #detection parameters
        #min_flow_resolution in mm/s
        self.min_flow_speed = kwargs.get('min_flow_speed',1*self.fps/15) #pxl/frame
        self.min_flow_res = kwargs.get('min_flow_resolution',0.05)/self.speed_conversion #pxl/frame
        self.ymax_min_size = kwargs.get('ymax_min_size',5) #min size of array
        self.max_x_motion = kwargs.get('max_x_motion',1 * 15.0/self.fps) #max pxl movement in x direction for ydir speed
        
#        self.exp_flow_dir = kwargs.get('exp_flow_dir','+y') #expected flow_dir, +- x/y
        

    def pre_process(self,frame):
        """
        Preprocess raw frame
        """
        self.frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.frame_gray = cv2.GaussianBlur(self.frame_gray, (3, 3),0)
        
    def apply_w_timestep(self,frame,*args,**kwargs):
        """
        Estimate speed and measure timestep for whole process
        """   
        e1 = cv2.getTickCount()
        self.apply_wo_timestep(frame)
        e2 = cv2.getTickCount()
        self.time_step_current = (e2 - e1)/ cv2.getTickFrequency()
        
        #testing data colection
        if self.collect_on:
            self.time_step.append(self.time_step_current)
            self.speed_step.append(self.speed_step_current)
            self.speed_mms.append(self.speed_step_current*self.speed_conversion)
#            self.num_rejected.append(np.size(self.v0) - np.count_nonzero(self.v0))
#            self.num_total.append(np.size(self.v0))
            if self.draw_on:
                flow_img = self.draw_flow()
                cv2.imshow("Optical flow",flow_img) # plot the flow vectors
            
            if self.plot_on:
                self.plot_data()
        
    def apply_wo_timestep(self,frame,*args,**kwargs):
        """
        Estimate speed, without measuring the time step
        
        cv2.calcOpticalFlowFarneback(prev, next, flow, 
                                     pyr_scale, levels, winsize, 
                                     iterations, poly_n, poly_sigma, 
                                     flags) → flow
        Parameters Descrpiton:	
            prev – first 8-bit single-channel input image.
            next – second input image of the same size and the same type as prev.
            flow – computed flow image that has the same size as prev and type CV_32FC2.
            
            pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image; 
                        pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
                        
            levels – number of pyramid layers including the initial image; 
                     levels=1 means that no extra layers are created and only the original images are used.
            
            winsize – averaging window size; 
                      larger values increase the algorithm robustness to image noise 
                      and give more chances for fast motion detection, 
                      but yield more blurred motion field.
                      
            iterations – number of iterations the algorithm does at each pyramid level.
            
            poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel; 
                     larger values mean that the image will be approximated with smoother surfaces, 
                     yielding more robust algorithm and more blurred motion field, 
                     typically poly_n =5 or 7.
                     
            poly_sigma – standard deviation of the Gaussian that is used to smooth 
                         derivatives used as a basis for the polynomial expansion; 
                         for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, 
                         a good value would be poly_sigma=1.5.
                         
            flags –
                operation flags that can be a combination of the following:
                OPTFLOW_USE_INITIAL_FLOW 
                  uses the input flow as an initial flow approximation.
                
                OPTFLOW_FARNEBACK_GAUSSIAN 
                  uses the Gaussian (winsize x winsize) 
                  filter instead of a box filter of the same size for optical flow 
                  estimation; usually, this option gives z more accurate flow than 
                  with a box filter, at the cost of lower speed; normally, 
                  winsize for a Gaussian window should be set to a larger value 
                  to achieve the same level of robustness.

        """
        self.pre_process(frame)
#        self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,None,
#                                                 0.5,3,15,
#                                                 3,5,1.2,0)

#        if self.use_prev_flow and self.flow is not None:
#            self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,self.flow,
#                                                     pyr_scale = 0.5,levels = 3,winsize = 11,iterations = 3,
#                                                     poly_n = 7,poly_sigma = 1.5, flags = cv2.OPTFLOW_USE_INITIAL_FLOW)

    
        self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,self.flow,
                                                 pyr_scale = 0.5,levels = 3,winsize = 11,iterations = 3,
                                                 poly_n = 7,poly_sigma = 1.5, 
                                                 flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        self.prev_gray = self.frame_gray 
        self.est_speed_in_ydir()

    def est_speed_in_ydir(self):
        """        
        Estimate flow speed in y direction (y downwards positive). 
        
        fx and fy is the flow vectors for each pixel/ evaluated point in the ROI.
        
        Steps:
            0. Round out small values, less than 'flow_rounding_decimals'
            1. Select the maximum flow points (indices) in y.
            2. Filter out indices with 'large' x-motion - larger than max_x_motion
            3. Filter out speed below 'min_flow_speed'
            4. Estimate the speed if the number of remaining points more than ymax_min_size
                a- by taking the median of the filtered ymax
                b- otherwise assume it is 0
            

        """
        self.flow.round(self.flow_rounding_decimals,out = self.flow)
        
        fx,fy = self.flow.T

        ymax_arg = np.argmax(fy,axis = 0)
        fx_ymax = fx[ymax_arg]
        fy_ymax = fy[ymax_arg]

        #x-filter
        x_sel = np.abs(fx_ymax) < self.max_x_motion
        fy_ymax = fy_ymax[x_sel]
        ymax = fy_ymax[fy_ymax > self.min_flow_speed] #min speed filter
        
        #no x-filter
#        ymax = np.max(fy,axis = 0)
#        ymax = ymax[ymax > self.min_flow_speed] #min speed filter

        #estimate on fy max only
        if ymax.size > self.ymax_min_size:
            ymax_mid = np.median(ymax)

            
            #other options
#            ymax_ave = np.average(ymax) #average
#            ymax_mean = np.mean(ymax)   #mean

#            ymax.sort()
#            ytop10 = int(ymax.size*0.1) 
#            ymax_max = np.average(ymax[-ytop10:])


        else:
            ymax_mid = 0
            
        #estimate with v = sqrt(fx^2 + fy^2)
#        try:
#            self.v_sqr = fx*fx + fy*fy 
#            
#        except FloatingPointError:
#            #Underflow: result so close to zero that some precision was lost.
#            print('FloatingPointError, underflow encountered in multiply')
#        
#        self.v_sqrt =  np.sqrt(self.v_sqr)
#        #remove small flow speed values
#        self.v0 = self.v_sqrt >= self.min_flow_speed
#        if np.any(self.v0): 
#            #mean of all selected velocities
##            v_ave = np.mean(self.v_sqrt[self.v0])
#            
#        else:
#            v_ave = 0

            
        self.speed_step_current = ymax_mid #store result
        
               
    def draw_flow(self,resize = 4):
        """ 
        Draws optical flow at sample points spaced 'step' pixels apart. 
        
        Args:
            resize (int): resize the ROI for visualisation
            
        Returns:
            vis (array): modified ROI which can be shown to user
        """
        
        vis = cv2.cvtColor(self.frame_gray,cv2.COLOR_GRAY2BGR)
#        vis[x_indices,y_indices] = (0,0,255)

        vis = cv2.resize(vis,(self.frame_gray.shape[1]*resize,self.frame_gray.shape[0]*resize))
        
        #draw all lines
        
        fx,fy = self.flow[self.y_ix,self.x_ix].T
        

        lines = np.vstack([self.x_ix*resize,self.y_ix*resize,
                           (self.x_ix+fx)*resize,(self.y_ix+fy)*resize]).T.reshape(-1,2,2)
        lines = np.int32(lines)
        
        #draw eval vline
        cv2.line(vis,(self.eval_vline*resize,0),(self.eval_vline*resize,vis.shape[0]),(0,0,255))
        
        #draw flow lines
        for (x1,y1),(x2,y2) in lines:
            cv2.line(vis,(x1,y1),(x2,y2),(255,255,0),1)
        
        return vis

    def plot_init(self):
        """
        """
        self.ax_pnts_acc = self.plot_axes[0,0]
        self.ax_flow_vector_r = self.plot_axes[0,1]
        self.ax_flow_vector_x = self.plot_axes[1,0]
        self.ax_flow_vector_y = self.plot_axes[1,1]
        self.ax_vline = self.plot_axes[2,0]
        self.ax_sel = self.plot_axes[2,1]
        

    def plot_data(self):
        """
        """
#        v_sel = self.v_sqrt[self.v0]
#        self.eval_hist.append(self.v_sqrt[:,self.eval_vline])
#        if v_sel.size:
#            v_1 = reject_outliers(v_sel, m = 2.,return_option = 2)
#            v_sel.sort()
#        else:
#            v_1 = 0
            
#        fx_vline,fy_vline = self.flow[self.eval_vline,:].T
        
        fx,fy = self.flow.T
        

#        if self.exp_flow_dir == '+y': #down
#            v_dir = fy[fy > 0]
#            xdir = fx[fy > 0]
#            ydir = fy[fy > 0]
        ymax_arg = np.argmax(fy,axis = 0)
#            fx_ymax = fx[ymax_arg]
        fy_ymax = fy[ymax_arg]
#            x_sel = np.abs(fx_ymax) < 0.01
#            fy_ymax = fy_ymax[x_sel]
            
            
        ymax = fy_ymax[fy_ymax > self.min_flow_speed]
        
        if ymax.size > 10:
            ymax.sort()
            ymax_ave = np.average(ymax)
            ymax_mid = np.median(ymax)
            ytop10 = int(ymax.size*0.1) 
            y_max_max = np.average(ymax[-ytop10:])
        else:
            ymax_ave = 0
            y_max_max = 0
            ymax_mid = 0
                
        ymax_mid  = y_max_max

#        v_dir = np.abs(v_dir)
#        v_dir = v_dir[v_dir > self.min_flow_speed]
#        if v_dir.size > 0:
#            v_dir_ave = np.mean(v_dir)
#        else:
#            v_dir_ave = 0
        
        
#        self.ax_pnts_acc.cla()
#        self.ax_pnts_acc.plot(self.num_total,'o-',label = 'num_total')
#        self.ax_pnts_acc.plot(self.num_rejected,'o-',label = 'num_rejected')
#        self.ax_pnts_acc.legend()
#        
##        fx,fy = self.flow.T
#        
#        self.ax_flow_vector_r.cla()
#        self.ax_flow_vector_r.plot(v_sel*self.speed_conversion,'+',label = 'v_sel sorted')
#        self.ax_flow_vector_r.axhline(self.min_flow_speed*self.speed_conversion,c='r',ls = '--')
#        self.ax_flow_vector_r.axhline(self.speed_step_current*self.speed_conversion,c='k',ls = '--')
#        self.ax_flow_vector_r.axhline(v_1*self.speed_conversion,c='gray',ls = '--',label = 'v1')
#        self.ax_flow_vector_r.axhline(v_dir_ave*self.speed_conversion,c='g',ls = '--',label = 'v_dir_ave')
#        
#        self.ax_flow_vector_r.legend()
#        self.ax_flow_vector_r.axis(ymin = 0,ymax = 4)
#        
#        self.ax_vline.cla()
#        if len(self.eval_hist) >= 3:
#            self.ax_vline.plot(self.eval_hist[-3]*self.speed_conversion,'gray') 
#        if len(self.eval_hist) >= 2:
#            self.ax_vline.plot(self.eval_hist[-2]*self.speed_conversion,'b') 
#        if len(self.eval_hist) >= 1:
#            self.ax_vline.plot(self.eval_hist[-1]*self.speed_conversion,'r') 
#        
#                
#        self.ax_flow_vector_x.cla()
#        self.ax_flow_vector_x.plot(fx_vline,'o-',label = 'fx_vline')
#        self.ax_flow_vector_x.plot(fy_vline,'o-',label = 'fy_vline')
#        self.ax_flow_vector_x.legend()
#        
#        self.ax_flow_vector_y.cla()
#        xdir_flat = xdir.flatten()
#        ydir_flat = ydir.flatten()
#        self.ax_flow_vector_y.plot(xdir_flat,'o-',label = 'xdir')
#        self.ax_flow_vector_y.plot(ydir_flat,'o-',label = 'ydir')
#        
#        x_sel = np.abs(xdir_flat) < 0.1
#        y_sel = ydir_flat > self.min_flow_speed
#        ysel = x_sel & y_sel
#        ysel_ix = ysel.nonzero()[0]
#        ydir_flat_sel = ydir_flat[ysel]
#        if ydir_flat_sel.size > 0:
#            ydir_sel_ave = np.average(ydir_flat_sel)
#        else:
#            ydir_sel_ave = 0
#        self.ax_flow_vector_y.plot(ysel_ix,ydir_flat[ysel],'ro',label = 'ysel')
#        self.ax_flow_vector_y.axhline(ydir_sel_ave,c='r',ls = '--',label = 'ydir_sel_ave')
#        self.ax_flow_vector_y.legend()
#        
        
        self.ax_sel.cla()
        self.ax_sel.plot(ymax,'o')
        self.ax_sel.axhline(ymax_ave,c='g',ls = '--',label = 'ymax_ave')
        self.ax_sel.axhline(y_max_max,c='k',ls = '--',label = 'y_max_max')
        self.ax_sel.axhline(ymax_mid,c='r',ls = '--',label = 'ymax_mid')
        self.ax_sel.legend()
        self.ax_sel.axis(ymin=0,ymax = 10)
        

#==============================================================================
#==============================================================================
class FilamentWidthMethod():
    """
    Estimate the filament width.
    
    Args:
        frame (array): roi area in main frame
        roi_pnts (tuple): (x1,y1,x2,y2),  coords relative to main frame
            
    
    Method:
        Assume user aligns roi so that vline (vertical alignment line) is 
        approx. on the left and right edge of the filament.
        
        Then extract two sub-rois, left and right, assuming the filament edges
        is in this area - within a certain distance from the left and right vline.
        
        Then zoom (magnify) these areas.
        
        Estimate edges using min/max of the first derivative 
        along the average over the x cols.
        
        Convert back for final answer.
    
    """
    def __init__(self,frame,roi_pnts,**kwargs):
        #info
        self.name_id = kwargs.get('name_id','Width_0')
        
        #debug/testing control
        self.plot_on = kwargs.get('plot_on',False) # plots data
        self.draw_on = kwargs.get('draw_on',False) #draws addionational screens
        self.collect_on = kwargs.get('collect_on',False) #collects data in list
        self.collect_on = self.collect_on or self.plot_on or self.draw_on
        
        #plotting placeholder, for debuging/testing
        self.plot_axes = None
        
        #configs
        self.configs = kwargs
        self.material =self.configs.get('mat_colour','white').lower()
        
 
        #sub_pixel method
        self.w_zf = self.configs.get('w_zf',4) #zoomfactor
        self.w_detection_half_width = self.configs.get('w_detection_half_width',10)
        self.w_vline_left_border = self.configs.get('w_vline_left_border',20)
        self.w_vline_right_border = self.configs.get('w_vline_right_border',15)
        
        #data collection
        self.time_step_current = 0
        
        #addtional data for review, used with collect_on/plot_on
        if self.collect_on:
            self.width_history = []
            self.time_step = []
            
        #START INIT
        #roi region, size parameters - recalculate on change of ROI/params etc.
        self.roi_x1 ,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        roi_bgr = frame[self.roi_y1:self.roi_y2,self.roi_x1:self.roi_x2]
        self.roi = cv2.GaussianBlur(roi_bgr, (3, 3),0)
        self.gray = cv2.cvtColor(self.roi,cv2.COLOR_BGR2GRAY)
        self.update_roi_params(roi_pnts,**kwargs)

        #set apply method
        w_timestep = kwargs.get('w_timestep',False)
        if w_timestep:
            self.apply = self.apply_w_timestep
        else:
            self.apply = self.apply_wo_timestep


    def update_roi_params(self,roi_pnts,**kwargs):
        """
        Update roi parameters and dependent vars
        """
        self.cal_factor = kwargs.get('cal_factor',0.0212) 
        
        #width values
        self.width_value = self.configs.get('width_default',150)  #current value for filament width

        
        #roi
        self.roi_x1 ,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        self.roi_pnt1 = (self.roi_x1,self.roi_y1)
        self.roi_pnt2 = (self.roi_x2,self.roi_y2)
        self.roi_height = self.roi_y2 - self.roi_y1
        self.roi_width  = self.roi_x2 - self.roi_x1
        
        self.w_roi_middle = int((self.roi_width)/2) #middle of roi rectangle, approx middle of filament
        self.w_roi_middle_left = int(self.w_roi_middle/2) # middle of left half
        self.w_roi_middle_right = int(self.w_roi_middle + self.w_roi_middle/2) #middle of right half

        self.w_vline_left_roi = self.w_vline_left_border #x-position of vline in roi
        self.w_vline_right_roi = self.roi_width  - self.w_vline_right_border #x-position of vline right 
        
        self.width_left_pnt = self.w_vline_left_roi #min point - left edge
        self.width_right_pnt = self.w_vline_right_roi #max point - right edge
        
#       print(self.w_vline_left_roi,self.w_detection_half_width)
        #assume left and right edge are in these sub-roi areas
        self.left_left = self.w_vline_left_roi - self.w_detection_half_width
        self.left_right = self.w_vline_left_roi + self.w_detection_half_width
        roi_left = self.roi[:,self.left_left:self.left_right]
        
        self.right_left = self.w_vline_right_roi - self.w_detection_half_width
        self.right_right = self.w_vline_right_roi + self.w_detection_half_width
        roi_right = self.roi[:,self.right_left : self.right_right]
    
        self.w_vline_left_roix = roi_left.shape[1] - self.w_detection_half_width
        self.w_vline_right_roix = self.w_detection_half_width
        
        self.w_zoom_left = (roi_left.shape[1]*self.w_zf,  roi_left.shape[0]*self.w_zf)
        self.w_zoom_right= (roi_right.shape[1]*self.w_zf, roi_right.shape[0]*self.w_zf)

        
    def apply_w_timestep(self,frame,*args,**kwargs):
        """
        Estimate speed and measure timestep for whole process
        """   
        e1 = cv2.getTickCount()
        self.apply_wo_timestep(frame)
        e2 = cv2.getTickCount()
        self.time_step_current = (e2 - e1)/ cv2.getTickFrequency()
        if self.collect_on:
            self.time_step.append(self.time_step_current)
            self.width_history.append(self.width_value)
            if self.plot_on:
                self.plot_data()
            if self.draw_on:
                self.draw_edges()
        
    def apply_wo_timestep(self,roi_bgr,*args,**kwargs):
        """
        Estimate width, without measuring the time step
        
        
        """
        #pre-process
        self.roi = cv2.GaussianBlur(roi_bgr, (3, 3),0)
        self.gray = cv2.cvtColor(self.roi,cv2.COLOR_BGR2GRAY)

        #process
        self.width_subpixel_method()
                     
    def width_subpixel_method(self):
        """
        The filament width ROI has 2 sub-ROIs - left and right.
        
        Detect edge in each sub-ROI. 
        Subpixel resolution achieved by zooming image.
        
        """
        #SUB ROI AREA
        gray_left = self.gray[:,self.left_left:self.left_right]
        gray_right = self.gray[:,self.right_left : self.right_right]
        
        #ZOOM
        self.gray_left = cv2.resize(gray_left,self.w_zoom_left)
        self.gray_right = cv2.resize(gray_right,self.w_zoom_right)
        
        #ESTIMATE
        self.edge_left = self.width_est_edge_zoomed(self.gray_left,left_edge = True)
        self.edge_right = self.width_est_edge_zoomed(self.gray_right,left_edge = False)
        
        #SHIFT
        self.width_left_pnt = self.edge_left/self.w_zf + self.left_left
        self.width_right_pnt = self.edge_right/self.w_zf + self.right_left
        
        #STORE
        self.width_value = self.width_right_pnt - self.width_left_pnt
       
    def width_est_edge_zoomed(self,gray_roi,left_edge = True):
        """
        Args:
            gray_roi (array): roi over expected edge area, must be grayscale.
            left_edge (bool): min (False) or max (True) in first derivitive
        
        """
        #ESTIMATE
        ave_over_columns = np.average(gray_roi,0) #for gray scale ROI
        diff_ave = np.diff(ave_over_columns) #first derivative
        
        #arg max/min of first derivative
        if left_edge:
            turn_pnt = diff_ave.argmax()
        else:
            turn_pnt = diff_ave.argmin()
        
        #arg max average value
#        max_ave = diff_ave[turn_pnt-self.w_zf:turn_pnt+self.w_zf]
#        if len(max_ave) > 0:
#            max_ave_centre = np.average(max_ave, weights=max_ave)
#            edge_pos = turn_pnt-self.w_zf + max_ave_centre
#        else:
#            edge_pos = turn_pnt
#        return edge_pos    
        return turn_pnt

    def plot_init(self):
        """
        """
        self.axes_all = self.plot_axes[0,0]
        self.axes_lr = self.plot_axes[0,1]
        
        self.axes_avecols_left  = self.plot_axes[1,0]
        self.axes_avecols_right = self.plot_axes[1,1]
        
        self.axes_diffcols_left  = self.plot_axes[2,0]
        self.axes_diffcols_right = self.plot_axes[2,1]

    def plot_data(self):
        """
        """
        all_ave = np.average(self.gray,0)
        
        gray_left_unz = self.gray[:,self.left_left:self.left_right]
        gray_right_unz = self.gray[:,self.right_left : self.right_right]
        xpnts_left = np.arange(gray_left_unz.shape[1])
        xpnts_right = np.arange(gray_right_unz.shape[1]) + gray_left_unz.shape[1]
        
        
        colave_left_unz = np.average(gray_left_unz,0) 
        colave_right_unz = np.average(gray_right_unz,0)
        
        #zoomed
        colave_left = np.average(self.gray_left,0) 
        colave_right = np.average(self.gray_right,0)
        
        diff_left  = np.diff(colave_left)
        diff_right = np.diff(colave_right)
        
        max_left = diff_left.argmax()
        max_right = diff_right.argmin()
        
        #PLOT WHOLE ROI
        self.axes_all.cla()
        self.axes_all.plot(all_ave,'gray',label = 'all_ave')
        
        self.axes_all.axvline(self.w_roi_middle,c = 'k',linestyle = '--',label = 'w_roi_middle')
        self.axes_all.axvline(self.w_roi_middle_left,c = 'gray',linestyle = '-.',label = 'w_roi_middle_left')
        self.axes_all.axvline(self.w_roi_middle_right,c = 'gray',linestyle = '-.',label = 'w_roi_middle_right')
        
        self.axes_all.axvline(self.left_left,c = 'gray',linestyle = '--',label = 'left_left')
        self.axes_all.axvline(self.left_right,c = 'gray',linestyle = '--',label = 'left_right')
        
        self.axes_all.axvline(self.right_left,c = 'gray',linestyle = '--',label = 'right_left')
        self.axes_all.axvline(self.right_right,c = 'gray',linestyle = '--',label = 'right_right')

        self.axes_all.axvline(self.width_left_pnt,c = 'r',linestyle = '-',label = 'width_left_pnt')
        self.axes_all.axvline(self.width_right_pnt,c = 'r',linestyle = '-',label = 'width_right_pnt')
       
#        self.axes_all.set_title('whole roi ave_cols')
        
        #PLOT left and right, before zoom
        self.axes_lr.cla()
        self.axes_lr.plot(xpnts_left,colave_left_unz)
        self.axes_lr.plot(xpnts_right,colave_right_unz)
        
        #PLOT LEFT ZOOMED
        self.axes_avecols_left.cla()
        self.axes_avecols_left.plot(colave_left,'gray',label = 'colave_left')
        self.axes_avecols_left.axvline(self.edge_left,c = 'r',linestyle = '--',label = 'left_edge_pos')
        
        self.axes_diffcols_left.cla()
        self.axes_diffcols_left.plot(diff_left)
        self.axes_diffcols_left.axvline(max_left,c = 'k',linestyle = '--',label = 'max_left')
        self.axes_diffcols_left.axvline(self.edge_left,c = 'r',linestyle = '--',label = 'left_edge_pos')
        
        #PLOT RIGHT ZOOMED
        self.axes_avecols_right.cla()
        self.axes_avecols_right.plot(colave_right,'gray',label = 'colave_right')
        self.axes_avecols_right.axvline(self.edge_right,c = 'r',linestyle = '--',label = 'right_edge_pos')
        
        self.axes_diffcols_right.cla()
        self.axes_diffcols_right.plot(diff_right)
        self.axes_diffcols_right.axvline(max_right,c = 'k',linestyle = '--',label = 'max_right')
        self.axes_diffcols_right.axvline(self.edge_right,c = 'r',linestyle = '--',label = 'right_edge_pos')
        
    def draw_edges(self):
        """
        Draws edges on ROI.
        """
        gray = cv2.cvtColor(self.gray,cv2.COLOR_GRAY2BGR)
        edge_left= int(round(self.width_left_pnt))
        edge_right= int(round(self.width_right_pnt))
        cv2.line(gray,(edge_left,0),(edge_left,gray.shape[1]),(0,0,255))
        cv2.line(gray,(edge_right,0),(edge_right,gray.shape[1]),(0,0,255))
        gray = cv2.resize(gray,(gray.shape[0]*16,gray.shape[1]*1))
        cv2.imshow('gray',gray)
        
        edge_left_zoomed = int(round(self.edge_left))
        gray_left = cv2.cvtColor(self.gray_left,cv2.COLOR_GRAY2BGR)
        cv2.line(gray_left,(edge_left_zoomed,0),(edge_left_zoomed,gray_left.shape[0]),(0,0,255))
        cv2.imshow('gray_left',gray_left)
        
        edge_right_zoomed = int(round(self.edge_right))
        gray_right = cv2.cvtColor(self.gray_right,cv2.COLOR_GRAY2BGR)
        cv2.line(gray_right,(edge_right_zoomed,0),(edge_right_zoomed,gray_right.shape[0]),(0,0,255))
        cv2.imshow('gray_right',gray_right)
        
        cv2.moveWindow('gray',700,0)
        cv2.moveWindow('gray_left',700 + gray.shape[1] +50,0)
        cv2.moveWindow('gray_right',700 + gray.shape[1] + gray_left.shape[1] +50,0)

        

#==============================================================================
class TestROI():
    """
    Apply_methods, list of dicts, defining which methods to apply.
        
    """
    def __init__(self,video_fullname,video_config,apply_methods,**kwargs):
        
        #options
        self.profile = kwargs.get('profile',False) # run profiling, do not plot
        self.video_playback = True
        
        #data
        self.video_fullname = video_fullname
        self.config = video_config
        
        #SOURCE
        self.init_video()
        
        #first frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
        retval, frame = self.video_capture.read()
        
        #ROI SETUP
        self.init_windows()
        
        #DETECTION METHODS SETUP
        self.method_types = {} #method and roi points
        self.method_types['FWM'] = [FilamentWidthMethod,'width']
        self.method_types['gearSDM'] = [SmallAreaDenseMethod,'gearSDM']
        self.method_types['filSDM'] = [SmallAreaDenseMethod,'filSDM']
                         
        self.methods = [] #list of methods to apply
        self.draw_roi_rects = {} #roi rectangles to draw 
        self.draw_roi_circles = {} 
        collect_data_all = not(self.profile)
        if not(isinstance(apply_methods,list)):
            apply_methods = [apply_methods]

        for index,new_method_kwarg in enumerate(apply_methods):
            method_name = new_method_kwarg.pop('method_name')
            method_id = new_method_kwarg.pop('id',index)
            
            method,method_type = self.method_types[method_name]
            method_roi_pnts,method_spec_kwarg = self.get_roi_coords(method_type)
            
            name_id = '{}_{}_{}'.format(method_name,method_id,index)
            
            new_method_kwarg.update(self.config_dict['vision'])
            new_method_kwarg.update(self.config_dict['camera'])
            new_method_kwarg.update(method_spec_kwarg)

            new_method = method(frame,method_roi_pnts,
                              name_id = name_id,
                              collect_on = collect_data_all,
                              **new_method_kwarg)
            
            self.methods.append(new_method)
            new_method.method_name = method_name
            new_method.method_type = method_type
            
        #PLOT SETUP
        self.plot_init()

    def main_loop(self):
        """
        """
        #INTERACTIVE CONTROL FLAGS
        self.pause = True and not(self.profile)
        self.plot_update = False
        self.first_it = True
        self.prev_id = self.frame_id -1
        self.plot_cont = False
        self.prev_thresh_pnt = 100
        self.process_frame =  False
        
        frame = self.video_capture.read()
        exit_main_loop = False
        
        
        while not(exit_main_loop):
#            print(self.frame_id)
            if self.video_playback: #vs realtime capture
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
            
            
            if self.prev_id != self.frame_id or self.process_frame:
                retval, frame = self.video_capture.read()
                
                
                
                #APPLY METHOD
                for method in self.methods:
#                    print('apply method; {}'.format(method.name_id))
                    
                    if not(self.profile):
                        self.get_roi_coords(method.method_type)
                        
                    roi = frame[method.roi_y1:method.roi_y2,method.roi_x1:method.roi_x2]
                    roi_send = roi.copy()
                    method.apply(roi_send,self.frame_id)
                    
                self.prev_id = self.frame_id    #prev id processed
                self.process_frame = False
                
                
            #draw    
            if not(self.profile):
                self.plot_data()
                self.draw_frame(frame)
            
            exit_main_loop = self.playback_control()
                
        #------ end of main loop ---
        self.exit_test()

    def draw_frame(self,frame):
        """
        draw information on frame
        """
        #POST FRAME 
        #ROIs
        for roi in self.draw_roi_rects.values():
            colour,xy1xy2 = roi
            cv2.rectangle(frame, (xy1xy2[0],xy1xy2[1]), (xy1xy2[2],xy1xy2[3]), colour, 2)
        
#        #mask circle roi
#        for circles in self.draw_roi_circles.values():
#            colour,cxy,radius = circles
#            cv2.circle(frame, cxy, radius, colour, 1)
        
         #cx and cx        
#        cv2.line(frame,(self.gear_centre[0],0),(self.gear_centre[0],frame.shape[0]),(255,255,0))
#        cv2.line(frame,(0,self.gear_centre[1]),(frame.shape[1],self.gear_centre[1]),(255,255,0))

        #vline left
        cv2.line(frame,(self.controls['vline_left'],0),(self.controls['vline_left'],frame.shape[0]),(255,255,0))
        cv2.line(frame,(self.controls['vline_right'],0),(self.controls['vline_right'],frame.shape[0]),(255,255,0))
        
        #show
        cv2.imshow('source',frame)
            

    def plot_init(self):
        """            
        """
        if not(self.profile):
            plt.ion()
            plt.show()
            fig, all_axes = plt.subplots(1,4)
            self.all_data_ax = all_axes[0]
            self.all_pxl_ax = all_axes[1]
            self.all_speed_ax = all_axes[2]
            self.all_width_ax = all_axes[3]

            for method in self.methods:
                if method.plot_on:
                    fig, method.plot_axes = plt.subplots(3,2)
                    method.plot_init()
                    

            #polar plot
            #fig, P_axes = plt.subplots(1,1,projection='polar')
#            f_p = plt.figure()
#            DM_polar_ax = plt.subplot(111, projection='polar')

    def plot_data(self):
        """
        """        
        if (self.plot_update or self.plot_cont):
            self.plot_update = False
            
            #plot all data
            self.all_data_ax.cla()
            self.all_pxl_ax.cla()
            self.all_speed_ax.cla()
            self.all_width_ax.cla()
            
            all_calc_speed = []
            for index,method in enumerate(self.methods):
                self.all_data_ax.plot(method.time_step,label = method.name_id)
                all_calc_speed.append(method.time_step)
                if method.method_name == 'FWM': #width
                    self.all_width_ax.plot(method.width_history,'o-')
                else:
                    self.all_pxl_ax.plot(method.speed_step,label = method.name_id)
                    
                    self.all_speed_ax.plot(method.speed_mms,label = method.name_id)
#                    self.all_speed_ax.plot(np.array(method.beta_speed)*method.speed_conversion,label = 'beta')
#                    self.all_speed_ax.plot(np.array(method.ceta_speed)*method.speed_conversion,label = 'ceta')
#                    self.all_speed_ax.plot(np.array(method.deta_speed)*method.speed_conversion,label = 'deta')
#                    
                    
                    
            
            all_calc_speed = np.sum(all_calc_speed,axis = 0)
            self.all_data_ax.plot(all_calc_speed,label = 'Total')
            
            #anotate
            self.all_data_ax.legend()
            self.all_data_ax.set_title('Time Step (s)')
            
            self.all_pxl_ax.legend()
            self.all_pxl_ax.set_title('Speed (pxl/frame)')
            
            self.all_speed_ax.legend()
            self.all_speed_ax.set_title('Speed (mm/s)')
            
#            self.all_width_ax.legend()
            self.all_width_ax.set_title('Width (pxl)')
            
               

    def playback_control(self):
        """
        Poll keyboard for user input
        """
        exit_flag = False
        #CONTROL
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
           exit_flag = True
       
        elif ch == ord('u'): #update plot
           self.plot_update = True
           self.pause = True
           
        elif ch == ord('p'): #pause/play
           self.pause = not(self.pause)
           self.plot_cont = False
           print('play/pause {}'.format(self.pause))
           
        elif ch == ord('n'): #step next frame
           self.pause = True
           self.frame_id = self.frame_id + 1 if self.frame_id + 1 < self.num_frames else self.frame_id
        
        elif ch == ord('b'): #step prev frame
           self.pause = True
           self.frame_id = self.frame_id - 1 if self.frame_id - 1 >= 0  else self.frame_id
        
        elif ch == ord('x'): #step next frame and update
           self.pause = True
           self.plot_update = True
           self.frame_id = self.frame_id + 1 if self.frame_id + 1 < self.num_frames else self.frame_id
        
        elif ch == ord('y'): #step prev frame and update
           self.pause = True
           self.plot_update = True
           self.frame_id = self.frame_id - 1 if self.frame_id - 1 >= 0  else self.frame_id
        
        elif ch == ord('o'): #cont. mode and plot
           self.plot_cont = True
           self.pause = False
           
        elif ch == ord('s'): #dump data
           self.plot_cont = False
           self.pause = True
           for method in self.methods:
               np.savetxt(method.name_id + '.txt',method.speed_step)
               print('saved {}, -speed_step- '.format(method.name_id))

           
           
        elif ch != 255:
           print('unkown command/key {}'.format(ch))
           print('u,p,n,b,x,y,o')
        
        if not(self.pause) and self.video_playback:
            if self.frame_id + 1 >= self.num_frames:
                self.pause = True
                self.plot_cont = False
                self.plot_update = True
                print('all frames')
                
                if self.profile:
                    exit_flag = True
    #            np.savetxt('test_1.txt',np.array(LK.speed_mms))
    #            print('saved')
            else:
                self.frame_id += 1 
            
        return exit_flag
                    

    def exit_test(self):
        """
        """
        #END MAIN LOOP - EXIT
        self.video_capture.release()    
        cv2.destroyAllWindows()
    

    def init_video(self):
        """
        load video and video configs, if available
        """
        self.video_capture = cv2.VideoCapture(self.video_fullname)
        self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = self.config.getint('camera','cam_fps')
        self.frame_id = self.config.getint('vision','frame_start')

        self.control_vars = []
        self.control_vars.append(('vline_right',True))
        self.control_vars.append(('vline_left',True))
        self.control_vars.append(('w_y2',False))
        self.control_vars.append(('w_vline_left_border',False))
        self.control_vars.append(('w_vline_right_border',False))
        self.control_vars.append(('fil_roi_x',True))
        self.control_vars.append(('fil_roi_y',True))
        self.control_vars.append(('fil_roi_size',False))
        self.control_vars.append(('gear_roi_x',True))
        self.control_vars.append(('gear_roi_y',True))
        self.control_vars.append(('gear_roi_size',False))
        

        self.config_dict = make_config_dict(self.config)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
        


    def init_windows(self,frame = None,**kwargs):
        """
        """
        
        cv2.namedWindow('source')
        
        self.controls = {}
        for variable,show_trackbar in self.control_vars:
            init_val = self.config_dict['vision'][variable]
            
            if show_trackbar:
                cv2.createTrackbar(variable,'source',init_val,640,self.on_trackbar) 
                
            self.controls[variable] = init_val
       
        
       
        
        cv2.moveWindow('source',0,0)
    #    cv2.moveWindow('ROI',640,0)
    #    cv2.moveWindow('Thresh',640*2,0)
        
    def get_roi_coords(self,roi_type = 'gear'):
        """

        
        """
        
        for variable,read_trackbar  in self.control_vars:
            if read_trackbar:
                self.controls[variable] = cv2.getTrackbarPos(variable,'source')
       
        method_kwargs = {}
        if roi_type == 'width':
            #PRE GEAR AREA- estimate filament width
            x_start = self.controls['vline_left'] - self.controls['w_vline_left_border']#20
            y_start = self.controls['w_y2'] - 40
            x_end = self.controls['vline_right'] + self.controls['w_vline_right_border']#15
            roi_pnts = (x_start,y_start, x_end, self.controls['w_y2'])
        
            width_default = self.controls['vline_right'] - self.controls['vline_left']
            method_kwargs['width_default'] = width_default
            method_kwargs['w_vline_left_border'] = self.controls['w_vline_left_border']
            method_kwargs['w_vline_right_border'] = self.controls['w_vline_right_border']
            
            self.draw_roi_rects['width'] = [(75,125,225),roi_pnts]
            
        elif roi_type == 'gearSDM':
            #GEAR
            x_start = self.controls['gear_roi_x'] #left top pnt
            y_start = self.controls['gear_roi_y']  #left top pnt
            x_end = x_start + self.controls['gear_roi_size']
            y_end = y_start + self.controls['gear_roi_size']*2
            roi_pnts = (x_start,y_start,x_end,y_end)

            self.draw_roi_rects['gearSDM'] = [(125,75,225),roi_pnts]
        
        elif roi_type == 'filSDM':
            #fill
            x_start = self.controls['fil_roi_x'] #left top pnt
            y_start = self.controls['fil_roi_y']  #left top pnt
            x_end = x_start + self.controls['fil_roi_size']*1
            y_end = y_start + self.controls['fil_roi_size']*2
            roi_pnts = (x_start,y_start,x_end,y_end)
           
            self.draw_roi_rects['filSDM'] = [(125,125,225),roi_pnts]
        
        return roi_pnts,method_kwargs

    def on_trackbar(self,new_val):
        """
        """
        self.process_frame = True
#==============================================================================
#==============================================================================
def reject_outliers(data, m = 2.,return_option = 0,default_value = 0):
    """
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
    
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    indices = np.array(s < m,dtype = int)


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
if __name__ == '__main__':
    """
    method type, general kwargs:
        method_name : LK, DM
        
        plot_on: False
        draw_on: False
        
        w_timestep: False
    
       
    """
    import configparser
    print('main')
    #------------------------------------------------------------------------
    do_profiling = False
    
    #------------------------------------------------------------------------
    #DATA to load
#    data_name = 'data\\Test_2017_02_17__08_45_46' #15 fps #gear speed estimation fails
#    data_name = 'data\\Test_2017_04_11__14_27_24' #30 fps
#    data_name = 'data\\N5_Mwhite285_2016_05_24__11_39_22_T2300_L500_S20_0' #15 fps

#    data_name = 'data\\SQR_H20_S20_F25_T230_2016_11_23__09_29_48' #15 fps
#    data_name = 'data\\SQR_H20_S40_F50_T230_2016_11_23__09_38_07' #15 fps
#    data_name = 'data\\SQR_H20_S60_F75_T230_2016_11_23__09_47_17' #15 fps
    data_name = 'data\\SQR_H20_S80_F100_T230_2016_11_23__09_59_59' #15 fps
    
    video_fullname = data_name +'.avi'
    config_filename = data_name +'.cfg'
    #get config
    config = configparser.RawConfigParser()
    config.optionxform = str
    config.read(config_filename)
        


        

        
    #------------------------------------------------------------------------
    #METHODS to appply    
    methods = []
    w_timestep = not(do_profiling)
    
    FWM_11 = {'method_name': 'FWM',
            'w_timestep': w_timestep,
            'id': 'sub',
            'plot_on':False,
            'draw_on':False}
    
    SMD_g1 = {'method_name': 'gearSDM',
            'w_timestep': w_timestep,
            'id': 'x',
            'plot_on':False,
            'draw_on':False}
    
    SMD_f1 = {'method_name': 'filSDM',
            'w_timestep': w_timestep,
            'id': 'x',
            'exp_flow_dir': '+y',
            'plot_on':False,
            'draw_on':False}
            

    methods.append(FWM_11)
    methods.append(SMD_g1)
    methods.append(SMD_f1)

    
    #------------------------------------------------------------------------
    test_roi = TestROI(video_fullname,config,methods,profile = do_profiling)
    
    test_roi.main_loop()
