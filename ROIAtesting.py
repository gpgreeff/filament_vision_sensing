# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 13:37:47 2016

@author: Greeff



"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt as math_sqrt

from QtLoadSaveObjects import make_config_dict
np.seterr(invalid='raise')


#==============================================================================
class SmallAreaDenseMethod():
    def __init__(self,frame,roi_pnts,**kwargs):
        '''
        Apply Dense Optical Flow Method to the a small area ROI
        
        frame = roi area in main frame
        roi_pnts = (x1,y1,x2,y2),   coords relative to main frame
        gear_centre = (cx,xy),      coords relative to main frame
        
        roi size 40x40
        '''
        #info
        self.name_id = kwargs.get('name_id','DM_0')
        
        #debug control
        self.plot_on = kwargs.get('plot_on',False) # plots data
        self.draw_on = kwargs.get('draw_on',False) #draws addionational screens
        self.collect_on = kwargs.get('collect_on',False) #collects data in list
        self.collect_on = self.collect_on or self.plot_on or self.draw_on
        
        #plotting placeholder, for debuging/testing
        self.plot_axes = None
        
        #detection parameters
        self.min_flow_speed = kwargs.get('min_flow_speed',2.0) #pxl/frame
                                    
        #START INIT
        #roi region,, size parameters and disk_mask - recalculate on change of ROI/params etc.
        self.update_roi(roi_pnts,**kwargs)

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
#        self.angle_x_pnt = None
#        self.angle_y_pnt = None
#        self.angle_pnt = None
#        self.angle_flow = None
#        self.angle_centre = None
#        self.angle_indices = None

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
#        print('downscale_factor: {}'.format(self.downscale_factor))
        
#------------------------------------------------------------------------------
    def update_roi(self,roi_pnts,**kwargs):
        '''
        Update roi parameters and dependent vars
        '''
        self.roi_x1,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        self.roi_pnt1 = (self.roi_x1,self.roi_y1)
        self.roi_pnt2 = (self.roi_x2,self.roi_y2)
        

        
        #speed conversion
        self.fps = kwargs.get('cam_fps',15)
        self.cal_factor = kwargs.get('cal_factor',0.0212) 
        self.speed_conversion = self.cal_factor*self.fps
        
#------------------------------------------------------------------------------
    def pre_process(self,frame):
        '''
        Preprocess raw frame, apply gear mask
        '''
        self.frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#------------------------------------------------------------------------------        
    def apply_w_timestep(self,frame,*args,**kwargs):
        '''
        Estimate speed and measure timestep for whole process
        '''   
        e1 = cv2.getTickCount()
        self.apply_wo_timestep(frame)
        e2 = cv2.getTickCount()
        self.time_step_current = (e2 - e1)/ cv2.getTickFrequency()
        if self.collect_on:
            self.time_step.append(self.time_step_current)
#------------------------------------------------------------------------------        
    def apply_wo_timestep(self,frame,*args,**kwargs):
        '''
        Estimate speed, without measuring the time step
        
        cv2.calcOpticalFlowFarneback(prev, next, flow, 
                                     pyr_scale, levels, winsize, 
                                     iterations, poly_n, poly_sigma, 
                                     flags) → flow
        Parameters:	
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

        '''
        self.pre_process(frame)
#        self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,None,
#                                                 0.5,3,15,
##                                                 3,5,1.2,0)
#        if self.use_prev_flow and self.flow is not None:
#            self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,self.flow,
#                                                     pyr_scale = 0.5,levels = 3,winsize = 11,iterations = 3,
#                                                     poly_n = 7,poly_sigma = 1.5, flags = cv2.OPTFLOW_USE_INITIAL_FLOW)
#        else:
    
        self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,self.flow,
                                                 pyr_scale = 0.5,levels = 3,winsize = 11,iterations = 3,
                                                 poly_n = 7,poly_sigma = 1.5, flags = 0)
        
        self.prev_gray = self.frame_gray 
        self.est_speed()
#------------------------------------------------------------------------------
    def est_speed(self):
        '''        
        estimate flow speed
        '''
        
        fy,fx = self.flow.T
        
        
        try:
            self.v_sqr = fx*fx + fy*fy 
        except FloatingPointError:
            #Underflow: result so close to zero that some precision was lost.
            print('FloatingPointError, underflow encountered in multiply')
#            pass
        
        v_sqrt =  np.sqrt(self.v_sqr)
        
        #remove small flow speed values
        v0 = v_sqrt >= self.min_flow_speed
        
        if np.any(v0): 
            #mean of all selected velocities
            v_ave = np.mean(v_sqrt[v0])
            
        else:
            v_ave = 0

            
        self.speed_step_current  = v_ave
        
        #testing
        if self.collect_on:
            self.speed_step.append(v_ave)
            self.speed_mms.append(v_ave*self.speed_conversion)
            if self.draw_on:
                # plot the flow vectors
                cv2.imshow("Optical flow",self.draw_flow(v0))
            self.num_rejected.append(np.size(v0) - np.count_nonzero(v0))
            self.num_total.append(np.size(v0))
            if self.plot_on:
                self.plot_data(v0,v_sqrt)
               

#------------------------------------------------------------------------------        
    def draw_flow(self,v0,resize = 4):
        """ 
        Plot optical flow at sample points
        spaced step pixels apart. 
        """

        vis = cv2.cvtColor(self.frame_gray,cv2.COLOR_GRAY2BGR)
#        vis[x_indices,y_indices] = (0,0,255)

        vis = cv2.resize(vis,(self.roi_new_w*resize,self.roi_new_h*resize))
        
        #draw all lines
        fy,fx = self.flow[self.x_indices,self.y_indices].T
        lines = np.vstack([self.y_indices*resize,self.x_indices*resize,
                           (self.y_indices+fy)*resize,(self.x_indices+fx)*resize]).T.reshape(-1,2,2)
        lines = np.int32(lines)
        for (x1,y1),(x2,y2) in lines:
            cv2.line(vis,(x1,y1),(x2,y2),(255,255,0),1)
        
        #draw filtered lines
        x_indices = self.x_indices[v0]
        y_indices = self.y_indices[v0]
        fy = fy[v0]
        fx = fx[v0]
        
        linesF = np.vstack([y_indices*resize,x_indices*resize,
                            (y_indices+fy)*resize,(x_indices+fx)*resize]).T.reshape(-1,2,2)
        linesF = np.int32(linesF)
       
        for (x1,y1),(x2,y2) in linesF:
            cv2.line(vis,(x1,y1),(self.cx_local*resize,self.cy_local*resize),(255,0,255),1)
            cv2.line(vis,(x1,y1),(x2,y2),(0,255,255),1)
#            cv2.circle(vis,(x1,y1),1,(0,255,0))
            cv2.circle(vis,(x2,y2),1,(0,0,255))
            
        #cx and cy        
        cv2.line(vis,(self.cx_local*resize,0),(self.cx_local*resize,vis.shape[0]),(255,255,0))
        cv2.line(vis,(0,self.cy_local*resize),(vis.shape[1],self.cy_local*resize),(255,255,0))
        
        return vis
#------------------------------------------------------------------------------
    def plot_init(self):
        '''
        '''
        self.ax_pnts_acc = self.plot_axes[0,0]
        self.ax_flow_vector_x = self.plot_axes[1,0]
        self.ax_flow_vector_y = self.plot_axes[1,1]
        self.ax_flow_vector_r = self.plot_axes[0,1]
        self.ax_angle = self.plot_axes[2,0]
        self.ax_angle_rej = self.plot_axes[2,1]
        
        self.ax_flow_w = self.ax_flow_vector_r.twinx()
#------------------------------------------------------------------------------
    def plot_data(self,v0,v_sqrt):
        '''
        '''
        self.ax_pnts_acc.cla()
        self.ax_pnts_acc.plot(self.num_total,'o-',label = 'num_total')
        self.ax_pnts_acc.plot(self.num_rejected,'o-',label = 'num_rejected')
        self.ax_pnts_acc.legend()
        
        fx,fy = self.flow[self.x_indices,self.y_indices].T

#        v = np.sqrt(fx*fx + fy*fy)
    
#        fy_selected = fy[v0]
#        fx_selected = fx[v0]
        
    #        v_sqrt = np.sqrt(self.v_sqr)
#        w_rot_speed = v*self.indice_radii
        
        
#        self.ax_flow_vector_r.plot(v,label = 'v')
        v_sel = v_sqrt[v0]
        v_sel.sort()
        v_sel_len = v_sel.shape[0]
        if v_sel_len > 10:
            v_sel_half = int(v_sel_len /2)  #half
            v_sel_p10  = int(v_sel_len *0.1) #ten percent
        
            upper_half = v_sel[v_sel_half:-v_sel_p10]
            try:
                mean_upqa = np.mean(upper_half)
            except FloatingPointError:
                mean_upqa = np.mean(v_sel)
        elif v_sel_len > 1:
            mean_upqa = np.mean(v_sel)
        else:
            mean_upqa = 0
        
        self.ax_flow_vector_r.cla()
        self.ax_flow_vector_r.plot(v_sel*self.speed_conversion,'+',label = 'v_sel sorted')
        self.ax_flow_vector_r.axhline(self.min_flow_speed*self.speed_conversion,c='r',ls = '--')
        self.ax_flow_vector_r.axhline(self.speed_step_current*self.speed_conversion,c='k',ls = '--')
        self.ax_flow_vector_r.axhline(mean_upqa*self.speed_conversion,c='b',ls = '--')
        self.ax_flow_vector_r.legend()
        
        self.ax_flow_vector_r.axis(ymin = 0,ymax = 4)
        self.ax_flow_w.cla()
#        self.ax_flow_w.plot(w_rot_speed,'g',label = 'w_rot_speed')
        
#        self.ax_flow_vector_x.cla()
#        self.ax_flow_vector_x.plot(fx,'o',label = 'fx')
#        self.ax_flow_vector_x.plot(fx_selected,label = 'fx_selected')
#        self.ax_flow_vector_x.legend()
#        
#        self.ax_flow_vector_y.cla()
#        self.ax_flow_vector_y.plot(fy,'o',label = 'fy')
#        self.ax_flow_vector_y.plot(fy_selected,label = 'fy_selected')
#        self.ax_flow_vector_y.legend()
#        
##        CW = fx*self.y_CW > self.x_CW*fy #CW rotation
#        
#        self.ax_angle.cla()
##        self.ax_angle.plot(np.rad2deg(np.arctan(fy/fx)),'o',label='all')    
#        self.ax_angle.plot(np.rad2deg(np.arctan(fy_selected/fx_selected)),'o',label='fy/fx')   
#        self.ax_angle.plot(np.rad2deg(-np.arctan(self.y_indices[self.x_indices!=0]/self.x_indices[self.x_indices!=0])),'o',label='std')
#        self.ax_angle.legend()
#        
#        self.ax_angle_rej.cla()
#        self.ax_angle_rej.plot(self.indice_radii)

#        self.ax_angle_rej.legend()
#        
#==============================================================================
class GearDenseMethod():
    def __init__(self,frame,roi_pnts,**kwargs):
        '''
        Apply Dense Optical Flow Method to the a gear ROI
        
        frame = roi area in main frame
        roi_pnts = (x1,y1,x2,y2),   coords relative to main frame
        gear_centre = (cx,xy),      coords relative to main frame
        
        kwargs:
            fps = 15
            downscale_factor = 2
            plot_on = True
            
            
        detection kwargs:
            min_flow_speed:
                 reject flow speed values less than this
                 flow speed at min citeria = sqrt(fx**2 + fy**2)
         '''
        #info
        self.name_id = kwargs.get('name_id','DM_0')
        
        #debug control
        self.plot_on = kwargs.get('plot_on',False) # plots data
        self.draw_on = kwargs.get('draw_on',False) #draws addionational screens
        self.collect_on = kwargs.get('collect_on',False) #collects data in list
        self.collect_on = self.collect_on or self.plot_on or self.draw_on
        
        #plotting placeholder, for debuging/testing
        self.plot_axes = None
        
        #detection parameters
        self.min_flow_speed = kwargs.get('min_flow_speed',2.0) #pxl/frame
        self.cw_filter = kwargs.get('cw_filter',True)

#        self.use_prev_flow = kwargs.get('use_prev_flow',False) # quick check did not result in speed up
        
#        self.angle_mod = 1.*np.pi #mod, 180 degrees
#        self.min_angle_range = np.deg2rad(15) # +-15 degrees from 90
#        self.min_angle = np.deg2rad(90) - self.min_angle_range 
#        self.max_angle = np.deg2rad(90) + self.min_angle_range 
                                    
        #START INIT
        #roi region,, size parameters and disk_mask - recalculate on change of ROI/params etc.
        self.update_roi(roi_pnts,**kwargs)

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
#        self.angle_x_pnt = None
#        self.angle_y_pnt = None
#        self.angle_pnt = None
#        self.angle_flow = None
#        self.angle_centre = None
#        self.angle_indices = None

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
#        print('downscale_factor: {}'.format(self.downscale_factor))
        
#------------------------------------------------------------------------------
    def update_roi(self,roi_pnts,**kwargs):
        '''
        Update roi parameters and dependent vars
        '''
        self.roi_x1,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        self.cx,self.cy  = kwargs.get('gear_centre',(0,0))

        self.roi_pnt1 = (self.roi_x1,self.roi_y1)
        self.roi_pnt2 = (self.roi_x2,self.roi_y2)
        
        self.cx_local = self.cx - self.roi_x1
        self.cy_local = self.cy - self.roi_y1
        
        #resize/shrink with downscale_factor
        self.downscale_factor = float(kwargs.get('downscale_factor',1))
        self.roi_new_h = int(round((self.roi_y2  - self.roi_y1)/self.downscale_factor))
        self.roi_new_w = int(round((self.roi_x2  - self.roi_x1)/self.downscale_factor))
        
        self.cx_local_dwn = self.cx_local/self.downscale_factor
        self.cy_local_dwn = self.cy_local/self.downscale_factor

        
        #roi disc mask setup
        self.gear_mask_generate(**kwargs)
        
        #speed conversion
        self.fps = kwargs.get('cam_fps',15)
        self.speed_conversion = self.cal_factor*self.fps
#------------------------------------------------------------------------------
    def gear_mask_generate(self,**kwargs):
        '''
        Make an disc ROI by masking
        
        cx and cy are relative to main frame shape
        
        kwargs:
            main_frame = (640,480)
            steps = 8 # sample every i'th point in gear_mask_disk_inv in calculated flow 
                        and use these samples to estimate the speed
            gear_radius_mm       mask radius, +cr_add -cr_min
            cal_factor           convert radius to pxl and speed_step to speed_mms
            od_resize_factor     ajdustment factor
            cr_add
            cr_min
        '''
        main_frame = kwargs.get('main_frame',(640,480))
        
        #init disc (dougnut) mask size, based on an the gear radius
        gear_radius_mm = kwargs.get('gear_radius_mm',4.0)
        self.cal_factor = kwargs.get('cal_factor',0.0212) 
        self.od_resize_factor = kwargs.get('od_resize_factor',1.07) 
        
        self.cr_od = int(round(self.od_resize_factor * gear_radius_mm / self.cal_factor)) 
        
        cr_add = kwargs.get('cr_add',20)
        cr_min = kwargs.get('cr_min',10)
        
#        cr_space = cr_add + cr_min
        self.cr_max = self.cr_od + cr_add
        self.cr_min = self.cr_od - cr_min 
#        print('cr_min: {}; cr_add: {}'.format(cr_min,cr_add))
#        print('cr_min: {}; cr_max: {}'.format(self.cr_min,self.cr_max))
        
        #draw disk - on main frame copy
        self.gear_mask = np.zeros((main_frame[1], main_frame[0]), np.uint8)
        cv2.circle(self.gear_mask,(self.cx,self.cy),self.cr_max,(255),-1)
        cv2.circle(self.gear_mask,(self.cx,self.cy),self.cr_min,0,-1)
        
        #select gear roi in main frame and resize
        self.gear_mask = self.gear_mask[self.roi_y1:self.roi_y2,self.roi_x1:self.roi_x2] #

        #from here on everything is donwscaled - shrunked
        if self.downscale_factor != 1:
            self.gear_mask = cv2.resize(self.gear_mask,(self.roi_new_w,self.roi_new_h))
            self.min_flow_speed = self.min_flow_speed/self.downscale_factor

        self.gear_mask_indices = self.gear_mask == 0
        self.gear_mask_inv_indices = self.gear_mask != 0
        
        #x and y coords used to sample flow
        self.x_indices,self.y_indices = self.gear_mask_inv_indices.nonzero()
        
        #radius at each evaluation point
        dx = self.x_indices - self.cx_local_dwn
        dy = self.y_indices - self.cy_local_dwn
        all_radii =  np.sqrt(dx*dx + dy*dy)
        xyr = np.vstack((self.x_indices,self.y_indices,all_radii))
        xyr = xyr.T
        xyr = xyr[xyr[:,2].argsort()]
        self.x_indices = xyr[:,0].astype(np.int)
        self.y_indices = xyr[:,1].astype(np.int)
        self.indice_radii = xyr[:,2]
        
        #select only every i'th step
        steps = kwargs.get('steps',32)
        self.x_indices = self.x_indices[0::steps]
        self.y_indices = self.y_indices[0::steps]
        self.indice_radii = self.indice_radii[0::steps]
        
        #vector for CW check
        self.x_CW = self.x_indices + self.cx_local_dwn
        self.y_CW = self.y_indices + self.cy_local_dwn
        
        #array init        
        self.v_sqr = np.zeros_like(self.x_indices,dtype = np.float32) #prefer array over list
        
        
        
        #testing
        if self.draw_on: 
            cv2.imshow('gear_mask_DQM',self.gear_mask)
            
            eval_pnts = self.gear_mask.copy()
            eval_pnts = cv2.cvtColor(eval_pnts,cv2.COLOR_GRAY2BGR)
            eval_pnts[self.x_indices,self.y_indices,:] = [0,0,255]
            eval_pnts = cv2.resize(eval_pnts,(eval_pnts.shape[1]*4,eval_pnts.shape[0]*4))
            cv2.imshow('gear_indices_eval_pnts',eval_pnts)
        
#------------------------------------------------------------------------------
    def pre_process(self,frame):
        '''
        Preprocess raw frame, apply gear mask
        '''
        self.frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.downscale_factor != 1:
            self.frame_gray = cv2.resize(self.frame_gray,(self.roi_new_w,self.roi_new_h))
        self.frame_gray[self.gear_mask_indices]  = 0


#------------------------------------------------------------------------------        
    def apply_w_timestep(self,frame,*args,**kwargs):
        '''
        Estimate speed and measure timestep for whole process
        '''   
        e1 = cv2.getTickCount()
        self.apply_wo_timestep(frame)
        e2 = cv2.getTickCount()
        self.time_step_current = (e2 - e1)/ cv2.getTickFrequency()
        if self.collect_on:
            self.time_step.append(self.time_step_current)
#------------------------------------------------------------------------------        
    def apply_wo_timestep(self,frame,*args,**kwargs):
        '''
        Estimate speed, without measuring the time step
        
        cv2.calcOpticalFlowFarneback(prev, next, flow, 
                                     pyr_scale, levels, winsize, 
                                     iterations, poly_n, poly_sigma, 
                                     flags) → flow
        Parameters:	
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

        '''
        self.pre_process(frame)
#        self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,None,
#                                                 0.5,3,15,
##                                                 3,5,1.2,0)
#        if self.use_prev_flow and self.flow is not None:
#            self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,self.flow,
#                                                     pyr_scale = 0.5,levels = 3,winsize = 11,iterations = 3,
#                                                     poly_n = 7,poly_sigma = 1.5, flags = cv2.OPTFLOW_USE_INITIAL_FLOW)
#        else:
    
        self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.frame_gray,self.flow,
                                                 pyr_scale = 0.5,levels = 3,winsize = 11,iterations = 3,
                                                 poly_n = 7,poly_sigma = 1.5, flags = 0)
        
        self.prev_gray = self.frame_gray 
        self.est_speed()
#------------------------------------------------------------------------------
    def est_speed(self):
        '''        
        estimate flow speed
        '''
        
        fy_dwnscaled,fx_dwnscaled = self.flow[self.x_indices,self.y_indices].T
        
        
        try:
            self.v_sqr[:] = fx_dwnscaled*fx_dwnscaled + fy_dwnscaled*fy_dwnscaled 
        except FloatingPointError:
            #Underflow: result so close to zero that some precision was lost.
#            print('FloatingPointError, underflow encountered in multiply')
            pass
        
        
        v_sqrt =  np.sqrt(self.downscale_factor*self.v_sqr)
        
        #remove small flow speed values
        v0 = v_sqrt >= self.min_flow_speed
        
#        w_rot_speed = v_sqrt*self.indice_radii #should be component tangent to centre

        #flow vector angle considerations possible
        #flow vector angle should be approx. orthogonal to radius line
        if self.cw_filter:
            CW = fx_dwnscaled*self.y_CW > self.x_CW*fy_dwnscaled #CW rotation
            #select indices - bitwise_and
            v0 = v0 & CW #& ang_indices
        
        if np.any(v0): 
            #mean of all selected velocities
#            v_ave = np.mean(v_sqrt[v0])
            
            
            #upper half selection
            #sort the velocities, small to great
            #then use mean or upper half (upto upper 10%)
            #for mean velocity.
           
            v_sel = v_sqrt[v0]
            v_sel.sort()
            self.v_sel_len = v_sel.shape[0]
            if self.v_sel_len > 10:
                self.v_sel_half = int(self.v_sel_len /2)  #half
                self.v_sel_p10  = int(self.v_sel_len *0.1) #ten percent
            
                upper_half = v_sel[self.v_sel_half:-self.v_sel_p10]
                try:
                    v_ave = np.mean(upper_half)
                except FloatingPointError:
                    v_ave = np.mean(v_sel)
            else:
                v_ave = np.mean(v_sel)

                
#                
            
            #rotation speed method - to dependant on cx,cy and evaluation radius
#            w_mean = np.mean(w_rot_speed[v0])
#            v_ave = w_mean/self.cr_od
            
            v_ave *= self.downscale_factor #upscale
         
        else:
            v_ave = 0
#            print('v_ave: {}, v:{},v0:{} - {}'.format(v_ave,v.shape,v0.shape,v0.any()))
            
        self.speed_step_current  = v_ave
        
        #testing
        if self.collect_on:
            self.speed_step.append(v_ave)
            self.speed_mms.append(v_ave*self.speed_conversion)
            if self.draw_on:
                # plot the flow vectors
                cv2.imshow("Optical flow",self.draw_flow(v0))
            self.num_rejected.append(np.size(v0) - np.count_nonzero(v0))
            self.num_total.append(np.size(v0))
            if self.plot_on:
                self.plot_data(v0,v_sqrt)
               

#------------------------------------------------------------------------------        
    def draw_flow(self,v0,resize = 4):
        """ 
        Plot optical flow at sample points
        spaced step pixels apart. 
        """

        vis = cv2.cvtColor(self.frame_gray,cv2.COLOR_GRAY2BGR)
#        vis[x_indices,y_indices] = (0,0,255)

        vis = cv2.resize(vis,(self.roi_new_w*resize,self.roi_new_h*resize))
        
        #draw all lines
        fy,fx = self.flow[self.x_indices,self.y_indices].T
        lines = np.vstack([self.y_indices*resize,self.x_indices*resize,
                           (self.y_indices+fy)*resize,(self.x_indices+fx)*resize]).T.reshape(-1,2,2)
        lines = np.int32(lines)
        for (x1,y1),(x2,y2) in lines:
            cv2.line(vis,(x1,y1),(x2,y2),(255,255,0),1)
        
        #draw filtered lines
        x_indices = self.x_indices[v0]
        y_indices = self.y_indices[v0]
        fy = fy[v0]
        fx = fx[v0]
        
        linesF = np.vstack([y_indices*resize,x_indices*resize,
                            (y_indices+fy)*resize,(x_indices+fx)*resize]).T.reshape(-1,2,2)
        linesF = np.int32(linesF)
       
        for (x1,y1),(x2,y2) in linesF:
            cv2.line(vis,(x1,y1),(self.cx_local*resize,self.cy_local*resize),(255,0,255),1)
            cv2.line(vis,(x1,y1),(x2,y2),(0,255,255),1)
#            cv2.circle(vis,(x1,y1),1,(0,255,0))
            cv2.circle(vis,(x2,y2),1,(0,0,255))
            
        #cx and cy        
        cv2.line(vis,(self.cx_local*resize,0),(self.cx_local*resize,vis.shape[0]),(255,255,0))
        cv2.line(vis,(0,self.cy_local*resize),(vis.shape[1],self.cy_local*resize),(255,255,0))
        
        return vis
#------------------------------------------------------------------------------
    def plot_init(self):
        '''
        '''
        self.ax_pnts_acc = self.plot_axes[0,0]
        self.ax_flow_vector_x = self.plot_axes[1,0]
        self.ax_flow_vector_y = self.plot_axes[1,1]
        self.ax_flow_vector_r = self.plot_axes[0,1]
        self.ax_angle = self.plot_axes[2,0]
        self.ax_angle_rej = self.plot_axes[2,1]
        
        self.ax_flow_w = self.ax_flow_vector_r.twinx()
#------------------------------------------------------------------------------
    def plot_data(self,v0,v_sqrt):
        '''
        '''
        self.ax_pnts_acc.cla()
        self.ax_pnts_acc.plot(self.num_total,'o-',label = 'num_total')
        self.ax_pnts_acc.plot(self.num_rejected,'o-',label = 'num_rejected')
        self.ax_pnts_acc.legend()
        
        fx,fy = self.flow[self.x_indices,self.y_indices].T

#        v = np.sqrt(fx*fx + fy*fy)
    
#        fy_selected = fy[v0]
#        fx_selected = fx[v0]
        
    #        v_sqrt = np.sqrt(self.v_sqr)
#        w_rot_speed = v*self.indice_radii
        
        self.ax_flow_vector_r.cla()
#        self.ax_flow_vector_r.plot(v,label = 'v')
        v_sel = v_sqrt[v0]*self.downscale_factor
        v_sel.sort()
        v_sel_len = v_sel.shape[0]
        if v_sel_len > 10:
            v_sel_half = int(v_sel_len /2)  #half
            v_sel_p10  = int(v_sel_len *0.1) #ten percent
        
            upper_half = v_sel[v_sel_half:-v_sel_p10]
            try:
                mean_upqa = np.mean(upper_half)
            except FloatingPointError:
                mean_upqa = np.mean(v_sel)
        elif v_sel_len > 1:
            mean_upqa = np.mean(v_sel)
        else:
            mean_upqa = 0
            
        self.ax_flow_vector_r.plot(v_sel*self.speed_conversion,'+',label = 'v_sel sorted')
        self.ax_flow_vector_r.axhline(self.min_flow_speed*self.speed_conversion,c='r',ls = '--')
        self.ax_flow_vector_r.axhline(self.speed_step_current*self.speed_conversion,c='k',ls = '--')
        self.ax_flow_vector_r.axhline(mean_upqa*self.speed_conversion,c='b',ls = '--')
        self.ax_flow_vector_r.legend()
        
        self.ax_flow_vector_r.axis(ymin = 0,ymax = 4)
        self.ax_flow_w.cla()
#        self.ax_flow_w.plot(w_rot_speed,'g',label = 'w_rot_speed')
        
#        self.ax_flow_vector_x.cla()
#        self.ax_flow_vector_x.plot(fx,'o',label = 'fx')
#        self.ax_flow_vector_x.plot(fx_selected,label = 'fx_selected')
#        self.ax_flow_vector_x.legend()
#        
#        self.ax_flow_vector_y.cla()
#        self.ax_flow_vector_y.plot(fy,'o',label = 'fy')
#        self.ax_flow_vector_y.plot(fy_selected,label = 'fy_selected')
#        self.ax_flow_vector_y.legend()
#        
##        CW = fx*self.y_CW > self.x_CW*fy #CW rotation
#        
#        self.ax_angle.cla()
##        self.ax_angle.plot(np.rad2deg(np.arctan(fy/fx)),'o',label='all')    
#        self.ax_angle.plot(np.rad2deg(np.arctan(fy_selected/fx_selected)),'o',label='fy/fx')   
#        self.ax_angle.plot(np.rad2deg(-np.arctan(self.y_indices[self.x_indices!=0]/self.x_indices[self.x_indices!=0])),'o',label='std')
#        self.ax_angle.legend()
#        
#        self.ax_angle_rej.cla()
#        self.ax_angle_rej.plot(self.indice_radii)

#        self.ax_angle_rej.legend()
#        
        
#==============================================================================
class GearLKMethod():
    def __init__(self,frame,roi_pnts,**kwargs):
        '''
        Apply Lukase-Kanade Optical Flow Method to the a gear ROI
        '''
        #info
        self.name_id = kwargs.get('name_id','LK_0')
        
        #debug control
        self.plot_on = kwargs.get('plot_on',False) # plots data
        self.draw_on = kwargs.get('draw_on',False) #draws addionational screens
        self.collect_on = kwargs.get('collect_on',False) #collects data in list
        self.collect_on = self.collect_on or self.plot_on or self.draw_on
        
        self.plot_axes = None
        
#        self.gear_use_mask = kwargs.get('use_mask',True)
#        self.thresh_pnt = kwargs.get('thresh_pnt',125) #manual threshpoint value
        
        #internal data
        self.tracks = []
        self.speed_xy_per_track = []
        self.speed_w_per_track = [] #rotational speed
        
        self.prev_gray = None
        self.gray = None
        self.thresh = None
        
        self.frame_id = 0
        self.first_analysis = True
        
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
        
        #point selection
        self.gear_sp_min_h_sqrd = 2#self.gear_sp_min_R**2
        
        #set roi and disk mask
        self.update_roi(roi_pnts,**kwargs)
  
        
        #data collection: measurands
        self.speed_step_current  = 0
        self.time_step_current = 0
        
        #list collection with self.collect_on
        if self.collect_on:
            self.pnt_raddii = []
            self.speed_step = []
            self.speed_mms = []
            self.time_step = []
        
        #set apply method
        w_timestep = kwargs.get('w_timestep',False)
        if w_timestep:
            self.apply = self.apply_w_timestep
        else:
            self.apply = self.apply_wo_timestep

#------------------------------------------------------------------------------
    def update_roi(self,roi_pnts,**kwargs):
        '''
        Update roi parameters and dependent vars
        '''
        self.roi_x1,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        self.cx,self.cy  = kwargs.get('gear_centre',(0,0))

        self.roi_pnt1 = (self.roi_x1,self.roi_y1)
        self.roi_pnt2 = (self.roi_x2,self.roi_y2)
        
        self.cy_flipped = (self.roi_y2 - self.roi_y1) - self.cy
        self.cx_local = self.cx - self.roi_x1
        self.cy_local = self.cy - self.roi_y1
        
        #resize/shrink with downscale_factor
        self.downscale_factor = kwargs.get('downscale_factor',2)
        self.roi_new_h = int(round((self.roi_y2  - self.roi_y1)/self.downscale_factor))
        self.roi_new_w = int(round((self.roi_x2  - self.roi_x1)/self.downscale_factor))
        self.cx_local = int(round(self.cx_local/self.downscale_factor))
        self.cy_local = int(round(self.cy_local/self.downscale_factor))
        self.cy_flipped = int(round(self.cy_flipped/self.downscale_factor))
        
        #roi disc mask setup
        self.gear_mask_generate(**kwargs)
        
        #speed conversion
        self.fps = kwargs.get('cam_fps',15)
        self.speed_conversion = self.cal_factor*self.fps
#------------------------------------------------------------------------------
    def gear_mask_generate(self,**kwargs):
        '''
        Make an disc ROI by masking
        
        cx and cy are relative to main frame shape
        
        kwargs:
            main_frame = (640,480)

            gear_radius_mm       mask radius, +cr_add -cr_min
            cal_factor           convert radius to pxl and speed_step to speed_mms
            od_resize_factor     ajdustment factor
            cr_add
            cr_min
        '''
        main_frame = kwargs.get('main_frame',(640,480))

        
        #init disc (dougnut) mask size, based on an the gear radius
        gear_radius_mm = kwargs.get('gear_radius_mm',4.0)
        self.cal_factor = kwargs.get('cal_factor',0.0212) 
        self.od_resize_factor = kwargs.get('od_resize_factor',1.07) 
        

        self.cr_od = int(round(self.od_resize_factor * gear_radius_mm / self.cal_factor)) 
        
        cr_add = kwargs.get('cr_add',20)
        cr_min = kwargs.get('cr_min',10)
#        cr_space = cr_add + cr_min
        self.cr_max = self.cr_od + cr_add
        self.cr_min = self.cr_od - cr_min 
        
        #draw disk - on main frame copy
        self.gear_mask = np.zeros((main_frame[1], main_frame[0]), np.uint8)
        cv2.circle(self.gear_mask,(self.cx,self.cy),self.cr_max,(255),-1)
        cv2.circle(self.gear_mask,(self.cx,self.cy),self.cr_min,0,-1)
        
        #select gear roi in main frame and resize
        self.gear_mask = self.gear_mask[self.roi_y1:self.roi_y2,self.roi_x1:self.roi_x2] #
        self.gear_mask = cv2.resize(self.gear_mask,(self.roi_new_w,self.roi_new_h))

        self.gear_mask_indices = self.gear_mask == 0
        self.gear_mask_inv_indices = self.gear_mask != 0
        
        if self.draw_on:
            cv2.imshow('gear_mask_LK',self.gear_mask)

#------------------------------------------------------------------------------
    def apply_w_timestep(self,roi,frame_id):
        '''
        '''
        e1 = cv2.getTickCount()
        
        self.apply_wo_timestep(roi,frame_id)
        
        e2 = cv2.getTickCount()
        self.time_step_current = (e2 - e1)/ cv2.getTickFrequency()
        if self.collect_on:
            self.time_step.append(self.time_step_current)
            self.speed_step.append(self.speed_step_current)
            self.speed_mms.append(self.speed_step_current*self.speed_conversion)
            if self.plot_on:
                self.plot_data()
                self.pnt_raddii = []
#------------------------------------------------------------------------------
    def apply_wo_timestep(self,roi,frame_id):
        '''
        '''
        self.frame_id = frame_id
        
        self.gear_preprocess(roi)
        gear_speed =  self.flow_track_gear()
        gear_speed *= self.downscale_factor
        self.speed_step_current  = gear_speed
        
        self.first_analysis = False
        
#------------------------------------------------------------------------------
    def gear_preprocess(self,roi):
        '''
        '''
#        if self.draw_on:
#            cv2.imshow('roi_LKM',roi)
            
        self.roi_test = roi.copy()# cv2.GaussianBlur(roi, (3, 3),0)
        self.roi_test[:,:,1:] = 0 #remove blue/red, helps somehow with blue background
                     
        self.gray = cv2.cvtColor(self.roi_test,cv2.COLOR_BGR2GRAY)
        if self.downscale_factor != 1:
            self.gray = cv2.resize(self.gray,(self.roi_new_w,self.roi_new_h))
        
        self.gray = cv2.GaussianBlur(self.gray, (3, 3),0)
        
        #mask gray
#       ret_val,self.thresh = cv2.threshold(self.gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#       ret_val,self.thresh = cv2.threshold(self.gray,thresh_pnt,255,cv2.THRESH_BINARY_INV)
        self.thresh  = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 1)          
        self.thresh[self.gear_mask_indices]  = 0

        #threshold
        self.thresh = cv2.GaussianBlur(self.thresh, (3, 3),0)
        
#------------------------------------------------------------------------------
    def flow_track_gear(self):
        '''

        '''
        self.speed_xy_per_track = []
        #----------------------------------------------------------------------
        if self.tracks:
            self.track_points_update()
        #----------------------------------------------------------------------
        #update speed
        if len(self.speed_xy_per_track) > 1:
#            gear_speed = reject_outliers(np.array(self.speed_xy_per_track),
#                                         return_option = 1)
                                         
#            gear_speed = np.mean(self.speed_xy_per_track)
            
            
#            gear_w_mean = np.mean(self.speed_w_per_track)
#            gear_speed = gear_w_mean/self.cr_od
            
#            gear_speed = reject_outliers(np.array(self.speed_xy_per_track),
#                                         return_option = 1)
#            gear_speed = math_sqrt(gear_speed)

            #upper half for gear speed est
            self.speed_xy_per_track.sort()
            len_data = len(self.speed_xy_per_track)
            if len_data > 10:
                half = int(len_data/2)
                p10 = int(len_data*0.1)
                upper_half = self.speed_xy_per_track[half:-p10]
                try:
                    gear_speed = np.mean(upper_half)
                except FloatingPointError:
                    gear_speed = np.mean(self.speed_xy_per_track)

            else:
                gear_speed = np.mean(self.speed_xy_per_track)
                

            
        else:
            gear_speed = 0
        #----------------------------------------------------------------------
        #get new points
        self.get_new_track_points()

        return gear_speed
#------------------------------------------------------------------------------  
    def track_points_update(self):
        '''
        '''
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.thresh, p0, None, **self.lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(self.thresh, self.prev_gray, p1, None, **self.lk_params) #reverse
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
                self.gear_speed_append(x,y,prev_pnt) #this can be vectorized
    
            except IndexError:
                pass #first point in track
        #end for loop        
        self.tracks = new_tracks        
    #            if self.draw_on:
    #                cv2.circle(vis, (x, y), 2, (0, 0,255), -1)
        
    #        if self.draw_on:
    #            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
             #all points in a track
    #            for track in self.tracks:
    #                for x,y in track:
    #                    cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)
    
    #            cv2.imshow('track_pnts',vis)    
#------------------------------------------------------------------------------
    def get_new_track_points(self):
        '''
        '''
        if self.frame_id % self.detect_interval == 0 or self.first_analysis:
            try:
                mask = self.new_pnts_mask.copy()
            except AttributeError:    
                self.new_pnts_mask = np.zeros_like(self.thresh) + 255
                mask = self.new_pnts_mask.copy()

            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), self.track_pnt_mask_radius, 0, -1)
            p = cv2.goodFeaturesToTrack(self.thresh, mask = mask, **self.feature_params)
            
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])
#                    if self.draw_on:
#                         cv2.circle(self.roi_test, (x, y), 2, (0, 255, 0), -1)
        #update prev
        self.prev_gray = self.thresh.copy()
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
            x0 = prev_pnt[0] - self.cx
            y0 = prev_pnt[1] - self.cy_flipped
#            det = dx*y0 - x0*dy
            if dx*y0 < x0*dy: #CW ROTATION
#                self.speed_xy_per_track.append(h) #sqrt later 

                v = math_sqrt(h)    #sqrt now                      
                self.speed_xy_per_track.append(v) 
                
                pnt_radius_dx = x - self.cx_local
                pnt_radius_dy = y - self.cy_local
                pnt_radius = math_sqrt(pnt_radius_dx*pnt_radius_dx + pnt_radius_dy*pnt_radius_dy)
                self.pnt_raddii.append(pnt_radius)
#                w = v*pnt_radius
#                self.speed_w_per_track.append(w)

#------------------------------------------------------------------------------
    def plot_init(self):
        '''
        '''
        self.ax_xy_p_track = self.plot_axes[0,0]
        self.ax_01 = self.plot_axes[0,1]
        
        self.ax_10 = self.plot_axes[1,0]
        
        self.ax_11 = self.plot_axes[1,1]
#------------------------------------------------------------------------------
    def plot_data(self):
        '''
        '''
        self.ax_xy_p_track.cla()
       
        vr_track = np.vstack([self.speed_xy_per_track,self.pnt_raddii])
        vr_track = vr_track.T
        vr_track = vr_track[vr_track[:,0].argsort()]
        
        mean = np.mean(self.speed_xy_per_track)
#        
#        median = np.median(self.speed_xy_per_track)
        data = vr_track[:,0]
#        m = 2.
#        d = np.abs(data - np.median(data))
#        mdev = np.median(d)
#        s = d/mdev if mdev else 0.
#        indices = np.array(s < m,dtype = int)
#        new_set = data[s < m]
#        x  = indices.nonzero()[0]
        
        len_data = data.shape[0]
        half = int(len_data/2)
        p10 = int(len_data*0.1)
        upper_half = data[half:-p10]
        
        mean_upqa = np.mean(upper_half)
        
            
        self.ax_xy_p_track.plot(vr_track[:,0],'go-',label = 'v_track')
#        self.ax_xy_p_track.plot(x,new_set,'ro-',label = 'new_set')
        self.ax_xy_p_track.plot(np.arange(half,len_data-p10,1),upper_half,'ko-',label = 'new_set')
        
        self.ax_xy_p_track.axhline(math_sqrt(self.gear_sp_min_h_sqrd),c='r',ls = '--',label = 'min')
#        self.ax_xy_p_track.axhline(self.speed_step_current,c='k',ls = '--',label='current')
        self.ax_xy_p_track.axhline(mean,c='g',ls = '--',label='mean')
#        self.ax_xy_p_track.axhline(median,c='cyan',ls = '--',label='median')
        self.ax_xy_p_track.axhline(mean_upqa,c='b',ls = '--',label='mean_upqa')
        
        self.ax_xy_p_track.legend()   
        self.ax_xy_p_track.axis(ymin=0,ymax=10)
        
        self.ax_01.plot(vr_track[:,0]*self.speed_conversion,'o-',label = 'v_track')
        self.ax_01.axhline(2,c='r',ls = '--')
        self.ax_01.axhline(self.speed_step_current*self.speed_conversion,c='k',ls = '--')
        self.ax_01.axis(ymin=0,ymax=3)
        self.ax_01.legend()  
#
        self.ax_10.cla()
        self.ax_10.plot(vr_track[:,1],'o-',label = 'radii_track')
#        self.ax_10.legend() 
#        
        self.ax_11.cla()
        self.ax_11.plot(vr_track[:,1],vr_track[:,0],'o',label = 'v vs radii')
        self.ax_11.legend() 
        
#==============================================================================
#==============================================================================
class FilamentLKMethod():
    def __init__(self,frame,roi_pnts,**kwargs):
        '''
        Apply Lukase-Kanade Optical Flow Method to the a filament ROI, 
        to estimate filament speed (assumption - downwards)
        '''
        #info
        self.name_id = kwargs.get('name_id','fill_LK0')
        
        #debug/test control
        self.plot_on = kwargs.get('plot_on',False) # plots data
        self.draw_on = kwargs.get('draw_on',False) #draws addionational screens
        self.collect_on = kwargs.get('collect_on',False) #collects data in list
        self.collect_on = self.collect_on or self.plot_on or self.draw_on
        
        #configs
        self.configs = kwargs
        self.material = self.configs.get('mat_colour','white').lower()
        
        #preprocess
        self.thresh_inv = self.configs.get('thresh_inv',False) #delta ave method - invert min/max
        if self.thresh_inv:
            self.thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        else:
            self.thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
            
        if self.material == 'green':
            self.filament_sp_filter_blue = True
            self.filament_sp_use_right = False
            
        elif self.material == 'white':
            self.filament_sp_filter_blue = False
            self.filament_sp_use_right = False

        elif self.material == 'clear':
            self.filament_sp_use_right = True
            self.filament_sp_filter_blue = False
        
        elif self.material == 'red':
            self.filament_sp_use_right = True
            self.filament_sp_filter_blue = True
            
        elif self.material == 'model':
            self.filament_sp_use_right = True
            self.filament_sp_filter_blue = True
            self.ave_method_use_red = True
        
        
            
        #roi setup
        self.update_roi(roi_pnts,**kwargs)
        
        #zero arrays
#        self.meas_vals_init()   

        #speed estimation
        self.track_len = 2
        self.detect_interval = 2
        self.filament_sp_min_step = 0.75
        self.ave_method_use_red = False
        
        self.track_pnt_mask_radius = 2
        self.feature_params = dict(maxCorners = kwargs.get('maxCorners',50),
                                   qualityLevel = kwargs.get('qualityLevel',0.3),
                                   minDistance = kwargs.get('minDistance',2),
                                   blockSize = kwargs.get('blockSize',3))
        
        lk_params_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

#        self.lk_params = dict( winSize  = (30, 30),
#                              maxLevel = 2,
#                              criteria = lk_params_criteria)
                              
        self.lk_params = dict( winSize  = (15, 15),
                              maxLevel = 2,
                              criteria = lk_params_criteria)
        
        
        
        #data collection
        self.filament_speed = 0
        
        #internal
        self.prev_gray = None
        self.gray = None

        self.tracks = []
        
        #control
        self.frame_id = 0
        self.first_analysis = True

        
        #set apply method
        w_timestep = kwargs.get('w_timestep',False)
        if w_timestep:
            self.apply = self.apply_w_timestep
        else:
            self.apply = self.apply_wo_timestep
            
        
        if self.collect_on:
#            self.num_used_tracks = []
#            self.num_not_vertical_x_vs_y_frame = []
#            self.num_y_neg_frame = []
            #list collection with self.collect_on
            self.speed_step = []
            self.speed_mms = []
            self.time_step = []
            self.speed_compare = []
      
        
#------------------------------------------------------------------------------
    def update_roi(self,roi_pnts,**kwargs):
        '''
        Update roi parameters, dependent vars and other vars
        '''
        self.roi_x1,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        
        self.roi_pnt1 = (self.roi_x1,self.roi_y1)
        self.roi_pnt2 = (self.roi_x2,self.roi_y2)
        
        #speed conversion
        self.fps = kwargs.get('cam_fps',15)
        self.cal_factor = kwargs.get('cal_factor',0.0212) 
        self.speed_conversion = self.cal_factor*self.fps

#------------------------------------------------------------------------------
    def apply_w_timestep(self,roi,frame_id):
        '''
        '''
        e1 = cv2.getTickCount()
        
        self.apply_wo_timestep(roi,frame_id)
        
        e2 = cv2.getTickCount()
        self.time_step_current = (e2 - e1)/ cv2.getTickFrequency()
        if self.collect_on:
            self.time_step.append(self.time_step_current)
            self.speed_step.append(self.speed_step_current)
            self.speed_mms.append(self.speed_step_current*self.speed_conversion)
            if self.plot_on:
                self.plot_data()
        
#------------------------------------------------------------------------------
    def apply_wo_timestep(self,roi,frame_id):
        '''
        '''
        self.frame_id = frame_id
        
        self.fil_speed_preprocess(roi)
        fil_speed =  self.flow_track_filament()
#        fil_speed *= self.downscale_factor
        self.speed_step_current  = fil_speed
        
        self.first_analysis = False
        
#------------------------------------------------------------------------------
    def fil_speed_preprocess(self,roi):
        '''
        '''
        self.roi = cv2.GaussianBlur(roi, (3, 3),0)
        
        roi_test = self.roi.copy()
        if self.filament_sp_filter_blue:
            roi_test[:,:,1:] = 0
            
        self.gray = cv2.cvtColor(roi_test,cv2.COLOR_BGR2GRAY)
        left_start = 0  #max(0,x_left_ave)
        left_end = 30   #x_left_ave 
        
        ret_val,thresh_roi_1 = cv2.threshold(self.gray[:,left_start:left_end],0,255,self.thresh_type)

        self.gray[:,left_start:left_end] = thresh_roi_1
        
        if self.filament_sp_use_right:
            right_start = -10
            ret_val,thresh_roi_2 = cv2.threshold(self.gray[:,right_start:],0,255,self.thresh_type)
            self.gray[:,right_start:] = thresh_roi_2
            
            self.gray[:,left_end:right_start] = 0
                
        else:    
            self.gray[:,left_end:] = 0

#------------------------------------------------------------------------------
    def flow_track_filament(self):
        '''

        '''
        self.speed_xy_per_track = []
        #----------------------------------------------------------------------
        if self.tracks:
            self.track_points_update()
        #----------------------------------------------------------------------
        #update speed
        if len(self.speed_xy_per_track) > 1:
            fil_speed = reject_outliers(np.array(self.speed_xy_per_track),
                                         return_option = 1)
#            fil_speed = np.mean(self.speed_xy_per_track)

            
        else:
            fil_speed = 0
        #----------------------------------------------------------------------
        #get new points
        self.get_new_track_points()

        return fil_speed
#------------------------------------------------------------------------------  
    def track_points_update(self):
        '''
        '''
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, p0, None, **self.lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(self.gray, self.prev_gray, p1, None, **self.lk_params) #reverse
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
                self.fil_speed_append(x,y,prev_pnt) #this can be vectorised
    
            except IndexError:
                pass #first point in track
        #end for loop        
        self.tracks = new_tracks        
    #            if self.draw_on:
    #                cv2.circle(vis, (x, y), 2, (0, 0,255), -1)
        
    #        if self.draw_on:
    #            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
             #all points in a track
    #            for track in self.tracks:
    #                for x,y in track:
    #                    cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)
    
    #            cv2.imshow('track_pnts',vis)    
#------------------------------------------------------------------------------
    def get_new_track_points(self):
        '''
        '''
        if self.frame_id % self.detect_interval == 0 or self.first_analysis:
            try:
                mask = self.new_pnts_mask.copy()
            except AttributeError:    
                self.new_pnts_mask = np.zeros_like(self.gray) + 255
                mask = self.new_pnts_mask.copy()

            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), self.track_pnt_mask_radius, 0, -1)
            p = cv2.goodFeaturesToTrack(self.gray, mask = mask, **self.feature_params)
            
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])
#                    if self.draw_on:
#                         cv2.circle(self.roi_test, (x, y), 2, (0, 255, 0), -1)
        #update prev
        self.prev_gray = self.gray.copy()
#------------------------------------------------------------------------------
    def fil_speed_append(self,x,y,prev_pnt):
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
#                self.speed_xy_per_track.append(math_sqrt(h)) 
    

#------------------------------------------------------------------------------
    def plot_init(self):
        '''
        '''
        self.ax_xy_p_track = self.plot_axes[0,0]
        self.ax_1 = self.plot_axes[1,0]
#------------------------------------------------------------------------------
    def plot_data(self):
        '''
        '''
        self.speed_xy_per_track.sort()
        
        self.ax_xy_p_track.cla()
        self.ax_xy_p_track.plot(self.speed_xy_per_track)  
        self.ax_xy_p_track.axis(ymin=0,ymax=10)             
#==============================================================================
#==============================================================================
class FilamentWidthMethod():
    def __init__(self,frame,roi_pnts,**kwargs):
        '''
        Estimate filament width
        
        frame = roi area in main frame
        roi_pnts = (x1,y1,x2,y2),   coords relative to main frame
                
        
        Method:
            Assume user aligns roi so that vline (vertical alignment line) is 
            approx. on the left and right edge of the filament.
            
            Then extract two sub-rois, left and right, assuming the filament edges
            is in this area - within a certain distance from the left and right vline.
            
            Then zoom (magnify) these areas.
            
            Estimate edges using min/max of the first derivative 
            along the average over the x cols.
            
            Convert back for final answer.
            
            
        detection kwargs
        
         '''
        #info
        self.name_id = kwargs.get('name_id','Width_0')
        
        #debug control
        self.plot_on = kwargs.get('plot_on',False) # plots data
        self.draw_on = kwargs.get('draw_on',False) #draws addionational screens
        self.collect_on = kwargs.get('collect_on',False) #collects data in list
        self.collect_on = self.collect_on or self.plot_on or self.draw_on
        
        #plotting placeholder, for debuging/testing
        self.plot_axes = None
        
        #configs
        self.configs = kwargs
        
        self.material =self.configs.get('mat_colour','white').lower()
        
        #width estimation
 
        #sub_pixel method
        self.w_zf = self.configs.get('w_zf',4) #zoomfactor
        self.w_detection_half_width = self.configs.get('w_detection_half_width',10)
        self.w_vline_left_border = self.configs.get('w_vline_left_border',20)
        self.w_vline_right_border = self.configs.get('w_vline_right_border',15)
        
        
        #pre-process
        #material specific
  
                                    
        #START INIT
        #roi region, size parameters - recalculate on change of ROI/params etc.
        self.roi_x1 ,self.roi_y1,self.roi_x2,self.roi_y2 =  roi_pnts
        roi_bgr = frame[self.roi_y1:self.roi_y2,self.roi_x1:self.roi_x2]
        self.roi = cv2.GaussianBlur(roi_bgr, (3, 3),0)
        self.gray = cv2.cvtColor(self.roi,cv2.COLOR_BGR2GRAY)
        self.update_roi(roi_pnts,**kwargs)

        #init first frames
        
        
        #data collection
        self.time_step_current = 0
        
        #addtional data for review, used with collect_on/plot_on
        if self.collect_on:
            self.width_history = []
            self.time_step = []


        #set apply method
        w_timestep = kwargs.get('w_timestep',False)
        if w_timestep:
            self.apply = self.apply_w_timestep
        else:
            self.apply = self.apply_wo_timestep
            

        
#------------------------------------------------------------------------------
    def update_roi(self,roi_pnts,**kwargs):
        '''
        Update roi parameters and dependent vars
        '''
        self.cal_factor = kwargs.get('cal_factor',0.0212) 
        
        #width values
#        self.width_average = [] #average of w_vals_f per frame - track delta width
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
            

        
        
#------------------------------------------------------------------------------
    def width_preprocess(self,frame):
        '''
        Preprocess raw frame, apply gear mask
        '''
        self.frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.downscale_factor != 1:
            self.frame_gray = cv2.resize(self.frame_gray,(self.roi_new_w,self.roi_new_h))
        self.frame_gray[self.gear_mask_indices]  = 0


#------------------------------------------------------------------------------        
    def apply_w_timestep(self,frame,*args,**kwargs):
        '''
        Estimate speed and measure timestep for whole process
        '''   
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
#------------------------------------------------------------------------------        
    def apply_wo_timestep(self,roi_bgr,*args,**kwargs):
        '''
        Estimate width, without measuring the time step
        
        
        '''
        #pre-process
        self.roi = cv2.GaussianBlur(roi_bgr, (3, 3),0)
        self.gray = cv2.cvtColor(self.roi,cv2.COLOR_BGR2GRAY)

        #process
        self.width_subpixel_method()
                     
#------------------------------------------------------------------------------
    def width_subpixel_method(self):
        '''
        The filament width ROI has 2 sub-ROIs - left and right.
        
        Detect edge in each sub-ROI. 
        Subpixel resolution achieved by zooming image.
        
        '''
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
#        max_ave = diff_ave[turn_pnt-self.w_zf:turn_pnt+self.w_zf]
#        if len(max_ave) > 0:
#            max_ave_centre = np.average(max_ave, weights=max_ave)
#            edge_pos = turn_pnt-self.w_zf + max_ave_centre
#        else:
#            edge_pos = turn_pnt
#        return edge_pos    
        return turn_pnt
#------------------------------------------------------------------------------
    def plot_init(self):
        '''
        '''
        self.axes_all = self.plot_axes[0,0]
        self.axes_lr = self.plot_axes[0,1]
        
        self.axes_avecols_left  = self.plot_axes[1,0]
        self.axes_avecols_right = self.plot_axes[1,1]
        
        self.axes_diffcols_left  = self.plot_axes[2,0]
        self.axes_diffcols_right = self.plot_axes[2,1]
        
        
        
#------------------------------------------------------------------------------
    def plot_data(self):
        '''
        '''
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
        
#------------------------------------------------------------------------------
    def draw_edges(self):
        '''
        '''
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
    def __init__(self,video_fullname,video_config,apply_methods,**kwargs):
        '''
        apply_methods, list of dicts, defining which methods to apply.
        
        '''
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
        self.method_types = {}
        self.method_types['GLK'] = [GearLKMethod,'gear'] #method and roi points 
        self.method_types['GDM'] = [GearDenseMethod,'gear']
        self.method_types['FLK'] = [FilamentLKMethod,'fil_speed']
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
            
            #roi mask circles
            if method_name in ['GLK','GDM']:
                m0 = new_method
                self.draw_roi_circles['cr_max'] = [(0,255,125),(m0.cx, m0.cy),m0.cr_max]
                self.draw_roi_circles['cr_od'] = [(0,255,255),(m0.cx, m0.cy),m0.cr_od]
                self.draw_roi_circles['cr_min'] = [(0,255,125),(m0.cx, m0.cy),m0.cr_min]

        
        
        
        
        #PLOT SETUP
        self.plot_init()
        

#------------------------------------------------------------------------------
    def main_loop(self):
        '''
        '''
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
#------------------------------------------------------------------------------
    def draw_frame(self,frame):
        '''
        draw information on frame
        '''
        #POST FRAME 
        #ROIs
        for roi in self.draw_roi_rects.values():
            colour,xy1xy2 = roi
            cv2.rectangle(frame, (xy1xy2[0],xy1xy2[1]), (xy1xy2[2],xy1xy2[3]), colour, 2)
        
        #mask circle roi
        for circles in self.draw_roi_circles.values():
            colour,cxy,radius = circles
            cv2.circle(frame, cxy, radius, colour, 1)
        
         #cx and cx        
        cv2.line(frame,(self.gear_centre[0],0),(self.gear_centre[0],frame.shape[0]),(255,255,0))
        cv2.line(frame,(0,self.gear_centre[1]),(frame.shape[1],self.gear_centre[1]),(255,255,0))
        
        

        
        #vline left
        cv2.line(frame,(self.controls['vline_left'],0),(self.controls['vline_left'],frame.shape[0]),(255,255,0))
        
        #show
        cv2.imshow('source',frame)
                
            
#------------------------------------------------------------------------------
    def plot_init(self):
        '''            
        '''
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
#------------------------------------------------------------------------------
    def plot_data(self):
        '''
        '''        
        if (self.plot_update or self.plot_cont):
            self.plot_update = False
            
            #plot all data
            self.all_data_ax.cla()
            self.all_pxl_ax.cla()
            self.all_speed_ax.cla()
            self.all_width_ax.cla()
            
            
            for index,method in enumerate(self.methods):
                self.all_data_ax.plot(method.time_step,label = method.name_id)
                if method.method_name == 'FWM': #width
                    self.all_width_ax.plot(method.width_history,'o-')
                else:
                    self.all_pxl_ax.plot(method.speed_step,label = method.name_id)
                    self.all_speed_ax.plot(method.speed_mms,label = method.name_id)

            #anotate
            self.all_data_ax.legend()
            self.all_data_ax.set_title('Time Step (s)')
            
            self.all_pxl_ax.legend()
            self.all_pxl_ax.set_title('Speed (pxl/frame)')
            
            self.all_speed_ax.legend()
            self.all_speed_ax.set_title('Speed (mm/s)')
            
#            self.all_width_ax.legend()
            self.all_width_ax.set_title('Width (pxl)')
            
               
#------------------------------------------------------------------------------
    def playback_control(self):
        '''
        poll keyboard for user input
        '''
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
                    
#------------------------------------------------------------------------------
    def exit_test(self):
        '''
        '''
        #END MAIN LOOP - EXIT
        self.video_capture.release()    
        cv2.destroyAllWindows()
#------------------------------------------------------------------------------    

    def init_video(self):
        '''
        load video and video configs, if available
        '''
        self.video_capture = cv2.VideoCapture(self.video_fullname)
        self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        

        
        x_c = self.config.getint('vision','x_gear_c')
        y_c = self.config.getint('vision','y_gear_c')
        self.gear_centre = (x_c,y_c)
        self.fps = self.config.getint('camera','cam_fps')
        self.frame_id = self.config.getint('vision','frame_start')
        self.od_resize_factor = self.config.getfloat('vision','od_resize_factor')
        
        self.control_vars = []
        self.control_vars.append(('vline_right',False))
        self.control_vars.append(('vline_left',False))
        self.control_vars.append(('y_pre_end',False))
        self.control_vars.append(('y_post_start',False))
        self.control_vars.append(('x_gear_c',True))
        self.control_vars.append(('y_gear_c',False))
        
#
        self.control_vars.append(('x_gear_teeth',False))
        self.control_vars.append(('gear_roi_x2',False))
        self.control_vars.append(('gear_roi_y2',False))
        self.control_vars.append(('cr_add',False))
        self.control_vars.append(('cr_min',False))
#        
        self.control_vars.append(('w_vline_left_border',False))
        self.control_vars.append(('w_vline_right_border',False))


        
        self.config_dict = make_config_dict(self.config)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,self.frame_id)
        

#------------------------------------------------------------------------------
    def init_windows(self,frame = None,**kwargs):
        '''
        '''
        
        cv2.namedWindow('source')
        
        self.controls = {}
        for variable,show_trackbar in self.control_vars:
            init_val = self.config_dict['vision'][variable]
            
            if show_trackbar:
                cv2.createTrackbar(variable,'source',init_val,640,self.on_trackbar) 
                
            self.controls[variable] = init_val
       
        
        cv2.createTrackbar('aX','source',311,640,self.on_trackbar)
        cv2.createTrackbar('aY','source',167,640,self.on_trackbar)
        
        cv2.createTrackbar('bX','source',334,640,self.on_trackbar)
        cv2.createTrackbar('bY','source',364,640,self.on_trackbar)
        
        cv2.moveWindow('source',0,0)
    #    cv2.moveWindow('ROI',640,0)
    #    cv2.moveWindow('Thresh',640*2,0)
#------------------------------------------------------------------------------        
    def get_roi_coords(self,roi_type = 'gear'):
        '''

        
        '''
        
        for variable,read_trackbar  in self.control_vars:
            if read_trackbar:
                self.controls[variable] = cv2.getTrackbarPos(variable,'source')
       
        
        self.gear_centre = (self.controls['x_gear_c'],self.controls['y_gear_c'])
        
        method_kwargs = {}
        if roi_type == 'width':
            #PRE GEAR AREA- estimate filament width
            x_start = self.controls['vline_left'] - self.controls['w_vline_left_border']#20
            y_start = self.controls['y_pre_end'] - 40
            x_end = self.controls['vline_right'] + self.controls['w_vline_right_border']#15
            roi_pnts = (x_start,y_start, x_end, self.controls['y_pre_end'])
            
            width_default = self.controls['vline_right'] - self.controls['vline_left']
            method_kwargs['width_default'] = width_default
            method_kwargs['w_vline_left_border'] = self.controls['w_vline_left_border']
            method_kwargs['w_vline_right_border'] = self.controls['w_vline_right_border']
            
            self.draw_roi_rects['width'] = [(75,125,225),roi_pnts]
            
            
        elif roi_type == 'fil_speed':
            #POST GEAR - filament speed
            x_start = self.controls['vline_left'] -self.controls['x_gear_teeth']
            x_end = self.controls['vline_right'] + 10
            y_end = 480- 10 #self.controls['y_post_end_offset']
            roi_pnts = (x_start,self.controls['y_post_start'],x_end, y_end)
            
            self.draw_roi_rects['fil_speed'] = [(225,125,75),roi_pnts]
            
        elif roi_type == 'gear':
            #GEAR
            x_start = max(0,self.gear_centre[0]) #left top pnt
            y_start = max(0,self.gear_centre[1])  #left top pnt
            x_end = self.controls['gear_roi_x2']
            y_end = self.controls['gear_roi_y2']
            roi_pnts = (x_start,y_start,x_end,y_end)

            method_kwargs['gear_centre'] = self.gear_centre
            
            self.draw_roi_rects['gear'] = [(75,125,225),roi_pnts]
        
        elif roi_type == 'gearSDM':
            #GEAR
            x_start =  cv2.getTrackbarPos('aX','source') #left top pnt
            y_start = cv2.getTrackbarPos('aY','source')  #left top pnt
            x_end = x_start + 40
            y_end = y_start + 40
            roi_pnts = (x_start,y_start,x_end,y_end)

            
            self.draw_roi_rects['gearSDM'] = [(125,75,225),roi_pnts]
        
        elif roi_type == 'filSDM':
            #fill
            x_start =  cv2.getTrackbarPos('bX','source') #left top pnt
            y_start = cv2.getTrackbarPos('bY','source')  #left top pnt
            x_end = x_start + 40
            y_end = y_start + 40*2
            roi_pnts = (x_start,y_start,x_end,y_end)

            
            self.draw_roi_rects['filSDM'] = [(125,125,225),roi_pnts]
        
        return roi_pnts,method_kwargs
#------------------------------------------------------------------------------
    def on_trackbar(self,new_val):
        '''
        '''
        self.process_frame = True
#==============================================================================
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
if __name__ == '__main__':
    '''
    method type, general kwargs:
        method_name : LK, DM
        
        plot_on: False
        draw_on: False
        downscale_factor: 1
        w_timestep: False
    

        
    '''
    import configparser
    print('main')
    #------------------------------------------------------------------------
    do_profiling = False
    
    #------------------------------------------------------------------------
    #DATA to load
#    data_name = 'data\\Test_2017_02_17__08_45_46'
    data_name = 'data\\Test_2017_04_11__14_27_24'
    
    video_fullname = data_name +'.avi'
    config_filename = data_name +'.cfg'
    #get config
    config = configparser.RawConfigParser()
    config.optionxform = str
    config.read(config_filename)
        
#        video_fullname = 'data\\Test_2017_02_17__08_43_56.avi'
#        config_filename = 'data\\Test_2017_02_17__08_43_56.cfg'
        
#        video_fullname = 'data\\N5_Mwhite285_2016_05_24__11_39_22_T2300_L500_S20_0.avi'
#        config_filename = 'data\\N5_Mwhite285_2016_05_24__11_39_22_T2300_L500_S20_0.cfg'
        
#        video_fullname = 'data\\modelling\\unc_model_0b s2_L20.avi'
#        config_filename = 'data\\modelling\\unc_model_0b s2_L20.cfg'

#        video_fullname = 'data\\modelling\\unc_model_0c s2_L2_5_f262.avi'
#        config_filename = 'data\\modelling\\unc_model_0c s2_L2_5_f262.cfg'
        
    #------------------------------------------------------------------------
    #METHODS to appply    
    methods = []
    w_timestep = not(do_profiling)
    GLK_11 =   {'method_name': 'GLK',
               'downscale_factor':1,
               'w_timestep': w_timestep,
               'plot_on':False,
               'draw_on':False}
    
    GDM_11 = {'method_name': 'GDM',
            'downscale_factor':1,
            'w_timestep': w_timestep,
            'id': 'x1',
            'plot_on':False,
            'draw_on':False}
    
    GDM_21 = {'method_name': 'GDM',
            'downscale_factor':2,
            'w_timestep': w_timestep,
            'id': 'x2',
            'plot_on':False,
            'draw_on':False}

    FLK_11 = {'method_name': 'FLK',
            'w_timestep': w_timestep,
            'id': 'fil',
            'plot_on':False,
            'draw_on':False}
    
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
            'plot_on':False,
            'draw_on':False}
            
#    methods.append(GLK_11)
#    methods.append(GDM_11)
    methods.append(GDM_21)
    methods.append(FLK_11)
    methods.append(FWM_11)
    methods.append(SMD_g1)
    methods.append(SMD_f1)

    
    #------------------------------------------------------------------------
    test_roi = TestROI(video_fullname,config,methods,profile = do_profiling)
    
    test_roi.main_loop()
