# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:55:19 2017

https://www.boxcontrol.net/embedding-matplotlib-plot-on-pyqt5-gui.html
https://github.com/boxcontrol/matplotlibPyQt5/blob/master/matplotlibPyQt5.py

Modified 11.01.2017, GP Greeff

https://taher-zadeh.com/speeding-matplotlib-plotting-times-real-time-monitoring-purposes/


"""



import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#from matplotlib.pyplot import subplots

#===============================================================================
class MplCanvasWidget(FigureCanvas):
    """
    Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    """
    def __init__(self, parent = None, width = 5, height = 4, dpi = 100,
                 subplots_num = 2,initial_data = None):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = []
        
        if subplots_num == 2:
            self.axes.append(fig.add_subplot(121))
            self.axes.append(fig.add_subplot(122))
            
        elif  subplots_num == 4:          
            self.axes.append(fig.add_subplot(221))
            self.axes.append(fig.add_subplot(222))
            self.axes.append(fig.add_subplot(223))
            self.axes.append(fig.add_subplot(224))
#        fig, self.axes = subplots(2,2,num = 'Data',figsize=(width, height), dpi=dpi)

 
        self.compute_initial_figure(initial_data)

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        
#        self.background = self.copy_from_bbox(self.axes.bbox) # cache the background
        self.figure.set_facecolor('white')
#        fig.set_facecolor = 'white'
        print('init_done')
        
        
        
        
#------------------------------------------------------------------------------
    def compute_initial_figure(self,initial_data = None):
        '''
        intitial data: list of dicts, each dict kwargs for plot.
        
        '''
        self.artists = []
        
        if initial_data is None:
            line = self.axes.plot([1,2,3,4])
            self.artists.append(line[0])
        else:
            for axes_id,plot_args,plot_kwargs in initial_data:
#                print(axes_id)
#                print(plot_args)
#                print(plot_kwargs)
                xmin,xmax = plot_kwargs.pop('xmin',0),plot_kwargs.pop('xmax',100)
                ymin,ymax = plot_kwargs.pop('ymin',0),plot_kwargs.pop('ymax',100)
                if axes_id is None:
                    ax1 = self.axes
                    
                else:
#                    ax1 = self.axes.flat[axes_id] #subplots
                    ax1 = self.axes[axes_id]
                    
#                ax1.hold(True)
                line = ax1.plot(*plot_args,**plot_kwargs)
                self.artists.append(line[0])
                
                ax1.axis([xmin,xmax,ymin,ymax ])
        
#        self.draw()
        
#------------------------------------------------------------------------------    
    def fast_update(self,update_data):
        '''
        '''
        for index,xy in enumerate(update_data):
            self.artists[index].set_xdata(xy[0])
            self.artists[index].set_ydata(xy[1])
#            print()
#            print(index)
#            print(self.artists[index].properties())
        self.draw()
#------------------------------------------------------------------------------    
    def fast_update_y(self,update_data_y):
        '''
        '''
        for index,y in enumerate(update_data_y):
            self.artists[index].set_ydata(y)

        self.draw()        

        
         
#-------------------------------------------------------------------------------    
#    def resizeEvent(self,event):
#        super(MplCanvasWidget, self).resizeEvent(event) 
#        self.background = self.copy_from_bbox(self.axes.bbox) # cache the background
        
#-------------------------------------------------------------------------------        
    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)    

#===============================================================================
if __name__ == '__main__':
    pass
#    import traceback
#    if QtCore.QT_VERSION >= 0x50501:
#        def excepthook(type_, value, traceback_):
#            traceback.print_exception(type_, value, traceback_)
#            QtCore.qFatal('')
#        sys.excepthook = excepthook
#            
#    app = QApplication(sys.argv)
#
#    aw = ApplicationWindow()
#    aw.setWindowTitle("PyQt5 Matplot Example")
#    aw.show()
#    #sys.exit(qApp.exec_())
#    app.exec_()