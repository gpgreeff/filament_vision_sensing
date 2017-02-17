# filament_vision_sensing
**Vision based filament sensing in molten extrusion deposition printing.**

Python and OpenCV is used to detect filament width and feed speed, basic GUI with PyQt5 and QtDesigner.

Multiprocessing is used to run two applications, in an attempt to achieve realtime machine vision. These processes are called ExtruderCam and MainControlPlaceHolder.

**ExtruderCam** runs can record video files as well estimate feed gear speed, filament speed and filament width. The image processing is done in ROIA module. 

**MainControlPlaceHolder** is only a placeholder for the printer and experiment control. It displays a realtime plot of the measured data, collects and saves the results with the help of Pandas.

# Reference
This work is based on the code, as used in this article:  

[Closed Loop Control of Slippage during Filament Transport in Molten Material Extrusion](https://authors.elsevier.com/a/1UOGJ7tcTWLpqU)

http://dx.doi.org/10.1016/j.addma.2016.12.005

Journal: Additive Manufacturing

    
# Usage:
Run the file MainControl_placeholder.py. In the file, set it to use videofile mode or realtime (camera) mode.
Example video included.

MainControlPlaceHolder is to be replaced/modified with printer interface and control.

# Environment:
- Windows 10
- Python 3.5.2 |Anaconda custom (64-bit)|
- OpenCV 3.1.0
- PyQt5 PyQt5.QtCore.QT_VERSION: 329218
- IDE: Spyder 3.1.2
- Printer: RF1000
- Camera: Dino-Lite Pro - AM413T
