# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:26:59 2013


Special methods for pyqt5 load/save and configs,


REVISION:
    2.0.0   2017-01-06
            - Python 3 and PyQt5 version, only kept certain functions.
            - did not re-implement UiConfigInterface for now
            
    1.0.1   2015-07-10
            - added QDateEdit into ui_types
            
    1.0.0   2015_07_09  
            -general config methods
            -additional ui_config interface  
    
    0.1.1   2015_06_23  
            -Add UI file load/save and basic file interactions methods
            -Name changed to QtLoadSaveObjects
    
    0.1.0   2015_06_15  
            -First Use (with RUI app)
            
    0.0.1   2015-06_04 
            -Named QtConfigToolsObjects
        
"""

__version__ = "2.0.0"
__author__ = "GP Greeff"


#from PyQt5 import QtGui
from PyQt5 import QtWidgets,QtCore

import os,time,glob
#import logging
import json


#FILENAME GENERATION / FOLDERS
 
def make_date_filename(first = None,last = None,ext = None,folder = None):
    """
    Make a date & time based file name.

    format of filename:    
        folder \ 'first_text' + 'yyyy_mm_dd_hh_mm' + 'extra_text' + 'extension'
        
    E.g.:
        Test_2015_01_15__01_01_01_last.txt
    
    """
    
    filename = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    
    if first is not None:
        filename = '_'.join([first,filename])
        
    if last is not None:
        filename = '_'.join([filename,last])
        
    if ext is not None:
        filename = '.'.join([filename,ext.lstrip('.')])
    
    if folder is not None:
        filename = os.path.join(folder,filename)
        
    return filename 
    
def clean_text(text,default_text = 'text'):
    """
    remove unwanted chars from text, for folders/filenames
    text must be string
    from stackoverflow answer/comment
    """    
    text = str(text)
    new_text = ''.join(i for i in text if i not in "\/:*?<>|")
    new_text = new_text.strip()
    if len(new_text) == 0:
        new_text = default_text
    return new_text

def check_unique_filename(parent,dir_path,filename,name_delim = '_'):
    """
    determines if filepath exits,
    if does ask user if the file must be replaced,
    otherwise generate a new name.
    """
    msg = 'Overwrite file: {}?'
    new_dir_name = os.path.dirname(dir_path)
    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)
        new_dir_made =  True
    new_dir_made =  False

    if not(new_dir_made):
       #       'check unique'
        if os.path.isfile(dir_path + filename):
            msg = msg.format(filename)
            reply = QtWidgets.QMessageBox.question(parent, 'Overwrite?', 
                         msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            #make a unique name, must be in format text_text_0.ext             
            if reply  == QtWidgets.QMessageBox.No: 
                number_tries = 0
                max_tries = 99
                while number_tries < max_tries:
                    if os.path.isfile(dir_path + filename):
                        name,ext = os.path.splitext(filename)
                        name = name.split(name_delim)
                        try:
                            name[-1] = str(int(name[-1])+1)
                        except ValueError:
                            name.append('1')
                            number_tries = max_tries
                        new_name = name_delim.join(name)
                        new_name += ext
                        filename = new_name
                        number_tries += 1
                    else:
                        break
    dir_path += filename
    return dir_path 

def get_files_like(folder,filename_contains,iterator = False,ext = 'txt'):
    """
    get files in folder that contains filename_contains with extention ext
    
    return an iterator if True, otherwise a list
    """
    
    filenames = None
    if folder is not None:
        if folder[0] == 'D':
            search = folder+"*"+filename_contains+"*."+ext
        else:
            search = "."+folder+"*"+filename_contains+"*."+ext
    else:
        if filename_contains[:2] == 'C:':
            search = filename_contains+"*."+ext
        else:
            search = "*"+filename_contains+"*."+ext
    
    if iterator:    
        filenames = glob.iglob(search)
    else:
        
        filenames = glob.glob(search)
#        print(filenames)
        if not filenames:
            pass
#            logging.warning('No files found with search: {}'.format(search))
#            print('No files found with search: {}'.format(search))
    return filenames 

def get_folders(main_directory):
    """
    return a list of all the subdirectories in the main_directory
    http://stackoverflow.com/questions/973473/
    getting-a-list-of-all-subdirectories-in-the-current-directory
    
    return None if no sub-folders
    """
    folders = [x[0] for x in os.walk(main_directory)]
    folders = folders[1:]
    
    if len(folders) == 0:
        return None
    return folders
    
def get_txt_and_cfg_pair(folder):
    """
    find a matching pair in folder,
    X.cgf and X.txt
    
    if none found or first cfg found unpaired return None
    
    else return [X.cgf,X.txt]
    """
    cfg_file = None
    cfg_file_root = None
    txt_file = None
    txt_file_root = None
    files = [x[2] for x in os.walk(folder)][0]
    if len(files) >= 2: 
        for file_name in files:
            root,ext = os.path.splitext(file_name)
            if ext.find('cfg') >= 0:
                cfg_file = file_name
                cfg_file_root = root
                break
            
        #find matching txt
        for file_name in files:
            root,ext = os.path.splitext(file_name)
            if ext.find('txt') >= 0:
                txt_file = file_name
                txt_file_root = root
                if txt_file_root == cfg_file_root:
                    return [cfg_file,txt_file]
        
    return None
    
def set_save_file_location(parent, default_dir,caption = "Save Data",
                        fileType = 'Text File (*.txt)',defaultFile = None):
    """
    Launch a QFileDialog.getSaveFileName and check if filename is correct.
    
    Return True on success, with filename string
    """
    resultFileName = None
    if not(os.path.isdir(default_dir)):
        default_dir = os.getcwd()
    if defaultFile != None:
        default_dir += '\\'+defaultFile
        
    fname = str(QtWidgets.QFileDialog.getSaveFileName(parent, caption, default_dir,fileType))
    
#        print default_dir,fname
    if (fname != None) and (fname):
        try:
            resultFileName = os.path.relpath(fname)
        except ValueError: #not drive C:
            resultFileName = fname
        return True,resultFileName
            
    return False,resultFileName

def load_file_name(parent, default_dir = None,caption = "Load Data",
                 type_filter = 'Text File (*.txt)'):
    """
    user requested to selects filename, 
    return filename or None (not succesful)
    """
#    filename = None
    if default_dir is None or not(os.path.isdir(default_dir)):
        default_dir = os.getcwd()
    
    test_name = QtWidgets.QFileDialog.getOpenFileName(parent, caption, 
                                                  default_dir,type_filter)

    test_name = str(test_name[0])
#    print(default_dir,test_name)
#    if test_name != None:
#        if os.path.isfile(test_name):
#            try:
#                filename = os.path.relpath(test_name)
#            except ValueError:
#                filename = test_name
    return test_name

def load_directory_name(parent,default_dir,caption = 'Select Directory for Files'):
    """
    request user to select a folder
    """
    if not(os.path.isdir(default_dir)):
        default_dir = os.getcwd()

    test_dir = QtWidgets.QFileDialog.getExistingDirectory(parent, caption,default_dir)
    test_dir = str(test_dir)
    
    dir_name = test_dir_exists(test_dir)
    
    return dir_name

def test_dir_exists(test_dir):
    """
    test if test_dir exists
    """
    dir_name = None
    if test_dir != None and test_dir:
        if os.path.isdir(test_dir):
            try:
                dir_name = os.path.relpath(test_dir)
                dir_name = '\\' + dir_name + '\\'
            except ValueError:
                dir_name =  test_dir + '\\'
    return dir_name

def split_path(path):
    """
    http://stackoverflow.com/questions/3167154/
    how-to-split-a-dos-path-into-its-components-in-python
    """
    path = os.path.normpath(path)
    a,b = os.path.split(path)
    return (split_path(a) if len(a) and len(b) else []) + [b]


def set_combo_index(combo_box,text):
    """
    http://stackoverflow.com/questions/22797794/
    pyqt-how-to-set-combobox-to-item-knowing-items-text-a-title
    """
    index = combo_box.findText(text, QtCore.Qt.MatchFixedString)
    if index >= 0:
         combo_box.setCurrentIndex(index)

def make_config_dict(config):
    """
    config to dict, try to convert str to floats/ints
    
    http://stackoverflow.com/questions/1773793/
    convert-configparser-items-to-dictionary
    """

    return {s:{key:convert_str(value) for key,value in config.items(s)} for s in config.sections()}

def convert_str(convert_value):
    """
    first try float, if a float, try int
    else leave str
    """
    try:#as float
        convert_value = float(convert_value)
        if convert_value.is_integer():
            convert_value = int(convert_value)
    except ValueError:
        convert_value = str(convert_value)#as string
        
    return convert_value

def append_texteditor(text_edit_widget,data = '',replace = '"'):
    """
    append text editor and scroll to the bottom
    """
    data = json.dumps(data, indent = 4)
    for item in replace:
        data = data.replace(item,'')
    text_edit_widget.append(data)
    text_edit_widget.ensureCursorVisible()

if __name__ == '__main__':
    print('main')
    import configparser
    config = configparser.RawConfigParser()
    config.optionxform = str
    config_filename = 'config_vision.cfg'
#    
    read_result = config.read(config_filename)
    print('read config: ',read_result)
    
    for s in config.sections():
        print(s)
        for key,val in config.items(s):
            print(key,val,type(val))
#        print(config.items(s))
        print()
    config_dict = make_config_dict(config)
    
    print(config_dict)
    