# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:26:59 2013

@author: pgreeff

special methods for pyqt qtable

REVISION:
1.0   2015-05-30    RCS_2 first version
2.0   2017-01-13    Py3 and PyQt5   
        
"""

__version__ = "2.0."
__author__ = "Pieter Greeff"


from PyQt5 import QtCore
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QColor
#import copy

#TABLE METHODS
#-------------------------------------------------------------------------------
def set_row_headers(table,header_list, fmt = '{}'):
    '''
    set row headers as per list
    '''
    for index,header in enumerate(header_list):
        header = fmt.format(header)
        header_item = make_item(header)
        table.setVerticalHeaderItem(index,header_item) 
#-------------------------------------------------------------------------------
def set_column_headers(table,header_list, fmt = '{}'):
    '''
    set column headers as per list
    '''
    for index,header in enumerate(header_list):
        header = fmt.format(header)
        header_item = make_item(header)
        table.setHorizontalHeaderItem(index,header_item)        
#-------------------------------------------------------------------------------
def table_colours(color):
    '''
    '''
    if color == 'normal':
        return QColor("white")
    elif color == 'selected':
        return QColor("red")
    elif color == 'green':
        return QColor("green")
    elif color == 'grey':
        return QColor("grey")
    else:
        return QColor("white")

#-------------------------------------------------------------------------------
def make_item(itemStr,bckColor = None,enabledFlag = True,checked = False):
    '''
    '''
    table_item = QTableWidgetItem(str(itemStr))
    
    if not(enabledFlag):
        table_item.setFlags(QtCore.Qt.ItemIsEnabled)
        
    if checked:
        table_item.setCheckState(QtCore.Qt.Checked)
    
    if bckColor is not None:
        table_item.setBackground(table_colours(bckColor))
    return table_item
#-------------------------------------------------------------------------------
def set_item(table,itemStr,row, col,bckColor = None,enabledFlag = True,
             checked = False):
    '''
    set table with itemStr @ (row,col) with bckColor, editability is optional
    '''
    table_item = make_item(itemStr,bckColor,enabledFlag,checked)
    table.setItem(row,col,table_item)

#-------------------------------------------------------------------------------
def set_table_column(table,item_list,column,bck_color = 'normal',enabled_flag = True,
                   format_string = "%3.1f",checked = True):
    '''
    Set table with items in item_list @ (col),
    with bckColor, editability is optional
    '''
    for row,new_item in enumerate(item_list):
        if format_string != None:
            new_item = format_string % new_item
        table_item = make_item(new_item,bck_color,enabled_flag,checked = True)
        table.setItem(row,column,table_item)


#-------------------------------------------------------------------------------
def reset(table,clearStr = '',enabledFlag = True):
    '''
    '''
    #clear previous values from all cells
    for rowIndex in range(table.rowCount()):
        for colIndex in range(table.columnCount()):
            set_item(table,clearStr,rowIndex, colIndex,'normal',enabledFlag)

#-------------------------------------------------------------------------------
def set_bck_color(table,column,row,color = 'normal'):
    '''
    '''
    table.setCurrentCell(row,column)
    tableItem = table.currentItem()
    try:
        tableItem.setBackground(table_colours(color))
    except AttributeError:
        pass
#-------------------------------------------------------------------------------
def set_all_color(table,color = 'normal'):
    '''
    '''
    for row in range(table.rowCount()):
        for col in range(table.columnCount()):
            set_bck_color(table,col,row,color)
#-------------------------------------------------------------------------------
def set_index_color(table,column,color= 'normal',indexList = None):
    '''
    set background colour of rows in indexList, of column of table.
    if no indexList, then the entire column
    '''
    if indexList == None:
        indexList = range(table.rowCount())
    for row in indexList:
#        print row,column
        set_bck_color(table,column,row,color)
#-------------------------------------------------------------------------------
def get_row_vals(table,columns,row):
    '''
    '''
    #get values in row, from columns[0] to columns[1]
    rowVals = []
    #print 'tableget_row_vals',row,columns,range(columns[0],columns[1])
    for column in range(columns[0],columns[1]):
        if column < table.columnCount():
            table.setCurrentCell(row,column)
            try:
                itemText = table.currentItem().text()
    #            print 'text',itemText
                try:
                    itemText = float(itemText)
                    rowVals.append(itemText)
                except ValueError:
                    pass
            except AttributeError:
                pass
    return rowVals
#-------------------------------------------------------------------------------
def get_column_vals(table,column = 0,rows = None,only_if_checked = False,
                    min_num_vals = 1):
    '''
    try to get float values from colum
    in row, from rows[0] to rows[1], or all rows if None
    
    if only_if_checked True, only get checked items.
    
    if list len > min_num_vals
        return val list on 
    else:
        return None
    '''
    col_vals = []
#    all_col_vals = []
    if rows == None:
        rowGen = range(0,table.rowCount())
    else:
        rowGen = range(rows[0],rows[1])
#    print 'tableget_column_vals',rows,column,rowGen
    for row in rowGen:
        if row < table.rowCount():
            table.setCurrentCell(row,column)
            try: #check item exists
                current_item = table.currentItem()
                itemText = current_item.text()
                try: #check items is float
                    itemText = float(itemText)
                    if not(only_if_checked) or current_item.checkState():
                        col_vals.append(itemText)
#                    all_col_vals.append(itemText)
                except ValueError: #no float
                    pass
            except AttributeError: #no item
                pass
    if (len(col_vals) >= min_num_vals):
        return col_vals
    else:
        return None
        
        
    
#-------------------------------------------------------------------------------
def get_current_index(table):
    '''
    note: returns x,y
    '''
    return table.currentColumn(),table.currentRow()
#-------------------------------------------------------------------------------
def get_single_value_number(table,col,row):
    #try and get a number from a cell
    table.setCurrentCell(row,col)
    try:
        itemText = table.currentItem().text()
#        print itemText
        try:
            itemText = float(itemText)
            return itemText
        except ValueError:
            pass
    except AttributeError:
        pass
    return None
#-------------------------------------------------------------------------------
def set_table_dictlist(table,dictlist,col_order = None,header_upper = False):
    '''
    dictlist, list of dicts
    each dict with the same keys
    '''
    reset(table)
    num_rows = len(dictlist)
    num_cols = len(dictlist[0])

    col_headers = list(dictlist[0].keys())
    col_headers.sort()
    if col_order is not None:
        col_order.reverse()
        for col_header in col_order:
#            print(col_header,col_headers)
            col_headers.pop(col_headers.index(col_header))
            col_headers.insert(0,col_header)
    
    if header_upper:
        col_headers = [header.upper() for header in col_headers]
        
    table.setRowCount(num_rows)
    table.setColumnCount(num_cols)
    set_column_headers(table,col_headers)
    
    for row_index,row in enumerate(dictlist):
        for key,item in row.items():
            if header_upper:
                key = key.upper()
            col_index = col_headers.index(key)
            set_item(table,str(item),row_index,col_index)
    
    table.setCurrentCell(num_rows-1,0)
#    print('currentRow',table.currentRow())
#    set_row_headers(table,)
    
    
#-------------------------------------------------------------------------------