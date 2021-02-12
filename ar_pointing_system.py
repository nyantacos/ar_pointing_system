# -*- coding: utf-8 -*-
import cv2 
import sys
import random
import numpy as np
import pyautogui
import pandas as pd
from pycaret.regression import *

import tkinter as tk
TkRoot = tk.Tk()

display_width = TkRoot.winfo_screenwidth()
display_height = TkRoot.winfo_screenheight()
 
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

df = pd.DataFrame()

COLUMNS = ['MX_1', 'MY_1', 'MX_2', 'MY_2', 'MX_3', 'MY_3', 'MX_4', 'MY_4', 'X', 'Y']
LIST_EXP = ['MX_1', 'MY_1', 'MX_2', 'MY_2', 'MX_3', 'MY_3', 'MX_4', 'MY_4']
N_TRAIN = 50
WIDTH_TRAIN_MARKER = 20
AVG_WINDOW=6

def train():

    list_x=[]
    list_y=[]
    
    df = pd.DataFrame()
    
    cap = cv2.VideoCapture(0) 
    cv2.namedWindow('train_mode', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('train_mode', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
 
    target_x = WIDTH_TRAIN_MARKER
    target_y = WIDTH_TRAIN_MARKER
    count = 0
    while True:
        ret, frame = cap.read()
 
        img = cv2.resize(frame,(display_width,display_height))
        cv2.line(img, (target_x-WIDTH_TRAIN_MARKER, target_y), (target_x+WIDTH_TRAIN_MARKER, target_y), (0, 0, 0), thickness=3)
        cv2.line(img, (target_x, target_y-WIDTH_TRAIN_MARKER), (target_x, target_y+WIDTH_TRAIN_MARKER), (0, 0, 0), thickness=3)

        corners, ids, _ = aruco.detectMarkers(img, dictionary) 
        aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
 
        cv2.imshow('train_mode', img) 
 
        key = cv2.waitKey(1)
        if key & 0xFF == ord(' ') and len(corners)>0:
            _df = pd.DataFrame(np.array(corners[0]).flatten().tolist() + [target_x, target_y]).T
            _df.columns = COLUMNS
            if len(df) == 0:
                df = _df
            else:
                df = pd.concat([df, _df])
            target_x = int(random.random()*(display_width-2*WIDTH_TRAIN_MARKER)+WIDTH_TRAIN_MARKER)
            target_y = int(random.random()*(display_height-2*WIDTH_TRAIN_MARKER)+WIDTH_TRAIN_MARKER)
            count += 1
            if count >= N_TRAIN:
                break
        elif key & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    
    df = df.reset_index(drop=True)
    df.to_csv('df.csv', index=False)

    df_train_X = df[LIST_EXP+['X']]
    df_train_Y = df[LIST_EXP+['Y']]

    print(LIST_EXP+['X'])
    print(LIST_EXP+['Y'])

    setup(data = df_train_X, target = 'X', silent=True, session_id=42) 
    print('HOGEHOGE')
    m_x = create_model('tr', verbose = False)
    save_model(m_x, 'm_x')

    setup(data = df_train_Y, target = 'Y', silent=True, session_id=42)
    m_y = create_model('tr', verbose = False)
    save_model(m_y, 'm_y')

def main(): 

    m_x = load_model('m_x')
    m_y = load_model('m_y')

    list_x=[]
    list_y=[]

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('main_mode', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('main_mode', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
 
    while True:
 
        ret, frame = cap.read()

        img = cv2.resize(frame,(display_width,display_height))
        corners, ids, _ = aruco.detectMarkers(img, dictionary)

        if len(corners) > 0:
            d = pd.DataFrame(np.array(corners[0]).flatten().tolist()).T
            d.columns = COLUMNS[:-2]
            x = predict_model(m_x, data=d)['Label'][0]
            y = predict_model(m_y, data=d)['Label'][0]
            x = int(x)
            y = int(y)
            list_x.append(x)
            list_y.append(y)
            if len(list_x) > AVG_WINDOW:
                list_x = list_x[-AVG_WINDOW:]
                list_y = list_y[-AVG_WINDOW:]
            pyautogui.moveTo(sum(list_x)/len(list_x), sum(list_y)/len(list_y))
 
        aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
 
        cv2.imshow('main_mode', img)
 
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    args = sys.argv
    arg = args[1]
    if arg == 'marker':
        cv2.imwrite('marker.png', aruco.drawMarker(dictionary, 0, 100))
    elif arg == 'train':
        train()
    elif arg == 'main':
        main()
    else:
        print('invalid argument')
