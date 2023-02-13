
from config import config
import cv2
import numpy as np


class Alert_by_counting_frames():
    def __init__(self):
        self.ear_thesh = config['ear_thesh']
        self.mar_thesh = config['mar_thesh']
        self.ear_frame_count_thesh = config['ear_frame_count_thesh']
        self.mar_frame_count_thesh = config['mar_frame_count_thesh']
        self.phone_frame_count_thesh = config['phone_frame_count_thesh']
        self.cigarette_frame_count_thesh = config['cigarette_frame_count_thesh']
        self.number_of_func = 5

        #do not change the variables below
        self.dict_ar_lists = {
            'ear':[0]*15,
            'mar':[0]*15,
            'phone':[0]*10,
            'cigarette':[0]*10,
        }

    def put_warning_text(self, frame, text, idx):
        cv2.putText(frame,text, (frame.shape[1]*idx//self.number_of_func, frame.shape[0]*9//10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    
    def recieve_values_ear_mar(self, frame, ear, mar):
        if ear<self.ear_thesh:
            self.dict_ar_lists['ear'].append(1)
        else:
            self.dict_ar_lists['ear'].append(0)
            
        if mar>self.mar_thesh:
            self.dict_ar_lists['mar'].append(1)
        else:
            self.dict_ar_lists['mar'].append(0)
        
        self.dict_ar_lists['ear'].pop(0)
        self.dict_ar_lists['mar'].pop(0)
        
        if np.sum(self.dict_ar_lists['ear'])>self.ear_frame_count_thesh:
            self.put_warning_text(frame, "SLEEPY", 0)
            # cv2.putText(frame,f"SLEEPY", (frame.shape[1]*0//self.number_of_func, frame.shape[0]*9//10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 5)
        
        if np.sum(self.dict_ar_lists['mar'])>self.mar_frame_count_thesh:
            self.put_warning_text(frame, "YAWNING", 1)
            # cv2.putText(frame,f"YAWNING", (frame.shape[1]*1//self.number_of_func, frame.shape[0]*9//10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 5)

    
    def recieve_values_hand(self, frame, multi_hands):
        if multi_hands.multi_hand_landmarks != None:
            self.put_warning_text(frame, "HAND DETECTED", 2)
            # cv2.putText(frame,f"HAND DETECTED", (frame.shape[1]*2//self.number_of_func, frame.shape[0]*9//10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 5)
    
    def recieve_values_phone_cigarette(self, frame, xyxy_phone_cigarette):
        phone_detected, cigarette_detected = 0,0
        for xyxy in xyxy_phone_cigarette:
            cls = xyxy[-1]
            if cls == 0:
                phone_detected = 1
            else:
                cigarette_detected = 1
       
        self.dict_ar_lists['phone'].pop(0)
        self.dict_ar_lists['cigarette'].pop(0)
        self.dict_ar_lists['phone'].append(phone_detected)
        self.dict_ar_lists['cigarette'].append(cigarette_detected)
        if np.sum(self.dict_ar_lists['phone'])>self.phone_frame_count_thesh:
            self.put_warning_text(frame, "PHONE DETECTED", 2)
        if np.sum(self.dict_ar_lists['cigarette'])>self.cigarette_frame_count_thesh:
            self.put_warning_text(frame, "CIGARETTE DETECTED", 3)

import threading
class CustomThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
 
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
             
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return  