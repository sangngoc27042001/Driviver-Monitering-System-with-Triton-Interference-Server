config = {
    'ear_thesh':0.33,
    'mar_thesh' : 0.8,
    'ear_frame_count_thesh' : 5,
    'mar_frame_count_thesh' : 7,
    'YOLOV5_phone_cigarette_model_weight': './weights/best_phone_cigaretteV3.pt',
    'YOLOV5_conf': 0.65,
    'phone_frame_count_thesh':3,
    'cigarette_frame_count_thesh':3
}

import numpy as np
from scipy.spatial import distance
import cv2
import mediapipe as mp
import torch
import time
import threading

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
YOLOV5_phone_cigarette_model = torch.hub.load('ultralytics/yolov5','custom', path = config['YOLOV5_phone_cigarette_model_weight'], force_reload=True)
results_for_multithreading = {}

def get_multile_face_landmarks(image):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as face_mesh:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results.multi_face_landmarks

def get_multile_hand_landmarks(image):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6) as hands:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results

def get_xyxy_phone_cigarette(frame, conf_thresh):
    frame_in = frame.copy()
    frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
    a = YOLOV5_phone_cigarette_model([frame_in])
    labels, cords = a.xyxyn[0][:,-1], a.xyxyn[0][:,:-1]
    y_shape, x_shape, _ = frame.shape
    res = []
    for info in a.xyxyn[0]:
        x1, y1, x2, y2, conf, cls = int(info[0]*x_shape), int(info[1]*y_shape), int(info[2]*x_shape), int(info[3]*y_shape), info[4], int(info[5])
        if conf>conf_thresh:
            res.append((x1, y1, x2, y2, cls))
    return res

def draw_xyxy_phone_cigarette(frame, cords):
    for cord in cords:
        x1, y1, x2, y2, cls = cord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 5)

def draw_hand_landmark(image, hand_landmarks):
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

def draw_contours_frame(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_iris_connections_style())



def get_xy_headpose(face_landmarks):
    def preprocess_face_3d(face_3d):
        b1 = (face_3d[1]+face_3d[2])/2
        a1 = np.array([face_3d[0][0],b1[1],face_3d[0][2]])
        c1 = np.array([b1[0],b1[1],face_3d[0][2]])
        b2 = (face_3d[3]+face_3d[4])/2
        a2 = np.array([b2[0], face_3d[0][1],face_3d[0][2]])
        c2 = np.array([b2[0], b2[1],face_3d[0][2]])

        return a1, b1, c1, a2, b2, c2
    def cal_angle(ABC_coors):
        a, b, c = ABC_coors[0], ABC_coors[1], ABC_coors[2]
        ba = a - b
        bc = c - b
        cosine_angle = ba.dot(bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        if ba[0]<0 or ba[1]>0:
            angle = -angle
        return angle
    arrayPoint = np.array([[point.x, point.y, point.z] for point in face_landmarks.landmark])
    face_3d = arrayPoint[[1,129,358,4,2]].astype(np.float64)
    a1, b1, c1, a2, b2, c2 = preprocess_face_3d(face_3d)
    angle_y = cal_angle((a1, b1, c1))
    angle_x = cal_angle((a2, b2, c2))
    
    return angle_x, angle_y, face_3d
    
def draw_xy_headpose(frame, angle_x, angle_y, face_3d):
    img_h, img_w, img_c = frame.shape    
    # cv2.putText(frame,f"XV2:{angle_x}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # cv2.putText(frame,f"YV2:{angle_y}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    nose_2d = (int(face_3d[0][0]*img_w), int(face_3d[0][1]*img_h))

    p2 = (int(nose_2d[0] + angle_y*10), int(nose_2d[1] - angle_x*10))
    cv2.line(frame, nose_2d, p2, (255, 0, 0), 3)

    

def head_pose_handler(frame, face_landmarks):
    img_h, img_w, img_c = frame.shape
    arrayPoint = np.array([[point.x*img_w, point.y*img_h, point.z] for point in face_landmarks.landmark])
    face_3d = arrayPoint[[1,33,61,199,263,291]].astype(np.float64)
    face_2d = face_3d[:,:2].astype(np.float64)
        
    focal_length = 1*img_w
    
    cam_matrix = np.array([[focal_length, 0, img_h/2],
                          [0, focal_length, img_w/2],
                          [0, 0, 1]])
    
    dist_matrix = np.zeros((4,1))
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    x, y, z = angles[0]*360, angles[1]*360, angles[2]*360 
    
    
    
    cv2.putText(frame,f"X:{x}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame,f"Y:{y}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    nose_2d = (int(arrayPoint[1][0]), int(arrayPoint[1][1]))
    p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x*10 +50))
    cv2.line(frame, nose_2d, p2, (255, 0, 0), 3)
    
def return_variables(face_landmarks):
    arrayPoint = np.array([[point.x, point.y, point.z] for point in face_landmarks.landmark])
    def cal_aspect_ratio(four_points):
        A = distance.euclidean(four_points[1], four_points[3])
        C = distance.euclidean(four_points[0], four_points[2])
        ear = (A) / (C)
        return ear
    def cal_eye_aspect_ratio(arrayPoint):
        ear_1 = cal_aspect_ratio(arrayPoint[[33, 159, 133, 145]])
        ear_2 = cal_aspect_ratio(arrayPoint[[362, 386, 263, 374]])
        return (ear_1+ear_2)/2
    def cal_mouth_aspect_ratio(arrayPoint):
        return cal_aspect_ratio(arrayPoint[[78, 13, 308, 14]])
    return cal_eye_aspect_ratio(arrayPoint), cal_mouth_aspect_ratio(arrayPoint)
    
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
            self.put_warning_text(frame, "PHONE DETECTED", 3)
        if np.sum(self.dict_ar_lists['cigarette'])>self.cigarette_frame_count_thesh:
            self.put_warning_text(frame, "CIGARETTE DETECTED", 4)
        

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


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter('out6.avi', fourcc, 30, (640, 480*2))

import cv2
import mediapipe as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
args = parser.parse_args()

alert = Alert_by_counting_frames()
cap = cv2.VideoCapture(args.input)

i = 0
while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (640, 480))
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    start = time.time()    
    image_org = cv2.flip(image.copy(), 1)
    image = cv2.flip(image, 1)
    
    #define process 
    process_1 = CustomThread(target=get_multile_face_landmarks, args=(image,))
    process_2 = CustomThread(target=get_multile_hand_landmarks, args=(image,))
    process_3 = CustomThread(target=get_xyxy_phone_cigarette, args=(image, config['YOLOV5_conf'],))

    process_1.start()
    process_2.start()
    process_3.start()

    multi_face_landmarks = process_1.join()
    multi_hands = process_2.join()
    xyxy_phone_cigarette = process_3.join()

    # #get infromation from fram
    # multi_face_landmarks = get_multile_face_landmarks(image)
    # multi_hands = get_multile_hand_landmarks(image)
    # xyxy_phone_cigarette = get_xyxy_phone_cigarette(image, config['YOLOV5_conf'])

    alert.recieve_values_phone_cigarette(image, xyxy_phone_cigarette)

    if multi_face_landmarks!= None:
        for face_landmarks in multi_face_landmarks:
            #get the headpose of face
            angle_x, angle_y, face_3d = get_xy_headpose(face_landmarks)
            
            #draw head pose and contour
            draw_xy_headpose(image, angle_x, angle_y, face_3d)
            draw_contours_frame(image, face_landmarks)
            draw_xyxy_phone_cigarette(image, xyxy_phone_cigarette)
            
            ear, mar = return_variables(face_landmarks)
            alert.recieve_values_ear_mar(image, ear, mar)
            
            #draw EAR, MAR
            # cv2.putText(image,f"EAR: {ear}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # cv2.putText(image,f"MAR: {mar}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if multi_hands.multi_hand_landmarks!= None:
        for hand_landmarks in multi_hands.multi_hand_landmarks:
            draw_hand_landmark(image, hand_landmarks)
            alert.recieve_values_hand(image, multi_hands)

    cv2.putText(image,f"FPS: {round(1/(time.time()-start),2)}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)                
    cv2.imshow('MediaPipe Face Mesh', np.concatenate([image, image_org]))
    writer.write( np.concatenate([image, image_org]))
    if cv2.waitKey(5) & 0xFF == 27:
        break
    print(f"frame: {i} FPS: {round(1/(time.time()-start),2)}")
    i=i+1
writer.release()
cap.release()
cv2.destroyAllWindows()