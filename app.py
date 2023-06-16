from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import mediapipe as mp
import numpy as np
import pyttsx3
from scipy.spatial import distance as dis
import cv2 as cv
import os
import time

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/static/uploads'
application.config['MAX_CONTENT_PATH'] = 1000000

def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        point_scale = ((int)(point.x * width), (int)(point.y * height))     
        cv.circle(image, point_scale, 2, color, 1)
        
def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    left_right_dis = euclidean_distance(image, left, right)
    
    aspect_ratio = left_right_dis / top_bottom_dis
    return aspect_ratio

def extract_eye_landmarks(face_landmarks, eye_landmark_indices):
    eye_landmarks = []
    for index in eye_landmark_indices:
        landmark = face_landmarks.landmark[index]
        eye_landmarks.append([landmark.x, landmark.y])
    return np.array(eye_landmarks)

def calculate_midpoint(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    midpoint = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
    return midpoint

def check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
    left_eye_midpoint = calculate_midpoint(left_eye_points)
    right_eye_midpoint = calculate_midpoint(right_eye_points)
    left_iris_midpoint = calculate_midpoint(left_iris_points)
    right_iris_midpoint = calculate_midpoint(right_iris_points)
    deviation_threshold_horizontal = 2.8
    # deviation_threshold_vertical = 1.99
    return (abs(left_iris_midpoint[0] - left_eye_midpoint[0]) <= deviation_threshold_horizontal 
            and abs(right_iris_midpoint[0] - right_eye_midpoint[0]) <= deviation_threshold_horizontal) 
            # and abs(left_iris_midpoint[1] - left_eye_midpoint[1]) <= deviation_threshold_vertical 
            # and abs(right_iris_midpoint[1] - right_eye_midpoint[1]) <= deviation_threshold_vertical)


def gen_frames():  # generate frame by frame from camera
    
    STATIC_IMAGE = False
    REFINE_LANDMARKS = True
    MAX_NO_FACES = 1
    DETECTION_CONFIDENCE = 0.6
    TRACKING_CONFIDENCE = 0.6

    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_MAGENTA = (255, 0, 255)

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]

    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]

    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]

    FACE= [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
    
    face_mesh = mp.solutions.face_mesh
    face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces=MAX_NO_FACES,
                                refine_landmarks=REFINE_LANDMARKS,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

    camera = cv.VideoCapture(0)
    speech = pyttsx3.init()

    # variabel untuk optimasi deteksi mata tertutup
    frame_count = 0
    min_frame = 15
    min_tolerance = 5.0

    # variabel untuk optimasi deteksi iris tidak fokus
    detection_start_time = None
    warning_delay = 2

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            outputs = face_model.process(image_rgb)

            if outputs.multi_face_landmarks: 
                draw_landmarks(frame, outputs, FACE, COLOR_GREEN)
                draw_landmarks(frame, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
                draw_landmarks(frame, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_RED)
                draw_landmarks(frame, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
                draw_landmarks(frame, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)
                draw_landmarks(frame, outputs, UPPER_LOWER_LIPS, COLOR_BLUE)
                draw_landmarks(frame, outputs, LEFT_RIGHT_LIPS, COLOR_BLUE)
                
                mesh_points = []    
                for p in outputs.multi_face_landmarks[0].landmark:
                    x = int(p.x * img_w)
                    y = int(p.y * img_h)
                    mesh_points.append((x, y))
                mesh_points = np.array(mesh_points)            
                
                left_eye_points = mesh_points[LEFT_EYE]
                right_eye_points = mesh_points[RIGHT_EYE]
                left_iris_points = mesh_points[LEFT_IRIS]
                right_iris_points = mesh_points[RIGHT_IRIS]

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv.circle(frame, center_left, int(l_radius), (COLOR_MAGENTA), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (COLOR_MAGENTA), 1, cv.LINE_AA)

                # deteksi mata tertutup
                ratio_left_eye = get_aspect_ratio(frame, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
                ratio_right_eye = get_aspect_ratio(frame, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                ratio = (ratio_left_eye + ratio_right_eye) / 2
                
                if ratio > min_tolerance:
                    frame_count += 1
                else:
                    frame_count = 0
                if frame_count > min_frame:
                    speech.say('Please wake up')
                    speech.runAndWait()

                # deteksi bibir terbuka (menguap)
                ratio_lips = get_aspect_ratio(frame, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
                if ratio_lips < 1.8:
                    speech.say('Please take rest')
                    speech.runAndWait()
                
                # deteksi iris tidak fokus
                if not check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
                    if detection_start_time is None:
                        detection_start_time = time.time()
                    elif time.time() - detection_start_time >= warning_delay:
                        speech.say('Please pay attention')
                        speech.runAndWait()
                        detection_start_time = None
                else:
                    detection_start_time = None

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']

        global filename
        filename = application.config['Get Started'] + '/' + secure_filename(f.filename)

        try:
            f.save(filename)
            return render_template('form.html', filename=secure_filename(f.filename), notif='Uploaded Success')
        except:
            return render_template('upload_gagal.html')
    return render_template('form.html')

@application.route('/stream', methods=['GET', 'POST'])
def stream():
    if request.method == 'POST':
        return render_template('streaming.html')
    return render_template('upload_gagal.html')

@application.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    application.run(debug=True)