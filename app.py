import os
from flask import Flask, request, jsonify, render_template,Response
from src.pipelines.prediction_pipeline import CustomData,PredictPipline
import cv2
import mediapipe as mp
import time



app = Flask(__name__)
cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
output_file = 'body_size_measurement.txt'

def measure_distance(landmark1, landmark2):
    # Calculate the Euclidean distance between two 2D points
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5


def calculate_measurements(results):
    measurements = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Measure distance between left shoulder and right shoulder
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        measurements['Shoulder Length'] = measure_distance(left_shoulder, right_shoulder)

        # Measure distance between left hip and right hip
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        measurements['Hip Length'] = measure_distance(left_hip, right_hip)

        # Measure distance between left shoulder and left hip
        measurements['Chest Length'] = measure_distance(left_shoulder, left_hip)

        # Measure distance between right shoulder and right hip
        measurements['Waist Length'] = measure_distance(right_shoulder, right_hip)

        # Measure distance between left shoulder and left hip
        measurements['Shoulder to Waist'] = measure_distance(left_shoulder, left_hip)

    return measurements


final_measurements = []
def generate_frames(duration=10):
    start_time = time.time()
    while time.time() - start_time < duration:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get landmarks
        results = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate measurements
            measurements = calculate_measurements(results)
            
            for i in measurements:
                final_measurements.append(measurements[i])
            # Display the measured distances on the frame
            for idx, (measurement, value) in enumerate(measurements.items()):
                cv2.putText(frame, f"{measurement}: {value:.2f} px", (10, 30 + idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




picFolder = os.path.join('static', 'pics')

app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            Gender=int(request.form.get('Gender')),
            Age =int(request.form.get('Age')),
            # shoulder =int(request.form.get('ShoulderWidth')),
            # chest =int(request.form.get('ChestWidth ')),
            # waist=int(request.form.get('Waist ')),
            # hips =int(request.form.get('Hips ')),
            #shoulder_to_waist =int(request.form.get('ShoulderToWaist '))
            shoulder=int(final_measurements[0]*100),
            chest= int(final_measurements[1]*100),
            waist=int(final_measurements[2]*100),
            hips=int(final_measurements[3]*100),
            shoulder_to_waist=int(final_measurements[4]*100),
        )

        
        final_data = data.get_data_as_data_frame()
        gender = final_data['Gender'][0]
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)
        result = pred

        imageList1 = os.listdir('static/vshapemen')
        imagelist1 = ['vshapemen/1.jpeg','vshapemen/2.jpg','vshapemen/3.jpeg','vshapemen/4.jpg']
        
        imageList6 = os.listdir('static/vshape')
        imagelist6 = ['vshape/1.png','vshape/2.png','vshape/3.jpg','vshape/4.jpg']

        imageList2 = os.listdir('static/Rectangularmen')
        imagelist2 = ['Rectangularmen/1.png','Rectangularmen/2.png','Rectangularmen/3.jpg','Rectangularmen/4.png']

        imageList7 = os.listdir('static/Rectangular')
        imagelist7 = ['Rectangular/1.jpg','Rectangular/2.jpg','Rectangular/3.jpeg','Rectangular/4.png']

        imageList3 = os.listdir('static/Hourglass')
        imagelist3 = ['Hourglass/1.png','Hourglass/2.jpg','Hourglass/3.jpg','Hourglass/4.jpeg']

        imageList8 = os.listdir('static/Hourglassmen')
        imagelist8 = ['Hourglassmen/1.jpg','Hourglassmen/2.jpeg','Hourglassmen/3.jpg','Hourglassmen/4.png']

        imageList4 = os.listdir('static/Pear')
        imagelist4 = ['Pear/1.jpg','Pear/2.jpg','Pear/3.png','Pear/4.jpg']

        imageList9 = os.listdir('static/Pearmen')
        imagelist9 = ['Pearmen/1.jpeg','Pearmen/2.png','Pearmen/3.jpg','Pearmen/4.jpg']

        imageList5 = os.listdir('static/Triangle')
        imagelist5 = ['Triangle/1.png','Triangle/2.jpg','Triangle/3.jpg','Triangle/4.png']

        imageList10 = os.listdir('static/Trianglemen')
        imagelist10 = ['Trianglemen/1.jpg','Trianglemen/2.png','Trianglemen/3.jpg','Trianglemen/4.jpg']

        if result == "V-shape" and gender == 1:
            return render_template("Results1.html",imagelist=imagelist1)
        if result == "V-shape" and gender == 2 :
            return render_template("Results1.html",imagelist=imagelist6)
        
        elif result == "Rectangular" and gender == 1  :
            return render_template("Results2.html", imagelist=imagelist2)
        elif result == "Rectangular" and gender == 2  :
            return render_template("Results2.html", imagelist=imagelist7)
        
        elif result == "Hourglass" and gender == 1  :
            return render_template("Results3.html", imagelist=imagelist8)
        elif result == "Hourglass" and gender == 2  :
            return render_template("Results3.html", imagelist=imagelist3)
        
        elif result == "Pear" and gender == 1:
            return render_template("Results4.html",imagelist=imagelist9)
        elif result == "Pear" and gender == 2:
            return render_template("Results4.html",imagelist=imagelist4)
        
        elif result == "Triangle " and gender == 1:
            return render_template("Results5.html", imagelist=imagelist10)
        elif result == "Triangle " and gender == 2:
            return render_template("Results5.html", imagelist=imagelist5)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

# http://127.0.0.1:5000 