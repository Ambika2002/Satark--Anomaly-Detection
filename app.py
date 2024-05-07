from flask import Flask, render_template, request, Response, redirect
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from pytz import timezone
import os
import pafy
import math
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from moviepy.editor import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
# %matplotlib inline
from playsound import playsound  # Library to play sounds


app = Flask(__name__, static_folder='assets', template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notifications.db'  # Use SQLite for simplicity
db = SQLAlchemy(app)

# Notification model to store records
class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sent_time = db.Column(db.DateTime, default=datetime.utcnow)
    content = db.Column(db.Text)

# Load the trained models
accident_model = tf.keras.models.load_model('accidents.h5')  # Replace with the actual path to your saved accident detection model
fighting_model = tf.keras.models.load_model('voilence_model.h5')  # Replace with the actual path to your saved fighting detection model
# LRCN_model = load_model('LRCN_model.h5') 

# Set the image dimensions (assuming the same as used during training)
img_height = 250
img_width = 250

# Define class names (replace with your actual class names)
accident_class_names = ["Non Accident", "Accident"]
fighting_class_names = ["NonViolence", "Violence"]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16

receiver_email = 'hjhhjhj.com'  # Receiver email address

# Assuming your server is running in a different timezone (e.g., UTC)
server_timezone = timezone('UTC')
indian_timezone = timezone('Asia/Kolkata')

alert_sound = "alarm.mp3"

# Disable the favicon route
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route("/<a>")
def k1(a):
    t = a + ".html"
    return render_template(t)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    video_path = request.form['video_path']
    detection_option = request.form['detection_option']

    if detection_option == 'accident':
        return Response(process_accident_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif detection_option == 'violence':
        return Response(predict_violence(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid detection option"
    # process_fighting_video(video)

def process_accident_video(video_path):
    cap = cv2.VideoCapture(video_path)

    overall_accident_score = 0
    frame_count = 0
    notification_message = None

    with app.app_context():  # Ensure database access within Flask application context
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Preprocess the frame for accident model prediction
            img_accident = cv2.resize(frame, (img_height, img_width))
            img_array_accident = image.img_to_array(img_accident)
            img_array_accident = np.expand_dims(img_array_accident, axis=0)
            img_array_accident /= 255.0  # Normalize pixel values

            # Make predictions for accident detection
            predictions_accident = accident_model.predict(img_array_accident)
            accident_score = predictions_accident[0, accident_class_names.index('Accident')]
            pred_label = accident_class_names[np.argmax(predictions_accident[0])]
            cv2.putText(frame, f'Prediction: {pred_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Accumulate scores for each frame
            overall_accident_score += accident_score
            frame_count += 1

            # Check for accident and send notification
            if frame_count > 8:  # Send notification after processing 10 frames
                avg_accident_score = overall_accident_score / frame_count

                if avg_accident_score > 0.5:
                    playsound(alert_sound)
                    notification_message = 'Notification sent: Alert !!  Accident  detected'
                    print(notification_message)  # Print notification message to terminal
                    save_notification_record('Accident Detected', video_path)
                    send_notification(receiver_email, 'Alert !! Urgent !!', 'Accident Detected',
                                      'An accident has been detected in the video. Immediate attention is required.'
                                      'Please take the necessary actions to ensure safety. \n\n'
                                      'Video: ' + video_path)
                    print("Email sent ")
                    # Send email immediately upon detection
                    break

            # Encode the frame as JPEG and yield it
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    # Release the video capture object
    cap.release()

    # Return a message if no accident is detected
    if notification_message:
        return notification_message
    else:
        return 'No accident detected'




def predict_violence(video_file_path):
    with app.app_context():  # Ensure access to the Flask application context
        video_reader = cv2.VideoCapture(video_file_path)
        if not video_reader.isOpened():
            print("Error: Could not open video.")
            return

        frame_sequence = []

        while True:
            ret, frame = video_reader.read()
            if not ret:
                break

            # Preprocess the frame
            preprocessed_frame = preprocess_frame(frame)
            frame_sequence.append(preprocessed_frame)

            # If the sequence length is reached, predict violence
            if len(frame_sequence) == SEQUENCE_LENGTH:
                # Convert the sequence into a numpy array
                sequence_array = np.array([frame_sequence])

                # Predict violence for the sequence
                violence_score = fighting_model.predict(sequence_array)[0][1]
                violence_label = 'Violence' if violence_score > 0.8 else 'NonViolence'

                # Display the frame with predicted label
                cv2.putText(frame, f'Violence: {violence_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, encoded_image = cv2.imencode('.jpg', frame)
                frame_data = encoded_image.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

                # Remove the oldest frame from the sequence
                frame_sequence.pop(0)

                # Check for violence and send notification
                if violence_label == "Violence":
                    playsound(alert_sound)
                    notification_message = 'Notification sent: Alert !!  Violence  detected'
                    print(notification_message)  # Print notification message to terminal
                    save_notification_record('Violence Detected', video_file_path)
                    send_notification(receiver_email, 'Alert !! Urgent !!', 'Violence Detected',
                                      'Violence has been detected in the video. Immediate attention is required.'
                                      'Please take the necessary actions to ensure safety. \n\n'
                                      'Video: ' + video_file_path)
                    print("Email sent ")
                    # Send email immediately upon detection
                    

        video_reader.release()
        cv2.destroyAllWindows()

# # Function to process accident video and play alert if detected
# def process_accident_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     accident_detected = False

#     with app.app_context():  # Ensure database access within Flask application context
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Preprocess the frame for accident detection
#             img_accident = cv2.resize(frame, (img_height, img_width))
#             img_array_accident = image.img_to_array(img_accident)
#             img_array_accident = np.expand_dims(img_array_accident, axis=0)
#             img_array_accident /= 255.0  # Normalize pixel values

#             # Predict accident
#             predictions_accident = accident_model.predict(img_array_accident)
#             accident_score = predictions_accident[0, accident_class_names.index('Accident')]
#             pred_label = accident_class_names[np.argmax(predictions_accident[0])]

#             # Display prediction on the frame
#             cv2.putText(frame, f'Prediction: {pred_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # If accident detected, play alert and send notification
#             if pred_label == "Accident" and accident_score > 0.5 and not accident_detected:
#                 playsound(alert_sound)
#                 save_notification_record('Accident Detected', video_path)
#                 send_notification(
#                     receiver_email,
#                     'Alert !! Accident Detected',
#                     'Accident detected in the video.',
#                     'An accident has been detected. Please take appropriate action.'
#                 )
#                 accident_detected = True  # Mark as detected

#             # Encode and yield the frame
#             (flag, encodedImage) = cv2.imencode(".jpg", frame)
#             if not flag:
#                 continue
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

#     # Release resources
#     cap.release()

# # Function to process violence detection
# def predict_violence(video_path):
#     video_reader = cv2.VideoCapture(video_path)
#     frame_sequence = []

#     if not video_reader.isOpened():
#         return

#     while True:
#         ret, frame = video_reader.read()
#         if not ret:
#             break

#         # Preprocess the frame
#         preprocessed_frame = preprocess_frame(frame)
#         frame_sequence.append(preprocessed_frame)

#         # If sequence length is reached, predict violence
#         if len(frame_sequence) == SEQUENCE_LENGTH:
#             sequence_array = np.array([frame_sequence])
#             predictions = fighting_model.predict(sequence_array)
#             violence_score = predictions[0][fighting_class_names.index("Violence")]
#             violence_label = "Violence" if violence_score > 0.8 else "NonViolence"

#             # Display prediction on the frame
#             cv2.putText(frame, f'Violence: {violence_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#             # Play alert and send notification if violence detected
#             if violence_label == "Violence":
#                 playsound(alert_sound)
#                 save_notification_record('Violence Detected', video_path)
#                 send_notification(
#                     receiver_email,
#                     'Alert !! Violence Detected',
#                     'Violence detected in the video.',
#                     'Violence has been detected. Please take appropriate action.'
#                 )

#             # Encode and yield the frame
#             _, encoded_image = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')

#             # Remove the oldest frame from the sequence
#             frame_sequence.pop(0)

#     video_reader.release()



def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return resized_frame / 255.0


def predict_violence_score(frame):
    # Predict violence score for a single frame
    return fighting_model.predict(np.expand_dims(frame, axis=0))[0][1]


###########################################################################################################
def save_notification_record(content, additional_info):
    # Save record to the database
    ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
    notification = Notification(content=f'{content} - {additional_info}', sent_time=ist_time)
    db.session.add(notification)
    db.session.commit()

def send_notification(receiver_email, subject, body, additional_info):
    # Replace these with your Gmail credentials
    sender_email = 'aokokoko@gmail.com'  # Your Gmail email address
    sender_password ='jhjghjghgfhfhfhfhfcgh'  # Your Gmail password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'🚨 {subject}'  # Adding an alert symbol to the subject

    # Structured email content
    email_content = f'''
    <html>
        <body>
            <p>Dear User,</p>
            <p>🚨 {body}</p>  
            <p>{additional_info}</p>
            <p>Stay Safe,</p>
            <p>Satark</p>
        </body>
    </html>
    '''

    msg.attach(MIMEText(email_content, 'html'))

    # Save the current time in IST
    ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
    
    # Save record to the database with IST time
    save_notification_record('Email Sent', ist_time)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)

    text = msg.as_string()
    server.sendmail(sender_email, receiver_email, text)

    server.quit()

def display_sweetalert(message):
    script = f'''
        Swal.fire({{
            title: 'Status',
            text: '{message}',
            icon: 'warning',
            confirmButtonText: 'OK',
            customClass: {{
                title: 'sweet-alert-title',
                content: 'sweet-alert-content',
                confirmButton: 'sweet-alert-confirm-button'
            }},
            showCancelButton: false,
            allowOutsideClick: false
        }});
    '''
    return script

# Notification history route
@app.route('/notification_history')
def notification_history():
    # Fetch all notification records from the database
    notifications = Notification.query.all()

    return render_template('notification_history.html', notifications=notifications)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables
     

    app.run(debug=True)
























# def preprocess_frame(frame):
#     resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#     return resized_frame / 255.0


# def predict_violence(video_file_path):
#     with app.app_context():  # Ensure access to the Flask application context
#         video_reader = cv2.VideoCapture(video_file_path)
#         if not video_reader.isOpened():
#             print("Error: Could not open video.")
#             return

#         frame_sequence = []

#         while True:
#             ret, frame = video_reader.read()
#             if not ret:
#                 break

#             # Preprocess the frame
#             preprocessed_frame = preprocess_frame(frame)
#             frame_sequence.append(preprocessed_frame)

#             # If the sequence length is reached, predict violence
#             if len(frame_sequence) == SEQUENCE_LENGTH:
#                 # Convert the sequence into a numpy array
#                 sequence_array = np.array(frame_sequence)

#                 # Predict violence for the sequence
#                 violence_score = fighting_model.predict(np.expand_dims(sequence_array, axis=0))[0]
#                 violence_label = fighting_class_names[np.argmax(violence_score)]

#                 # Display the frame with predicted label
#                 cv2.putText(frame, f'Violence: {violence_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 _, encoded_image = cv2.imencode('.jpg', frame)
#                 frame_data = encoded_image.tobytes()
#                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

#                 # Remove the oldest frame from the sequence
#                 frame_sequence.pop(0)

#                 # Check for violence and send notification
#                 if violence_label == "Violence":
#                     notification_message = 'Notification sent: Alert !!  Voilence  detected'
#                     print(notification_message)  # Print notification message to terminal
#                     save_notification_record('Voilence Detected', video_file_path)
#                     send_notification(receiver_email, 'Alert !! Urgent !!', 'Voilence Detected',
#                                       'Voilence has been detected in the video. Immediate attention is required.'
#                                       'Please take the necessary actions to ensure safety. \n\n'
#                                       'Video: ' + video_file_path)
#                     print("Email sent ") 
                    
#         video_reader.release()
#         cv2.destroyAllWindows()

# from flask import Flask, render_template, request, Response, redirect
# import cv2
# import numpy as np
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage
# from email.mime.base import MIMEBase
# from email import encoders
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# from tensorflow import keras
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime, timedelta
# from pytz import timezone
# import os
# import pafy
# import math
# import random
# import tensorflow as tf
# from collections import deque
# import matplotlib.pyplot as plt
# from moviepy.editor import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.models import load_model
# # %matplotlib inline

# app = Flask(__name__, static_folder='assets', template_folder='templates')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notifications.db'  # Use SQLite for simplicity
# db = SQLAlchemy(app)

# # Notification model to store records
# class Notification(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     sent_time = db.Column(db.DateTime, default=datetime.utcnow)
#     content = db.Column(db.Text)

# # Load the trained models
# accident_model = tf.keras.models.load_model('accidents.h5')  # Replace with the actual path to your saved accident detection model
# fighting_model = tf.keras.models.load_model('convlstm_model_.h5')  # Replace with the actual path to your saved fighting detection model
# LRCN_model = load_model('LRCN_model.h5') 

# # Set the image dimensions (assuming the same as used during training)
# img_height = 250
# img_width = 250

# # Define class names (replace with your actual class names)
# accident_class_names = ["Non Accident", "Accident"]
# fighting_class_names = ["Abuse", "Fighting"]

# receiver_email = 'ambika.sanap@vit.edu.in'  # Receiver email address

# # Assuming your server is running in a different timezone (e.g., UTC)
# server_timezone = timezone('UTC')
# indian_timezone = timezone('Asia/Kolkata')

# # Disable the favicon route
# @app.route('/favicon.ico')
# def favicon():
#     return app.send_static_file('favicon.ico')

# @app.route("/<a>")
# def k1(a):
#     t = a + ".html"
#     return render_template(t)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     video_path = request.form['video_path']
#     detection_option = request.form['detection_option']

#     if detection_option == 'accident':
#         return Response(process_accident_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     elif detection_option == 'violence':
#         return Response(predict_on_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return "Invalid detection option"
#     # process_fighting_video(video)

# def process_accident_video(video_path):
#     cap = cv2.VideoCapture(video_path)

#     overall_accident_score = 0
#     frame_count = 0
#     notification_message = None

#     with app.app_context():  # Ensure database access within Flask application context
#         while cap.isOpened():
#             ret, frame = cap.read()

#             if not ret:
#                 break

#             # Preprocess the frame for accident model prediction
#             img_accident = cv2.resize(frame, (img_height, img_width))
#             img_array_accident = image.img_to_array(img_accident)
#             img_array_accident = np.expand_dims(img_array_accident, axis=0)
#             img_array_accident /= 255.0  # Normalize pixel values

#             # Make predictions for accident detection
#             predictions_accident = accident_model.predict(img_array_accident)
#             accident_score = predictions_accident[0, accident_class_names.index('Accident')]
#             pred_label = accident_class_names[np.argmax(predictions_accident[0])]
#             cv2.putText(frame, f'Prediction: {pred_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Accumulate scores for each frame
#             overall_accident_score += accident_score
#             frame_count += 1

#             # Check for accident and send notification
#             if frame_count > 10:  # Send notification after processing 10 frames
#                 avg_accident_score = overall_accident_score / frame_count

#                 if avg_accident_score > 0.5:
#                     notification_message = 'Notification sent: Alert !!  Accident  detected'
#                     save_notification_record('Accident Detected', video_path)
#                     send_notification(receiver_email, 'Alert !! Urgent !!', 'Accident Detected',
#                                       'An accident has been detected in the video. Immediate attention is required.'
#                                       'Please take the necessary actions to ensure safety. \n\n'
#                                       'Video: ' + video_path)
#                     break

#             # Encode the frame as JPEG and yield it
#             (flag, encodedImage) = cv2.imencode(".jpg", frame)
#             if not flag:
#                 continue
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

#     # Release the video capture object
#     cap.release()

#     # Return a message if no accident is detected
#     if notification_message:
#         return notification_message
#     else:
#         return 'No accident detected'


# #################################################################################

# # Make the Output directory if it does not exist
# test_videos_directory = 'videos'
# os.makedirs(test_videos_directory, exist_ok = True)


# video_title = "Fighting"
# SEQUENCE_LENGTH = 20
# IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# # Specify the number of frames of a video that will be fed to the model as one sequence.
# SEQUENCE_LENGTH = 20

# # Specify the directory containing the UCF50 dataset. 
# DATASET_DIR = "dataset"

# CLASSES_LIST = ["Abuse","Fighting"]

# output_video_file_path = f'{test_videos_directory}/1-Out-SeqLen{SEQUENCE_LENGTH}.mp4'


# # Display the output video.
# def predict_on_video(video_file_path, SEQUENCE_LENGTH=20):
#     '''
#     This function will perform action recognition on a video using the LRCN model.
#     Args:
#     video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
#     output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
#     SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
#     '''

#     # Initialize the VideoCapture object to read from the video file.
#     video_reader = cv2.VideoCapture(video_file_path)
#     overall_abuse_score = 0
#     frame_count = 0
#     notification_message = None

#     # Get the width and height of the video.
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))


#     # Declare a queue to store video frames.
#     frames_queue = deque(maxlen = SEQUENCE_LENGTH)

#     # Initialize a variable to store the predicted action being performed in the video.
#     predicted_class_name = ''

#     # Iterate until the video is accessed successfully.
#     while video_reader.isOpened():

#         # Read the frame.
#         ok, frame = video_reader.read() 
        
#         # Check if frame is not read properly then break the loop.
#         if not ok:
#             break

#         # Resize the Frame to fixed Dimensions.
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
#         normalized_frame = resized_frame / 255

#         # Appending the pre-processed frame into the frames list.
#         frames_queue.append(normalized_frame)

#         # Check if the number of frames in the queue are equal to the fixed sequence length.
#         if len(frames_queue) == SEQUENCE_LENGTH:

#             # Pass the normalized frames to the model and get the predicted probabilities.
#             predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]
#             print(predicted_labels_probabilities)
#             # Get the index of class with highest probability.
#             predicted_label = np.argmax(predicted_labels_probabilities)
#             # print(predicted_label)
#             # Get the class name using the retrieved index.
#             predicted_class_name = CLASSES_LIST[predicted_label]
#             print(predicted_class_name)

#         # Write predicted class name on top of the frame.
#         cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Write The frame into the disk using the VideoWriter Object.
#         # video_writer.write(frame)
        
#     # Release the VideoCapture and VideoWriter objects.
#         (flag, encodedImage) = cv2.imencode(".jpg", frame)
#         if not flag:
#             continue
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
#     video_reader.release()
#     # video_writer.release()

# ###########################################################################################################
# def save_notification_record(content, additional_info):
#     # Save record to the database
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
#     notification = Notification(content=f'{content} - {additional_info}', sent_time=ist_time)
#     db.session.add(notification)
#     db.session.commit()

# def send_notification(receiver_email, subject, body, additional_info):
#     # Replace these with your Gmail credentials
#     sender_email = 'aksanap2002@gmail.com'  # Your Gmail email address
#     sender_password = 'jinq bdcg aolb xeik'  # Your Gmail password

#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = f'🚨 {subject}'  # Adding an alert symbol to the subject

#     # Structured email content
#     email_content = f'''
#     <html>
#         <body>
#             <p>Dear User,</p>
#             <p>🚨 {body}</p>  
#             <p>{additional_info}</p>
#             <p>Stay Safe,</p>
#             <p>Satark</p>
#         </body>
#     </html>
#     '''

#     msg.attach(MIMEText(email_content, 'html'))

#     # Save the current time in IST
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
    
#     # Save record to the database with IST time
#     save_notification_record('Email Sent', ist_time)

#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     server.login(sender_email, sender_password)

#     text = msg.as_string()
#     server.sendmail(sender_email, receiver_email, text)

#     server.quit()

# def display_sweetalert(message):
#     script = f'''
#         Swal.fire({{
#             title: 'Status',
#             text: '{message}',
#             icon: 'warning',
#             confirmButtonText: 'OK',
#             customClass: {{
#                 title: 'sweet-alert-title',
#                 content: 'sweet-alert-content',
#                 confirmButton: 'sweet-alert-confirm-button'
#             }},
#             showCancelButton: false,
#             allowOutsideClick: false
#         }});
#     '''
#     return script

# # Notification history route
# @app.route('/notification_history')
# def notification_history():
#     # Fetch all notification records from the database
#     notifications = Notification.query.all()

#     return render_template('notification_history.html', notifications=notifications)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()  # Create the database tables

#     app.run(debug=True)

#############################################################################################################################333333333333333333333333333333###################


















    # from flask import Flask, render_template, request
# import cv2
# import numpy as np
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage
# from email.mime.base import MIMEBase
# from email import encoders
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# from tensorflow import keras
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime, timedelta
# from pytz import timezone

# app = Flask(__name__, static_folder='assets', template_folder='templates')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notifications.db'  # Use SQLite for simplicity
# db = SQLAlchemy(app)

# # Notification model to store records
# class Notification(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     sent_time = db.Column(db.DateTime, default=datetime.utcnow)
#     content = db.Column(db.Text)

# # Load the trained models
# accident_model = tf.keras.models.load_model('accidents.h5')  # Replace with the actual path to your saved accident detection model
# fighting_model = tf.keras.models.load_model('convlstm_model_.h5')  # Replace with the actual path to your saved fighting detection model

# # Set the image dimensions (assuming the same as used during training)
# img_height = 250
# img_width = 250

# # Define class names (replace with your actual class names)
# accident_class_names = ["Non Accident", "Accident"]
# fighting_class_names = ["Abuse", "Fighting"]

# receiver_email = 'ambika.sanap@vit.edu.in'  # Receiver email address

# # Assuming your server is running in a different timezone (e.g., UTC)
# server_timezone = timezone('UTC')
# indian_timezone = timezone('Asia/Kolkata')

# @app.route("/<a>")
# def k1(a):
#     t = a + ".html"
#     return render_template(t)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     video_path = request.form['video_path']
#     detection_option = request.form['detection_option']

#     if detection_option == 'accident':
#         # Process the video for accident detection
#         notify = process_accident_video(video_path)
#     elif detection_option == 'violence':
#         # Process the video for fighting detection
#         notify = process_fighting_video(video_path)
#     else:
#         return "Invalid detection option"

#     # Display SweetAlert notification
#     display_sweetalert(notify)

#     return render_template('records.html', message=notify)

# def process_accident_video(video_path):
#     cap = cv2.VideoCapture(video_path)

#     overall_accident_score = 0
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break

#         # Preprocess the frame for accident model prediction
#         img_accident = cv2.resize(frame, (img_height, img_width))
#         img_array_accident = image.img_to_array(img_accident)
#         img_array_accident = np.expand_dims(img_array_accident, axis=0)
#         img_array_accident /= 255.0  # Normalize pixel values

#         # Make predictions for accident detection
#         predictions_accident = accident_model.predict(img_array_accident)
#         accident_score = predictions_accident[0, accident_class_names.index('Accident')]

#         # Accumulate scores for each frame
#         overall_accident_score += accident_score
#         frame_count += 1

#         # Check for accident and send notification
#         if frame_count > 8:  # Send notification after processing 10 frames
#             avg_accident_score = overall_accident_score / frame_count

#             if avg_accident_score > 0.5:
#                 # Output notification to console
#                 print('Notification: Alert !!  Accident  detected')

#                 # Save record to the database
#                 save_notification_record('Accident Detected', video_path)

#                 # Send email notification with structured content
#                 send_notification(receiver_email, 'Alert !! Urgent !!', 'Accident Detected',
#                                   'An accident has been detected in the video. Immediate attention is required.'
#                                   'Please take the necessary actions to ensure safety. \n\n'
#                                   'Video: ' + video_path)  # Include video path in the email
#                 cap.release()
#                 return 'Notification sent: Alert !!  Accident  detected'

#     cap.release()
#     return 'No accident detected'

# def frames_extraction(video_path, sequence_length=20, image_height=64, image_width=64):
#     '''
#     This function will extract the required frames from a video after resizing and normalizing them.
#     Args:
#         video_path: The path of the video in the disk, whose frames are to be extracted.
#         sequence_length: The number of frames of a video that will be fed to the model as one sequence.
#         image_height: Height to which each video frame will be resized.
#         image_width: Width to which each video frame will be resized.
#     Returns:
#         frames_list: A list containing the resized and normalized frames of the video.
#     '''

#     # Declare a list to store video frames.
#     frames_list = []

#     # Read the Video File using the VideoCapture object.
#     video_reader = cv2.VideoCapture(video_path)

#     # Get the total number of frames in the video.
#     video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Calculate the interval after which frames will be added to the list.
#     skip_frames_window = max(int(video_frames_count / sequence_length), 1)

#     # Iterate through the Video Frames.
#     for frame_counter in range(sequence_length):

#         # Set the current frame position of the video.
#         video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

#         # Reading the frame from the video.
#         success, frame = video_reader.read()

#         # Check if Video frame is not successfully read then break the loop
#         if not success:
#             break

#         # Resize the Frame to fixed height and width.
#         resized_frame = cv2.resize(frame, (image_height, image_width))

#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
#         normalized_frame = resized_frame / 255

#         # Append the normalized frame into the frames list
#         frames_list.append(normalized_frame)

#     # Release the VideoCapture object.
#     video_reader.release()

#     # Return the frames list.
#     return frames_list

# def process_fighting_video(video_path):
#     # Extract frames from the video
#     frames = frames_extraction(video_path, sequence_length=20, image_height=64, image_width=64)

#     # Ensure that the number of frames matches the SEQUENCE_LENGTH
#     if len(frames) != 20:
#         return 'Skipping video: {}. Insufficient frames.'.format(video_path)

#     # Convert the frames to numpy array and add batch dimension
#     frames = np.expand_dims(frames, axis=0)

#     # Make predictions
#     predictions = fighting_model.predict(frames)

#     # Get the predicted class index
#     predicted_class_index = np.argmax(predictions)

#     # Map the predicted class index to the actual class name
#     predicted_class_name = fighting_class_names[predicted_class_index]

#     # Check for fighting or abuse and send notification
#     if predicted_class_name in ['Fighting', 'Abuse']:
#         # Output notification to console
#         print(f'Notification: Alert !! {predicted_class_name} detected')

#         # Save record to the database
#         save_notification_record(f'{predicted_class_name} Detected', video_path)

#         # Send email notification with structured content
#         send_notification(receiver_email, 'Alert !! Urgent !!', f'{predicted_class_name} Detected',
#                           f'{predicted_class_name} has been detected in the video. Immediate attention is required.'
#                           'Please take the necessary actions to ensure safety. \n\n'
#                           'Video: ' + video_path)  # Include video path in the email

#         return f'Notification sent: Alert !! {predicted_class_name} detected'
#     else:
#         return 'No fighting or abuse detected'


# def save_notification_record(content, additional_info):
#     # Save record to the database
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
#     notification = Notification(content=f'{content} - {additional_info}', sent_time=ist_time)
#     db.session.add(notification)
#     db.session.commit()

# def send_notification(receiver_email, subject, body, additional_info):
#     # Replace these with your Gmail credentials
#     sender_email = 'aksanap2002@gmail.com'  # Your Gmail email address
#     sender_password = 'jinq bdcg aolb xeik'  # Your Gmail password

#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = f'🚨 {subject}'  # Adding an alert symbol to the subject

#     # Structured email content
#     email_content = f'''
#     <html>
#         <body>
#             <p>Dear User,</p>
#             <p>🚨 {body}</p>  
#             <p>{additional_info}</p>
#             <p>Stay Safe,</p>
#             <p>Satark</p>
#         </body>
#     </html>
#     '''

#     msg.attach(MIMEText(email_content, 'html'))

#     # Save the current time in IST
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
    
#     # Save record to the database with IST time
#     save_notification_record('Email Sent', ist_time)

#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     server.login(sender_email, sender_password)

#     text = msg.as_string()
#     server.sendmail(sender_email, receiver_email, text)

#     server.quit()

# def display_sweetalert(message):
#     script = f'''
#         Swal.fire({{
#             title: 'Status',
#             text: '{message}',
#             icon: 'warning',
#             confirmButtonText: 'OK',
#             customClass: {{
#                 title: 'sweet-alert-title',
#                 content: 'sweet-alert-content',
#                 confirmButton: 'sweet-alert-confirm-button'
#             }},
#             showCancelButton: false,
#             allowOutsideClick: false
#         }});
#     '''
#     return script

# # Notification history route
# @app.route('/notification_history')
# def notification_history():
#     # Fetch all notification records from the database
#     notifications = Notification.query.all()

#     return render_template('notification_history.html', notifications=notifications)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()  # Create the database tables

#     app.run(debug=True)







##################################################################################################################################################################################3
# from flask import Flask, render_template, request, Response, redirect, jsonify
# import cv2
# import numpy as np
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage
# from email.mime.base import MIMEBase
# from email import encoders
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime, timedelta
# from pytz import timezone
# from threading import Thread

# app = Flask(__name__, static_folder='assets', template_folder='templates')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notifications.db'
# db = SQLAlchemy(app)

# class Notification(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     sent_time = db.Column(db.DateTime, default=datetime.utcnow)
#     content = db.Column(db.Text)

# accident_model = tf.keras.models.load_model('accidents.h5')
# fighting_model = tf.keras.models.load_model('convlstm_model_.h5')
# img_height = 250
# img_width = 250
# accident_class_names = ["Non Accident", "Accident"]
# fighting_class_names = ["Abuse", "Fighting"]
# receiver_email = 'ambika.sanap@vit.edu.in'
# server_timezone = timezone('UTC')
# indian_timezone = timezone('Asia/Kolkata')

# @app.route('/favicon.ico')
# def favicon():
#     return app.send_static_file('favicon.ico')

# @app.route("/<a>")
# def k1(a):
#     t = a + ".html"
#     return render_template(t)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     video_path = request.form['video_path']
#     detection_option = request.form['detection_option']

#     if detection_option == 'accident':
#         return Response(process_accident_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     elif detection_option == 'violence':
#         return Response(process_fighting_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return "Invalid detection option"

# def process_accident_video(video_path):
#     cap = cv2.VideoCapture(video_path)

#     overall_accident_score = 0
#     frame_count = 0
#     notification_message = None

#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break

#         img_accident = cv2.resize(frame, (img_height, img_width))
#         img_array_accident = image.img_to_array(img_accident)
#         img_array_accident = np.expand_dims(img_array_accident, axis=0)
#         img_array_accident /= 255.0

#         predictions_accident = accident_model.predict(img_array_accident)
#         accident_score = predictions_accident[0, accident_class_names.index('Accident')]

#         overall_accident_score += accident_score
#         frame_count += 1

#         if frame_count > 10:
#             avg_accident_score = overall_accident_score / frame_count

#             if avg_accident_score > 0.5:
#                 notification_message = 'Notification sent: Alert !!  Accident detected'
#                 save_notification_record('Accident Detected', video_path)
#                 send_notification(receiver_email, 'Alert !! Urgent !!', 'Accident Detected',
#                                   'An accident has been detected in the video. Immediate attention is required.'
#                                   'Please take the necessary actions to ensure safety. \n\n'
#                                   'Video: ' + video_path)
#                 Thread(target=display_sweetalert, args=('Accident Detected',)).start()
#                 break

#         (flag, encodedImage) = cv2.imencode(".jpg", frame)
#         if not flag:
#             continue
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

#     cap.release()

#     if notification_message:
#         return jsonify({'play_video': True})
#     else:
#         return 'No accident detected'


# def display_sweetalert(message):
#     script = f'''
#         Swal.fire({{
#             title: 'Status',
#             text: '{message}',
#             icon: 'warning',
#             confirmButtonText: 'OK',
#             customClass: {{
#                 title: 'sweet-alert-title',
#                 content: 'sweet-alert-content',
#                 confirmButton: 'sweet-alert-confirm-button'
#             }},
#             showCancelButton: false,
#             allowOutsideClick: false
#         }});
#     '''
#     return script

# def process_fighting_video(video_path):
#     # Similar to process_accident_video
#     pass

# def save_notification_record(content, additional_info):
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
#     notification = Notification(content=f'{content} - {additional_info}', sent_time=ist_time)
#     with app.app_context():
#         db.session.add(notification)
#         db.session.commit()



# def send_notification(receiver_email, subject, body, additional_info):
#     # Replace these with your Gmail credentials
#     sender_email = 'aksanap2002@gmail.com'  # Your Gmail email address
#     sender_password = 'jinq bdcg aolb xeik'  # Your Gmail password

#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = f'🚨 {subject}'  # Adding an alert symbol to the subject

#     # Structured email content
#     email_content = f'''
#     <html>
#         <body>
#             <p>Dear User,</p>
#             <p>🚨 {body}</p>  
#             <p>{additional_info}</p>
#             <p>Stay Safe,</p>
#             <p>Satark</p>
#         </body>
#     </html>
#     '''

#     msg.attach(MIMEText(email_content, 'html'))

#     # Save the current time in IST
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
    
#     # Save record to the database with IST time
#     save_notification_record('Email Sent', ist_time)

#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     server.login(sender_email, sender_password)

#     text = msg.as_string()
#     server.sendmail(sender_email, receiver_email, text)

#     server.quit()

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()

#     app.run(debug=True)





















    ############################################################################################################################################################

# from flask import Flask, render_template, request, Response, redirect
# import cv2
# import numpy as np
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage
# from email.mime.base import MIMEBase
# from email import encoders
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# from tensorflow import keras
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime, timedelta
# from pytz import timezone

# app = Flask(__name__, static_folder='assets', template_folder='templates')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notifications.db'  # Use SQLite for simplicity
# db = SQLAlchemy(app)

# # Notification model to store records
# class Notification(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     sent_time = db.Column(db.DateTime, default=datetime.utcnow)
#     content = db.Column(db.Text)

# # Load the trained models
# accident_model = tf.keras.models.load_model('accidents.h5')  # Replace with the actual path to your saved accident detection model
# fighting_model = tf.keras.models.load_model('convlstm_model_.h5')  # Replace with the actual path to your saved fighting detection model

# # Set the image dimensions (assuming the same as used during training)
# img_height = 250
# img_width = 250

# # Define class names (replace with your actual class names)
# accident_class_names = ["Non Accident", "Accident"]
# fighting_class_names = ["Abuse", "Fighting"]

# receiver_email = 'ambika.sanap@vit.edu.in'  # Receiver email address

# # Assuming your server is running in a different timezone (e.g., UTC)
# server_timezone = timezone('UTC')
# indian_timezone = timezone('Asia/Kolkata')

# # Disable the favicon route
# @app.route('/favicon.ico')
# def favicon():
#     return app.send_static_file('favicon.ico')

# @app.route("/<a>")
# def k1(a):
#     t = a + ".html"
#     return render_template(t)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     video_path = request.form['video_path']
#     detection_option = request.form['detection_option']

#     if detection_option == 'accident':
#         return Response(process_accident_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     elif detection_option == 'violence':
#         return Response(process_fighting_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return "Invalid detection option"

# def process_accident_video(video_path):
#     cap = cv2.VideoCapture(video_path)

#     overall_accident_score = 0
#     frame_count = 0
#     notification_message = None

#     with app.app_context():  # Ensure database access within Flask application context
#         while cap.isOpened():
#             ret, frame = cap.read()

#             if not ret:
#                 break

#             # Preprocess the frame for accident model prediction
#             img_accident = cv2.resize(frame, (img_height, img_width))
#             img_array_accident = image.img_to_array(img_accident)
#             img_array_accident = np.expand_dims(img_array_accident, axis=0)
#             img_array_accident /= 255.0  # Normalize pixel values

#             # Make predictions for accident detection
#             predictions_accident = accident_model.predict(img_array_accident)
#             accident_score = predictions_accident[0, accident_class_names.index('Accident')]

#             # Accumulate scores for each frame
#             overall_accident_score += accident_score
#             frame_count += 1

#             # Check for accident and send notification
#             if frame_count > 10:  # Send notification after processing 10 frames
#                 avg_accident_score = overall_accident_score / frame_count

#                 if avg_accident_score > 0.5:
#                     notification_message = 'Notification sent: Alert !!  Accident  detected'
#                     save_notification_record('Accident Detected', video_path)
#                     send_notification(receiver_email, 'Alert !! Urgent !!', 'Accident Detected',
#                                       'An accident has been detected in the video. Immediate attention is required.'
#                                       'Please take the necessary actions to ensure safety. \n\n'
#                                       'Video: ' + video_path)
#                     break

#             # Encode the frame as JPEG and yield it
#             (flag, encodedImage) = cv2.imencode(".jpg", frame)
#             if not flag:
#                 continue
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

#     # Release the video capture object
#     cap.release()

#     # Return a message if no accident is detected
#     if notification_message:
#         return notification_message
#     else:
#         return 'No accident detected'


# def process_fighting_video(video_path):
#     cap = cv2.VideoCapture(video_path)

#     overall_fighting_score = 0
#     frame_count = 0
#     notification_message = None

#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break

#         # Preprocess the frame for fighting model prediction
#         img_fighting = cv2.resize(frame, (img_height, img_width))
#         img_array_fighting = image.img_to_array(img_fighting)
#         img_array_fighting = np.expand_dims(img_array_fighting, axis=0)
#         img_array_fighting /= 255.0  # Normalize pixel values

#         # Make predictions for fighting detection
#         predictions_fighting = fighting_model.predict(img_array_fighting)
#         fighting_score = predictions_fighting[0, fighting_class_names.index('Fighting')]

#         # Accumulate scores for each frame
#         overall_fighting_score += fighting_score
#         frame_count += 1

#         # Check for fighting and send notification
#         if frame_count > 10:  # Send notification after processing 10 frames
#             avg_fighting_score = overall_fighting_score / frame_count

#             if avg_fighting_score > 0.5:
#                 notification_message = 'Notification sent: Alert !! Fighting detected'
#                 save_notification_record('Fighting Detected', video_path)
#                 send_notification(receiver_email, 'Alert !! Urgent !!', 'Fighting Detected',
#                                   'Fighting has been detected in the video. Immediate attention is required.'
#                                   'Please take the necessary actions to ensure safety. \n\n'
#                                   'Video: ' + video_path)
#                 break

#         # Encode the frame as JPEG and yield it
#         (flag, encodedImage) = cv2.imencode(".jpg", frame)
#         if not flag:
#             continue
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

#     # Release the video capture object
#     cap.release()

#     # Return a message if no fighting is detected
#     if notification_message:
#         return notification_message
#     else:
#         return 'No fighting detected'

# def save_notification_record(content, additional_info):
#     # Save record to the database
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
#     notification = Notification(content=f'{content} - {additional_info}', sent_time=ist_time)
#     db.session.add(notification)
#     db.session.commit()

# def send_notification(receiver_email, subject, body, additional_info):
#     # Replace these with your Gmail credentials
#     sender_email = 'aksanap2002@gmail.com'  # Your Gmail email address
#     sender_password = 'jinq bdcg aolb xeik'  # Your Gmail password

#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = f'🚨 {subject}'  # Adding an alert symbol to the subject

#     # Structured email content
#     email_content = f'''
#     <html>
#         <body>
#             <p>Dear User,</p>
#             <p>🚨 {body}</p>  
#             <p>{additional_info}</p>
#             <p>Stay Safe,</p>
#             <p>Satark</p>
#         </body>
#     </html>
#     '''

#     msg.attach(MIMEText(email_content, 'html'))

#     # Save the current time in IST
#     ist_time = datetime.now(server_timezone).astimezone(indian_timezone)
    
#     # Save record to the database with IST time
#     save_notification_record('Email Sent', ist_time)

#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     server.login(sender_email, sender_password)

#     text = msg.as_string()
#     server.sendmail(sender_email, receiver_email, text)

#     server.quit()

# def display_sweetalert(message):
#     script = f'''
#         Swal.fire({{
#             title: 'Status',
#             text: '{message}',
#             icon: 'warning',
#             confirmButtonText: 'OK',
#             customClass: {{
#                 title: 'sweet-alert-title',
#                 content: 'sweet-alert-content',
#                 confirmButton: 'sweet-alert-confirm-button'
#             }},
#             showCancelButton: false,
#             allowOutsideClick: false
#         }});
#     '''
#     return script

# # Notification history route
# @app.route('/notification_history')
# def notification_history():
#     # Fetch all notification records from the database
#     notifications = Notification.query.all()

#     return render_template('notification_history.html', notifications=notifications)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()  # Create the database tables

#     app.run(debug=True)
