!pip install imutils

import cv2
import imutils
import smtplib
import time
import threading
from email.message import EmailMessage
from google.colab.patches import cv2_imshow
from google.colab import files

EMAIL_ADDRESS = "your.email@gmail.com"             # <-- Remplace par ton adresse email
EMAIL_PASSWORD = "your_app_specific_password"      # <-- Remplace par ton mot de passe d'application
RECEIVER_EMAIL = "receiver.email@gmail.com"        # <-- Remplace par l'adresse email du destinataire

uploaded = files.upload()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path = 'essai face 1.mp4'                    # <-- Remplace par le nom exact de la vidéo importée
webcam = cv2.VideoCapture(video_path)

first_frame = None
last_email_time = 0
email_interval = 60  

def envoyer_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Alert: Motion or Face Detected'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL
    msg.set_content("Motion or a face was detected. See the attached image.")

    with open(image_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=f.name)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("[INFO] Email sent.")
    except Exception as e:
        print(f"[ERROR] Email: {e}")

def envoyer_email_thread(image_path):
    threading.Thread(target=envoyer_email, args=(image_path,)).start()

frame_count = 0
while True:
    check, frame = webcam.read()
    if not check:
        print("End of video")
        break

    text = "No Movement"
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    frame_diff = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue
        text = "Movement Detected"
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(roi_color, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            text = "Face Detected"

            current_time = time.time()
            if current_time - last_email_time >= email_interval:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = f"capture_{timestamp}.jpg"
                cv2.imwrite(image_path, frame)
                envoyer_email_thread(image_path)
                last_email_time = current_time

    cv2.putText(frame, f"Status: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2_imshow(frame)

    frame_count += 1
    if frame_count > 300:
        break

webcam.release()
cv2.destroyAllWindows()
