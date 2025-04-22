# Installer imutils si besoin : pip install imutils

import cv2
import imutils
import smtplib
import time
import threading
from email.message import EmailMessage
import os

# Chemin de la vidéo (modifie selon ton fichier)
video_path = r"C:\Users\GIGABYTE\OneDrive\Desktop\projet git\Nouveau dossier\essai face 1.mp4"

# Initialisation de la vidéo
webcam = cv2.VideoCapture(video_path)

if not webcam.isOpened():
    print("[ERREUR] Impossible d'ouvrir la vidéo. Vérifie le chemin.")
    exit()

# Initialisation du détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialisation pour l'enregistrement vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('video_output_detected.mp4', fourcc, 20.0, (500, 375))  # largeur = 500 après resize

first_frame = None
last_email_time = 0
email_interval = 60  # secondes

# Fonction d'envoi d'email (optionnelle)
def envoyer_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Alerte : Mouvement ou visage détecté'
    msg['From'] = "mabrouk122334@gmail.com"
    msg['To'] = "ahmed.mabrouk.tn@gmail.com"
    msg.set_content("Un mouvement ou un visage a été détecté. Voici l'image.")

    with open(image_path, 'rb') as f:
        file_data = f.read()
        file_name = f.name
    msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("mabrouk122334@gmail.com", "gilc ifdc svhz mhwj")
            smtp.send_message(msg)
            print("[INFO] Email envoyé.")
    except Exception as e:
        print(f"[ERREUR] Email : {e}")

def envoyer_email_thread(image_path):
    email_thread = threading.Thread(target=envoyer_email, args=(image_path,))
    email_thread.start()

frame_count = 0
while True:
    check, frame = webcam.read()
    if not check:
        print("Fin de la vidéo")
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
    
    # Affichage de la frame
    cv2.imshow("Detection", frame)

    # Enregistrement dans la vidéo de sortie
    output.write(frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    frame_count += 1
    if frame_count > 300:  # limite pour éviter que ça tourne trop longtemps
        break

# Nettoyage
webcam.release()
output.release()
cv2.destroyAllWindows()
