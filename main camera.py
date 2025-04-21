import cv2
import imutils
import smtplib
import time
import threading
from email.message import EmailMessage

# Remplacez par vos véritables informations d'identification email
EMAIL_ADDRESS = "votre.email@gmail.com"
EMAIL_PASSWORD = "votre_mot_de_passe_app_specifique"
RECEIVER_EMAIL = "email_destinataire@gmail.com"

# Mode sélectionné : "visage"
mode = "visage"  

# Chargement du modèle pour la détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Connexion à la webcam
webcam = cv2.VideoCapture(0)
first_frame = None
last_email_time = 0
email_interval = 60  # Intervalle en secondes entre l'envoi des emails

# Fonction pour envoyer un email avec l'image capturée
def envoyer_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Alerte : Mouvement ou visage détecté'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL
    msg.set_content("Un mouvement ou un visage a été détecté. Voici l'image du moment.")

    # Attacher l'image en pièce jointe
    with open(image_path, 'rb') as f:
        file_data = f.read()
        file_name = f.name
    msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

    try:
        # Envoi de l'email via le serveur SMTP de Gmail
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("[INFO] Email envoyé.")
    except Exception as e:
        print(f"[ERREUR] Envoi e-mail : {e}")

# Fonction pour envoyer l'email dans un thread séparé
def envoyer_email_thread(image_path):
    threading.Thread(target=envoyer_email, args=(image_path,)).start()

while True:
    check, frame = webcam.read()
    text = "Aucun mouvement"

    # Prétraitement de l'image
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5) if mode == "visage" else (21, 21), 0)

    # Initialisation du premier cadre pour la détection de mouvement
    if first_frame is None:
        first_frame = gray
        continue

    # Calcul de la différence entre le cadre initial et le cadre actuel
    frame_diff = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue
        text = "Mouvement détecté"
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(roi_color, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            text = "Visage détecté"

            # Envoi d'email si assez de temps a passé
            current_time = time.time()
            if current_time - last_email_time >= email_interval:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = f"image_capturee_{timestamp}.jpg"
                cv2.imwrite(image_path, frame)
                envoyer_email_thread(image_path)
                last_email_time = current_time

    # Affichage du statut sur l'image
    cv2.putText(frame, f"Statut: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Fil de sécurité", frame)

    # Quitter la boucle avec la touche 'ESC'
    if cv2.waitKey(1) == 27:  
        break

# Libérer la webcam et fermer toutes les fenêtres OpenCV
webcam.release()
cv2.destroyAllWindows()
