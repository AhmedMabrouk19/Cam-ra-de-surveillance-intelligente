# Installer les bibliothèques nécessaires!!: pip install imutils python-dotenv 

import cv2
import imutils
import smtplib
import time
import threading
from email.message import EmailMessage
import os
from dotenv import load_dotenv

# Charger les informations depuis le fichier .env
load_dotenv()
EMAIL_SOURCE = os.getenv("EMAIL_SOURCE")              # <-- Remplace par ton adresse email
EMAIL_DESTINATION = os.getenv("EMAIL_DESTINATION")    # <-- Remplace par l'adresse email du destinataire
EMAIL_MOT_DE_PASSE = os.getenv("EMAIL_MOT_DE_PASSE")  # <-- Remplace par ton mot de passe d'application


# Chemin de la vidéo à analyser
chemin_video = r"C:\Users\GIGABYTE\OneDrive\Desktop\projet git\Nouveau dossier\essai face 1.mp4"  # <-- Remplacer par le chemin de ta vidéo
# Initialisation de la capture vidéo
webcam = cv2.VideoCapture(chemin_video)
if not webcam.isOpened():
    print("[ERREUR] Impossible d'ouvrir la vidéo. Vérifie le chemin.")
    exit()

# Détecteur de visages
detecteur_visages = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuration de l’enregistrement vidéo
codec = cv2.VideoWriter_fourcc(*'mp4v')
sortie_video = cv2.VideoWriter('video_detectee.mp4', codec, 20.0, (500, 375))

premiere_image = None
dernier_envoi_email = 0
intervalle_email = 60  # secondes

# Fonction pour envoyer un email
def envoyer_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Alerte : Visage ou mouvement détecté'
    msg['From'] = EMAIL_SOURCE
    msg['To'] = EMAIL_DESTINATION
    msg.set_content("Un visage ou un mouvement a été détecté. Voir la pièce jointe.")

    with open(image_path, 'rb') as f:
        contenu_image = f.read()
        nom_fichier = os.path.basename(f.name)
    msg.add_attachment(contenu_image, maintype='image', subtype='jpeg', filename=nom_fichier)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SOURCE, EMAIL_MOT_DE_PASSE)
            smtp.send_message(msg)
            print("[INFO] Email envoyé avec succès.")
    except Exception as e:
        print(f"[ERREUR] Envoi de l'email : {e}")

# Lancer l'envoi d'email dans un thread séparé
def envoyer_email_thread(image_path):
    thread = threading.Thread(target=envoyer_email, args=(image_path,))
    thread.start()

nombre_images = 0
while True:
    check, image = webcam.read()
    if not check:
        print("Fin de la vidéo.")
        break

    etat = "Pas de mouvement"
    image = imutils.resize(image, width=500)
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (21, 21), 0)

    if premiere_image is None:
        premiere_image = gris
        continue

    difference = cv2.absdiff(premiere_image, gris)
    seuil = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
    seuil = cv2.dilate(seuil, None, iterations=2)

    contours, _ = cv2.findContours(seuil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue
        etat = "Mouvement détecté"
        (x, y, l, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+l, y+h), (0, 255, 0), 2)

        zone_gris = gris[y:y+h, x:x+l]
        zone_couleur = image[y:y+h, x:x+l]
        visages = detecteur_visages.detectMultiScale(zone_gris, scaleFactor=1.1, minNeighbors=5)

        for (vx, vy, vl, vh) in visages:
            cv2.rectangle(zone_couleur, (vx, vy), (vx+vl, vy+vh), (255, 0, 0), 2)
            etat = "Visage détecté"

            maintenant = time.time()
            if maintenant - dernier_envoi_email >= intervalle_email:
                horodatage = time.strftime("%Y%m%d_%H%M%S")
                chemin_image = f"capture_{horodatage}.jpg"
                cv2.imwrite(chemin_image, image)
                envoyer_email_thread(chemin_image)
                dernier_envoi_email = maintenant

    cv2.putText(image, f"Statut : {etat}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Détection", image)
    sortie_video.write(image)

    if cv2.waitKey(1) == ord('q'):
        break

    nombre_images += 1
    if nombre_images > 300:  # Limite de frames
        break

# Libération des ressources
webcam.release()
sortie_video.release()
cv2.destroyAllWindows()
