#!/usr/bin/env python3
"""
============================================================
ÉTAPE 1 : CAPTURE DES VISAGES
============================================================
Ce script capture les photos de visage pour construire
la base de données du système de reconnaissance.

Usage: python3 capture.py
============================================================
"""

from picamera2 import Picamera2
import cv2
import os
import time

def _haar_path():
    """Fichier Haar local s'il existe, sinon celui fourni par OpenCV."""
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'haarcascade_frontalface_default.xml')
    return local if os.path.exists(local) else (
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ============================================================
# CONFIGURATION
# ============================================================
NOMBRE_PHOTOS = 50          # Nombre de photos par personne (50 recommandé)
RESOLUTION = (640, 480)     # Résolution de la caméra
MIN_FACE_SIZE = (80, 80)    # Taille minimale de visage à détecter
PHOTO_SIZE = (200, 200)     # Taille de sauvegarde des photos


def afficher_menu():
    """Affiche le menu de sélection du type de personne."""
    print("\n" + "="*50)
    print("   SYSTÈME DE CAPTURE DE VISAGES")
    print("="*50)
    print("\nType de personne à enregistrer:")
    print("  [1] Personne AUTORISÉE (accès permis)")
    print("  [2] Personne REFUSÉE  (accès bloqué)")
    print("  [q] Quitter")
    print()


def initialiser_camera():
    """Initialise et configure la caméra Pi."""
    print("Initialisation de la caméra...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": RESOLUTION, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.5)  # Laisser la caméra se stabiliser
    print("✓ Caméra prête")
    return picam2


def charger_detecteur():
    """Charge le détecteur de visages Haar Cascade."""
    detecteur = cv2.CascadeClassifier(_haar_path())
    if detecteur.empty():
        print("\n✗ ERREUR: haarcascade_frontalface_default.xml non trouvé!")
        print("  Exécutez d'abord: bash setup.sh")
        exit(1)
    return detecteur


def capturer_visages(nom, dossier_base, picam2, detecteur):
    """
    Boucle principale de capture des photos.
    
    Args:
        nom: Nom de la personne
        dossier_base: 'data/authorized' ou 'data/denied'
        picam2: Instance de la caméra
        detecteur: Classificateur Haar Cascade
    
    Returns:
        int: Nombre de photos capturées
    """
    # Créer le dossier pour cette personne
    dossier = os.path.join(dossier_base, nom)
    os.makedirs(dossier, exist_ok=True)

    # Compter les photos existantes pour ne pas écraser
    photos_existantes = len([f for f in os.listdir(dossier) if f.endswith('.jpg')])
    compteur = photos_existantes
    frame_count = 0
    faces = []
    auto_capture = False
    dernier_capture = 0

    print(f"\n{'='*50}")
    print(f"  Capture pour: {nom}")
    print(f"  Objectif: {NOMBRE_PHOTOS} photos")
    print(f"  Photos existantes: {photos_existantes}")
    print(f"{'='*50}")
    print("\nContrôles:")
    print("  [ESPACE] → Capturer manuellement")
    print("  [A]      → Mode auto-capture (1 photo/sec)")
    print("  [Q]      → Terminer")
    print("\nPositionnez votre visage face à la caméra...\n")

    while compteur < NOMBRE_PHOTOS:
        frame = picam2.capture_array()
        display = frame.copy()

        # Détecter les visages toutes les 3 frames (optimisation CPU)
        frame_count += 1
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = detecteur.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=5, 
                minSize=MIN_FACE_SIZE
            )

        # Auto-capture: une photo toutes les secondes si visage détecté
        maintenant = time.time()
        if auto_capture and len(faces) > 0 and (maintenant - dernier_capture) > 1.0:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            (x, y, w, h) = faces[0]
            visage = gray[y:y+h, x:x+w]
            visage = cv2.resize(visage, PHOTO_SIZE)
            chemin = os.path.join(dossier, f"{nom}_{compteur:04d}.jpg")
            cv2.imwrite(chemin, visage)
            compteur += 1
            dernier_capture = maintenant
            print(f"  ✓ Auto: {compteur}/{NOMBRE_PHOTOS}")

        # Dessiner les rectangles et informations
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 100), 2)
            cv2.rectangle(display, (x, y-30), (x+w, y), (0, 255, 100), -1)
            cv2.putText(display, "Visage OK", (x+5, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Barre de progression
        progression = int((compteur / NOMBRE_PHOTOS) * (RESOLUTION[0] - 20))
        cv2.rectangle(display, (10, RESOLUTION[1]-25), (RESOLUTION[0]-10, RESOLUTION[1]-10), (60, 60, 60), -1)
        cv2.rectangle(display, (10, RESOLUTION[1]-25), (10+progression, RESOLUTION[1]-10), (0, 200, 100), -1)

        # Informations à l'écran
        mode_texte = "AUTO" if auto_capture else "MANUEL"
        cv2.putText(display, f"{nom} | {compteur}/{NOMBRE_PHOTOS} photos | Mode: {mode_texte}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)

        if len(faces) == 0:
            cv2.putText(display, "Aucun visage detecte - approchez-vous",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        cv2.imshow(f'Capture - {nom}', display)
        key = cv2.waitKey(1) & 0xFF

        # Contrôles clavier
        if key == ord(' '):
            # Capture manuelle
            if len(faces) > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                (x, y, w, h) = faces[0]
                visage = gray[y:y+h, x:x+w]
                visage = cv2.resize(visage, PHOTO_SIZE)
                chemin = os.path.join(dossier, f"{nom}_{compteur:04d}.jpg")
                cv2.imwrite(chemin, visage)
                compteur += 1
                print(f"  ✓ Manuel: {compteur}/{NOMBRE_PHOTOS}")
            else:
                print("  ✗ Aucun visage détecté!")

        elif key == ord('a') or key == ord('A'):
            # Toggle mode auto
            auto_capture = not auto_capture
            mode = "activé" if auto_capture else "désactivé"
            print(f"  → Mode auto-capture {mode}")

        elif key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()
    return compteur


def main():
    """Point d'entrée principal."""
    detecteur = charger_detecteur()
    picam2 = initialiser_camera()

    try:
        while True:
            afficher_menu()
            choix = input("Votre choix: ").strip().lower()

            if choix == 'q':
                print("\nAu revoir!")
                break

            elif choix == '1':
                dossier_base = "data/authorized"
                type_str = "AUTORISÉE"
            elif choix == '2':
                dossier_base = "data/denied"
                type_str = "REFUSÉE"
            else:
                print("Choix invalide, réessayez.")
                continue

            nom = input(f"Nom de la personne {type_str}: ").strip()
            if not nom:
                print("Nom vide, opération annulée.")
                continue

            # Vérifier si la personne existe déjà
            dossier = os.path.join(dossier_base, nom)
            if os.path.exists(dossier):
                photos = len([f for f in os.listdir(dossier) if f.endswith('.jpg')])
                print(f"\n⚠ {nom} existe déjà ({photos} photos)")
                continuer = input("Continuer la capture? [o/n]: ").strip().lower()
                if continuer != 'o':
                    continue

            nb = capturer_visages(nom, dossier_base, picam2, detecteur)

            if nb >= NOMBRE_PHOTOS:
                print(f"\n✓ SUCCÈS! {nb} photos de {nom} sauvegardées.")
            else:
                print(f"\n⚠ Seulement {nb}/{NOMBRE_PHOTOS} photos capturées.")

            continuer = input("\nCapturer une autre personne? [o/n]: ").strip().lower()
            if continuer != 'o':
                print("\n✓ Capture terminée. Lancez maintenant: python3 train.py")
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
