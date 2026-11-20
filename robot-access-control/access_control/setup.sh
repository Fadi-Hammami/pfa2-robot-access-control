#!/bin/bash
# ============================================================
# SCRIPT D'INSTALLATION - Système de contrôle d'accès
# Exécuter avec: bash setup.sh
# ============================================================

echo "============================================"
echo "  INSTALLATION DU SYSTÈME DE CONTRÔLE"
echo "============================================"

# Mise à jour du système
echo "[1/5] Mise à jour du système..."
sudo apt update && sudo apt upgrade -y

# Installation des dépendances système
echo "[2/5] Installation des dépendances système..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-picamera2 \
    python3-numpy \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    python3-flask \
    python3-flask-cors \
    git

# Installation des packages Python
echo "[3/5] Installation des packages Python..."
pip3 install --break-system-packages \
    flask \
    flask-cors \
    opencv-contrib-python \
    picamera2 \
    numpy

# Téléchargement du fichier Haar Cascade (détecteur de visages)
echo "[4/5] Téléchargement du détecteur de visages Haar Cascade..."
wget -q https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
echo "✓ haarcascade_frontalface_default.xml téléchargé"

# Création des dossiers du projet
echo "[5/5] Création de la structure des dossiers..."
mkdir -p data/authorized data/denied models logs templates static

echo ""
echo "============================================"
echo "  INSTALLATION TERMINÉE !"
echo "============================================"
echo ""
echo "Étapes suivantes:"
echo "  1. python3 capture.py     → Capturer les visages"
echo "  2. python3 train.py       → Entraîner le modèle"
echo "  3. python3 app.py         → Lancer le système"
echo ""
echo "Dashboard web: http://$(hostname -I | awk '{print $1}'):5000"
