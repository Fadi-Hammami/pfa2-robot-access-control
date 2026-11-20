#!/bin/bash
# Installe le service de contrôle d'accès (démarrage auto au boot).
# Usage : bash install_service.sh
set -e
USER_NAME=$(whoami)
DIR=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp)

[ -f "$DIR/app.py" ] || { echo "app.py introuvable dans $DIR"; exit 1; }
[ -f "$DIR/.env" ] || echo "⚠ .env absent : copiez .env.example en .env (alertes email inactives sinon)"

# Génère le service adapté à cette machine sans modifier le fichier du dépôt
sed -e "s|^User=.*|User=$USER_NAME|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=$DIR|" \
    -e "s|^ExecStart=.*|ExecStart=/usr/bin/python3 $DIR/app.py|" \
    "$DIR/access-control.service" > "$TMP"

echo "[1/4] Copie du fichier service..."
sudo cp "$TMP" /etc/systemd/system/access-control.service
sudo chmod 644 /etc/systemd/system/access-control.service
rm -f "$TMP"
echo "[2/4] Rechargement de systemd..."
sudo systemctl daemon-reload
echo "[3/4] Activation au démarrage..."
sudo systemctl enable access-control.service
echo "[4/4] Démarrage immédiat..."
sudo systemctl restart access-control.service
sleep 3
sudo systemctl status access-control.service --no-pager -l || true
echo "Service installé. Logs : sudo journalctl -u access-control -f"
