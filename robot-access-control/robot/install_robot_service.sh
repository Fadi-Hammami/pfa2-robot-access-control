#!/bin/bash
# Installe le service de mobilité du robot (démarrage auto au boot).
# Usage : bash install_robot_service.sh
set -e
USER_NAME=$(whoami)
DIR=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp)

[ -f "$DIR/robot_v2.py" ] || { echo "robot_v2.py introuvable dans $DIR"; exit 1; }

# Génère le service adapté à cette machine sans modifier le fichier du dépôt
sed -e "s|^User=.*|User=$USER_NAME|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=$DIR|" \
    -e "s|^ExecStart=.*|ExecStart=/usr/bin/python3 $DIR/robot_v2.py|" \
    "$DIR/robot.service" > "$TMP"

sudo cp "$TMP" /etc/systemd/system/robot.service
sudo chmod 644 /etc/systemd/system/robot.service
rm -f "$TMP"
sudo systemctl daemon-reload
sudo systemctl enable robot.service
sudo systemctl restart robot.service
sleep 2
sudo systemctl status robot.service --no-pager -l || true
echo "Service robot installé. Logs : sudo journalctl -u robot -f"
