#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
#  Robot Bluetooth RC — Raspberry Pi 3B+ + L298N + HC-05  v2
#  App : "Bluetooth RC Controller" (Android)
#  Watchdog : arrêt moteurs si silence > 3s (sécurité)
#  Restart  : géré par systemd (voir robot.service)
# ──────────────────────────────────────────────────────────────
import RPi.GPIO as GPIO
import serial
import time

# ── Broches L298N (BCM) ───────────────────────────────────────
IN1 = 17   # Pin 11 → Moteur Gauche
IN2 = 18   # Pin 12
IN3 = 22   # Pin 15 → Moteur Droit
IN4 = 23   # Pin 16

WATCHDOG_SEC = 3  # arrêt si aucune commande reçue pendant Ns

# ── Init GPIO ─────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for p in [IN1, IN2, IN3, IN4]:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, GPIO.LOW)

# ── Port série ────────────────────────────────────────────────
ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=0.1)
time.sleep(1)

# ── Fonctions moteur (sens corrigés) ──────────────────────────
def stop():
    GPIO.output(IN1, False); GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, False)

def avancer():      # F  — les deux moteurs en avant (sens inversé corrigé)
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, False); GPIO.output(IN4, True)

def reculer():      # B  — les deux moteurs en arrière
    GPIO.output(IN1, True);  GPIO.output(IN2, False)
    GPIO.output(IN3, True);  GPIO.output(IN4, False)

def gauche():       # L  — pivot gauche (G avance, D recule)
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, True);  GPIO.output(IN4, False)

def droite():       # R  — pivot droit (G recule, D avance)
    GPIO.output(IN1, True);  GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, True)

def avant_gauche(): # G  — courbe avant-gauche (D avance, G stop)
    GPIO.output(IN1, False); GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, True)

def avant_droite(): # I  — courbe avant-droite (G avance, D stop)
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, False); GPIO.output(IN4, False)

def arriere_gauche(): # H  — courbe arrière-gauche (D recule, G stop)
    GPIO.output(IN1, False); GPIO.output(IN2, False)
    GPIO.output(IN3, True);  GPIO.output(IN4, False)

def arriere_droite(): # J  — courbe arrière-droite (G recule, D stop)
    GPIO.output(IN1, True);  GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, False)

# ── Table de dispatch ─────────────────────────────────────────
CMDS = {
    'F': avancer,       'B': reculer,
    'L': gauche,        'R': droite,
    'G': avant_gauche,  'I': avant_droite,
    'H': arriere_gauche,'J': arriere_droite,
    'S': stop,
    '0': stop,          # vitesse 0 = stop (certaines versions de l'app)
}

# ── Boucle principale ─────────────────────────────────────────
print("Robot pret — attente connexion Bluetooth...")
stop()
last_cmd_time = time.time()

try:
    while True:
        now = time.time()

        # Watchdog : aucune commande depuis WATCHDOG_SEC secondes
        if now - last_cmd_time > WATCHDOG_SEC:
            stop()

        if ser.in_waiting > 0:
            raw = ser.read().decode('utf-8', errors='ignore').strip().upper()
            if raw in CMDS:
                print(f"CMD: {raw}")
                CMDS[raw]()
                last_cmd_time = now

        time.sleep(0.01)  # évite 100% CPU

except (serial.SerialException, OSError) as e:
    print(f"Erreur série (déconnexion BT probable): {e}")
except KeyboardInterrupt:
    print("Arrêt manuel.")
finally:
    stop()
    try: ser.close()
    except: pass
    GPIO.cleanup()
    print("Cleanup OK — systemd va redémarrer dans 2s...")
