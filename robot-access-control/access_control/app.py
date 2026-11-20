#!/usr/bin/env python3
"""
Système de Contrôle d'Accès par Reconnaissance Faciale — v8 FINAL
Projet de Fin d'Année — ENIT — Fadi Hammemi

Inspiré de camex8.py (face_recognition) — adapté LBPH Raspberry Pi.
Principe clé identique à camex8 :
  • recognizer, labels_map, personnes_ref → structures partagées protégées
    par un seul verrou (_model_lock)
  • email_poll modifie ces structures DIRECTEMENT (comme camex8 fait
    known_face_encodings.append() directement)
  • Pas d'intermédiaire _model_upd, pas de globals() — zéro ambiguïté

Logique de décision (A → B → C) :
  A. Visage REFUSÉ  → rouge  + "REFUSE: nom"
  B. Visage AUTORISÉ → vert   + "nom XX%"
  C. INCONNU         → orange + email admin
     admin répond ACCEPTER/REFUSER → update immédiat + stream mis à jour
"""

import cv2, pickle, numpy as np
import threading, sqlite3, logging, os, time, shutil, re
import smtplib, imaplib, email as email_lib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from collections          import deque, Counter
from datetime             import datetime, timedelta
from flask                import Flask, Response, render_template, jsonify
from picamera2            import Picamera2


# ════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _charger_env(chemin):
    """Lit un fichier .env (CLE=valeur) sans dépendance externe."""
    if not os.path.exists(chemin):
        return
    with open(chemin, encoding='utf-8') as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith('#') or '=' not in ligne:
                continue
            cle, val = ligne.split('=', 1)
            os.environ.setdefault(cle.strip(), val.strip().strip('"').strip("'"))


_charger_env(os.path.join(BASE_DIR, '.env'))

# Identifiants lus depuis .env (jamais écrits dans le code, voir .env.example)
EMAIL = {
    'expediteur':   os.environ.get('EMAIL_SENDER', ''),
    'mot_de_passe': os.environ.get('EMAIL_PASSWORD', ''),
    'destinataire': os.environ.get('EMAIL_RECIPIENT', ''),
}

# Détecteur Haar : fichier local s'il existe, sinon celui fourni par OpenCV
HAAR_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
if not os.path.exists(HAAR_PATH):
    HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

CFG = {
    'resolution':  (640, 480),
    'stream_fps':  20,
    'stream_jpeg': 65,
    'seuil':       65,       # LBPH conf < seuil → reconnu
    'detect_every': 3,
    'votes':       6,
    'ratio':       0.60,
    'min_votes':   3,
    'delai_alerte_min': 4,
    'email_poll_sec':   20,  # Poll toutes les 20s (plus réactif)
    'port': 5000,
}


# ════════════════════════════════════════════════════════════
#  INIT
# ════════════════════════════════════════════════════════════
for _d in ['logs', 'logs/visages_temp', 'models',
           'data/authorized', 'data/authorized/authorized_temp',
           'data/denied',     'data/denied/denied_temp']:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    filename='logs/system.log', level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')

app = Flask(__name__)


# ════════════════════════════════════════════════════════════
#  MODÈLE PARTAGÉ — structure centrale (pattern camex8)
#
#  Toutes les structures de reconnaissance sont protégées par
#  _model_lock.  Le thread email les modifie DIRECTEMENT,
#  comme camex8 fait known_face_encodings.append().
# ════════════════════════════════════════════════════════════
_model_lock    = threading.Lock()
_recognizer    = None          # cv2.face.LBPHFaceRecognizer
_labels_map    = {}            # { label_int: nom_str }
_personnes_ref = set()         # noms des personnes refusées
_next_label    = 0             # prochain label libre
_modele_ok     = False


def _charger_modele():
    """Charge le modèle depuis le disque. Appelé au démarrage."""
    global _recognizer, _labels_map, _personnes_ref, _next_label, _modele_ok
    if os.path.exists('models/model.yml') and os.path.exists('models/labels.pkl'):
        try:
            r = cv2.face.LBPHFaceRecognizer_create()
            r.read('models/model.yml')
            with open('models/labels.pkl', 'rb') as f:
                d = pickle.load(f)
            with _model_lock:
                _recognizer    = r
                _labels_map    = d.get('labels', {})
                _personnes_ref = set(d.get('denied', []))
                _next_label    = max(_labels_map.keys(), default=-1) + 1
                _modele_ok     = True
            print(f"  ✓ Modèle LBPH — {len(_labels_map)} personne(s)")
            print(f"    Autorisées : {[v for v in _labels_map.values() if v not in _personnes_ref]}")
            print(f"    Refusées   : {list(_personnes_ref)}")
        except Exception as e:
            print(f"  ✗ Modèle: {e}")
    else:
        print("  ⚠ Modèle absent — tout visage = INCONNU")
        print("    → Lancez: python3 capture.py  puis  python3 train.py")


def _update_modele_en_memoire(nom: str, action: str, gray_roi):
    """
    Met à jour le modèle en mémoire DIRECTEMENT — même principe que
    camex8.py: known_face_encodings.append(face_encoding).

    Appelé depuis le thread email_poll après décision admin.
    Protégé par _model_lock.
    """
    global _recognizer, _labels_map, _personnes_ref, _next_label, _modele_ok

    with _model_lock:
        if _recognizer is None:
            # Pas de modèle : créer un nouveau
            _recognizer = cv2.face.LBPHFaceRecognizer_create()

        if gray_roi is not None:
            try:
                roi = cv2.resize(gray_roi, (200, 200))
                roi = cv2.equalizeHist(roi)
                _recognizer.update([roi], np.array([_next_label]))
            except Exception as e:
                logging.error(f"recognizer.update: {e}")
                print(f"  ✗ recognizer.update: {e}")
                return

        _labels_map[_next_label] = nom
        if action == 'refuse':
            _personnes_ref.add(nom)
        _next_label += 1
        _modele_ok = True

    print(f"  ✓ Modèle mis à jour en mémoire : {nom} ({'REFUSÉ' if action=='refuse' else 'AUTORISÉ'})")
    logging.info(f"Modèle mis à jour : {nom} {action}")


# ════════════════════════════════════════════════════════════
#  ÉTAT PARTAGÉ ENTRE THREADS
# ════════════════════════════════════════════════════════════
_raw_frame = None
_raw_lock  = threading.Lock()
_raw_ready = threading.Event()

_ann_frame = None
_ann_lock  = threading.Lock()

_event     = {'type': 'attente', 'nom': '', 'heure': '', 'confiance': 0}
_ev_lock   = threading.Lock()

# { eid: {'chemin': str, 'gray_roi': ndarray} }
_pending      = {}
_pending_lock = threading.Lock()

_notifs      = deque(maxlen=20)
_notifs_lock = threading.Lock()

_stats   = {'total': 0, 'autorises': 0, 'refuses': 0, 'inconnus': 0}
_running  = True
camera_ok = False


# ════════════════════════════════════════════════════════════
#  BASE DE DONNÉES
# ════════════════════════════════════════════════════════════
DB = 'logs/access.db'

def init_db():
    c = sqlite3.connect(DB)
    c.execute('''CREATE TABLE IF NOT EXISTS acces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        horodatage TEXT, nom TEXT, statut TEXT, confiance REAL DEFAULT 0)''')
    c.commit(); c.close()

def db_log(nom, statut, conf=0.0):
    try:
        c = sqlite3.connect(DB)
        c.execute('INSERT INTO acces VALUES (NULL,?,?,?,?)',
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), nom, statut, round(conf,1)))
        c.commit(); c.close()
    except Exception as e:
        logging.error(f"DB: {e}")


# ════════════════════════════════════════════════════════════
#  NOTIFICATIONS DASHBOARD
# ════════════════════════════════════════════════════════════
def push_notif(type_: str, titre: str, corps: str):
    with _notifs_lock:
        _notifs.appendleft({'id': datetime.now().strftime('%f'),
                            'type': type_, 'titre': titre, 'corps': corps,
                            'heure': datetime.now().strftime('%H:%M:%S')})


# ════════════════════════════════════════════════════════════
#  PLACEHOLDER VIDÉO
# ════════════════════════════════════════════════════════════
def _placeholder(l1="Initialisation...", l2=""):
    W, H = CFG['resolution']
    img = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)
    cv2.putText(img, l1, (W//2 - len(l1)*7, H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80,180,255), 2, cv2.LINE_AA)
    if l2:
        cv2.putText(img, l2, (W//2 - len(l2)*5, H//2+38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,120), 1, cv2.LINE_AA)
    cv2.putText(img, datetime.now().strftime('%H:%M:%S'),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,90), 1)
    return img


# ════════════════════════════════════════════════════════════
#  STABILISATEUR DE DÉCISION
# ════════════════════════════════════════════════════════════
class Stabilisateur:
    def __init__(self, max_v, ratio, min_v):
        self._buf   = deque(maxlen=max_v)
        self._ratio = ratio
        self._min   = min_v
        self._last  = None

    def vote(self, label: int, conf: float):
        self._buf.append((label, conf))
        if len(self._buf) < self._min:
            return None, 0
        ctr = Counter(l for l,_ in self._buf)
        top, cnt = ctr.most_common(1)[0]
        if cnt / len(self._buf) >= self._ratio:
            mean_c = sum(c for l,c in self._buf if l==top) / cnt
            self._last = (top, mean_c)
            return top, mean_c
        return self._last if self._last else (None, 0)

    def reset(self):
        self._buf.clear()
        self._last = None


# ════════════════════════════════════════════════════════════
#  THREAD 1 — CAPTURE  (ne fait QUE capturer)
# ════════════════════════════════════════════════════════════
def thread_capture():
    global _raw_frame, _ann_frame, camera_ok

    picam2 = None
    for i in range(3):
        try:
            picam2 = Picamera2()
            cfg = picam2.create_preview_configuration(
                main={"size": CFG['resolution'], "format": "BGR888"},
                controls={"FrameRate": 30})
            picam2.configure(cfg)
            picam2.start()
            time.sleep(1.5)
            camera_ok = True
            print(f"  ✓ Caméra {CFG['resolution'][0]}×{CFG['resolution'][1]}")
            break
        except Exception as e:
            print(f"  ✗ Caméra tentative {i+1}/3 : {e}")
            if picam2:
                try: picam2.stop()
                except: pass
            picam2 = None
            time.sleep(2)

    if not camera_ok:
        ph = _placeholder("Camera indisponible", "Verifiez la connexion")
        while _running:
            with _raw_lock:  _raw_frame = ph
            with _ann_lock:  _ann_frame = ph
            time.sleep(0.1)
        return

    try:
        while _running:
            frame = picam2.capture_array()
            with _raw_lock:
                _raw_frame = frame
            _raw_ready.set()
            with _ann_lock:
                if _ann_frame is None:
                    _ann_frame = frame.copy()
    except Exception as e:
        logging.error(f"Capture: {e}")
    finally:
        try: picam2.stop()
        except: pass


# ════════════════════════════════════════════════════════════
#  THREAD 2 — RECONNAISSANCE
#  Lit _model_lock à chaque frame → voit immédiatement les
#  mises à jour faites par le thread email (comme camex8).
# ════════════════════════════════════════════════════════════
def thread_reconnaissance():
    global _ann_frame

    det = cv2.CascadeClassifier(HAAR_PATH)
    if det.empty():
        print("  ✗ haarcascade_frontalface_default.xml manquant")
        det = None

    stab         = Stabilisateur(CFG['votes'], CFG['ratio'], CFG['min_votes'])
    it           = 0
    faces        = np.array([])
    gray         = None
    couleur      = (200, 160, 0)
    texte        = "Analyse..."
    alerte_ts    = datetime.min
    delai_alerte = timedelta(minutes=CFG['delai_alerte_min'])
    log_memo     = {'last': '', 'last_t': datetime.min}

    print("  ✓ Thread reconnaissance démarré")

    while _running:
        _raw_ready.wait(timeout=0.1)
        _raw_ready.clear()

        with _raw_lock:
            raw = _raw_frame
        if raw is None:
            time.sleep(0.05); continue

        display = raw.copy()
        heure   = datetime.now().strftime('%H:%M:%S')
        it     += 1

        # Détection Haar
        if it % CFG['detect_every'] == 0 and det is not None:
            try:
                gray  = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                gray  = cv2.equalizeHist(gray)
                faces = det.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5,
                    minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE)
            except:
                faces = np.array([])

        if gray is None:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        nb = len(faces) if isinstance(faces, np.ndarray) else 0

        if nb == 0:
            stab.reset()
            couleur = (200, 160, 0)
            texte   = "Analyse..."
            with _ev_lock:
                _event.update({'type': 'attente', 'nom': '', 'heure': heure, 'confiance': 0})

        # ── Lire le modèle partagé UNE FOIS par cycle ────────
        # (même principe que camex8 lit known_face_encodings)
        with _model_lock:
            recognizer    = _recognizer
            labels_map    = dict(_labels_map)
            personnes_ref = set(_personnes_ref)
            modele_ok     = _modele_ok

        for (x, y, w, h) in (faces if nb > 0 else []):
            roi = gray[y:y+h, x:x+w] if gray is not None else None

            # ── Prédiction ───────────────────────────────────
            if recognizer is not None and roi is not None:
                try:
                    lbl, conf = recognizer.predict(roi)
                    vote_lbl  = lbl if conf < CFG['seuil'] else -1
                    dec, dec_conf = stab.vote(vote_lbl, conf)
                    if it % 25 == 0:
                        print(f"  [conf] {labels_map.get(lbl,'?')} = {conf:.1f}"
                              f" → {'OK' if vote_lbl>=0 else 'INCONNU'}")
                except Exception as e:
                    dec, dec_conf = -1, 0
                    logging.error(f"Predict: {e}")
            else:
                dec, dec_conf = -1, 0

            if dec is None:
                pass  # stabilisateur en cours

            # ══ A : REFUSÉ ════════════════════════════════════
            elif dec >= 0 and labels_map.get(dec, '') in personnes_ref:
                nom    = labels_map[dec]
                couleur = (40, 40, 230)
                texte   = f"REFUSE: {nom}"
                with _ev_lock:
                    _event.update({'type':'refuse','nom':nom,'heure':heure,
                                   'confiance':max(0,int(100-dec_conf))})
                _log_once(nom, 'REFUSE', dec_conf, log_memo)

            # ══ B : AUTORISÉ ══════════════════════════════════
            elif dec >= 0:
                nom    = labels_map.get(dec, f"label_{dec}")
                pct    = max(0, int(100 - dec_conf))
                couleur = (40, 200, 60)
                texte   = f"{nom}  {pct}%"
                with _ev_lock:
                    _event.update({'type':'autorise','nom':nom,
                                   'heure':heure,'confiance':pct})
                _log_once(nom, 'AUTORISE', dec_conf, log_memo)

            # ══ C : INCONNU ═══════════════════════════════════
            else:
                couleur = (50, 130, 255)
                texte   = "INCONNU"
                with _ev_lock:
                    _event.update({'type':'inconnu','nom':'Inconnu',
                                   'heure':heure,'confiance':0})
                # Alerte : seulement si pas déjà en attente
                now = datetime.now()
                with _pending_lock:
                    deja_attente = len(_pending) > 0
                if not deja_attente and (now - alerte_ts) > delai_alerte and roi is not None:
                    threading.Thread(
                        target=envoyer_alerte_email,
                        args=(display.copy(), roi.copy()),
                        daemon=True).start()
                    alerte_ts = now
                    db_log('INCONNU', 'ALERTE')
                    _stats['inconnus'] += 1; _stats['total'] += 1

            # Dessin
            lbl_y = max(0, y - 34)
            cv2.rectangle(display, (x,y), (x+w,y+h), couleur, 2)
            cv2.rectangle(display, (x,lbl_y), (x+w,y), couleur, -1)
            cv2.putText(display, texte, (x+6, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255,255,255), 1, cv2.LINE_AA)

        # Overlay
        W, H = CFG['resolution']
        cv2.putText(display, heure, (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
        s = f"Modele OK | seuil={CFG['seuil']}" if modele_ok else "SANS MODELE"
        cv2.putText(display, s, (8, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (80,220,100) if modele_ok else (50,50,210), 1, cv2.LINE_AA)

        with _ann_lock:
            _ann_frame = display


def _log_once(nom, statut, conf, memo):
    now  = datetime.now()
    clef = f"{nom}_{statut}"
    if clef != memo['last'] or (now - memo['last_t']).seconds > 30:
        db_log(nom, statut, conf)
        if statut == 'AUTORISE': _stats['autorises'] += 1
        else:                    _stats['refuses']   += 1
        _stats['total'] += 1
        memo['last'] = clef; memo['last_t'] = now


# ════════════════════════════════════════════════════════════
#  STREAM MJPEG
# ════════════════════════════════════════════════════════════
def _stream():
    interval = 1.0 / CFG['stream_fps']
    params   = [cv2.IMWRITE_JPEG_QUALITY, CFG['stream_jpeg']]
    while True:
        t0 = time.time()
        with _ann_lock:
            f = _ann_frame
        if f is None:
            f = _placeholder("Demarrage...", "Patientez")
        ok, buf = cv2.imencode('.jpg', f, params)
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        time.sleep(max(0.0, interval - (time.time() - t0)))


# ════════════════════════════════════════════════════════════
#  EMAIL — ENVOI ALERTE (simple comme camex8)
# ════════════════════════════════════════════════════════════
def envoyer_alerte_email(frame_bgr: np.ndarray, gray_roi: np.ndarray):
    """
    Enregistre le visage inconnu et envoie un email simple à l'admin.
    gray_roi est sauvegardé dans _pending pour la mise à jour du modèle.
    """
    now  = datetime.now()
    eid  = now.strftime('%Y%m%d%H%M%S')
    path = f"logs/visages_temp/inconnu_{eid}.jpg"

    try:
        cv2.imwrite(path, frame_bgr)
    except Exception as e:
        print(f"  ✗ Sauvegarde image: {e}"); return

    with _pending_lock:
        _pending[eid] = {'chemin': path, 'gray_roi': gray_roi}

    try:
        msg            = MIMEMultipart('related')
        msg['From']    = EMAIL['expediteur']
        msg['To']      = EMAIL['destinataire']
        msg['Subject'] = f"Alerte Visage Inconnu - ID:{eid}"

        # Email HTML simple (style camex8)
        html = f"""<html><body style="font-family:Arial,sans-serif">
<h2 style="color:#c0392b">Visage Inconnu Detecte</h2>
<p><b>Date :</b> {now.strftime('%d/%m/%Y %H:%M:%S')}</p>
<p><b>ID :</b> {eid}</p>
<img src="cid:photo" style="width:400px;border:3px solid #c0392b">
<hr>
<p><b>Repondez a cet email :</b></p>
<p>✅ <b>ACCEPTER Prenom</b> — personne autorisee</p>
<p>❌ <b>REFUSER Prenom</b> — personne bloquee</p>
</body></html>"""

        alt = MIMEMultipart('alternative')
        alt.attach(MIMEText(
            f"Visage inconnu ID:{eid}. Repondez ACCEPTER Prenom ou REFUSER Prenom.", 'plain'))
        alt.attach(MIMEText(html, 'html'))
        msg.attach(alt)

        with open(path, 'rb') as f:
            att = MIMEImage(f.read())
            att.add_header('Content-ID', '<photo>')
            att.add_header('Content-Disposition', 'inline', filename='inconnu.jpg')
            msg.attach(att)

        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL['expediteur'], EMAIL['mot_de_passe'])
            smtp.send_message(msg)

        logging.info(f"Alerte email — ID:{eid}")
        print(f"  📧 Alerte envoyée (ID:{eid})")

    except Exception as e:
        logging.error(f"Email alerte: {e}")
        print(f"  ✗ Email alerte: {e}")
        # Nettoyer si envoi échoué
        with _pending_lock:
            _pending.pop(eid, None)


# ════════════════════════════════════════════════════════════
#  EMAIL — POLLING RÉPONSES ADMIN
#  Même logique que camex8.check_email_responses()
# ════════════════════════════════════════════════════════════
def thread_email_poll():
    print("  ✓ Thread email poll démarré")
    while _running:
        try:
            _verifier_inbox()
        except Exception as e:
            logging.error(f"Poll email: {e}")
            print(f"  ⚠ Poll email: {e}")
        time.sleep(CFG['email_poll_sec'])


def _verifier_inbox():
    with imaplib.IMAP4_SSL('imap.gmail.com') as mail:
        mail.login(EMAIL['expediteur'], EMAIL['mot_de_passe'])
        mail.select('inbox')
        status, msgs = mail.search(None, '(UNSEEN SUBJECT "Alerte Visage Inconnu")')
        if status != 'OK' or not msgs[0]:
            return

        for num in msgs[0].split():
            try:
                st, data = mail.fetch(num, '(RFC822)')
                if st != 'OK': continue

                emsg  = email_lib.message_from_bytes(data[0][1])
                sujet = emsg.get('subject', '')

                # Extraire EID du sujet
                mid = re.search(r'ID:(\d{14})', sujet)
                if not mid: continue
                eid = mid.group(1)

                # Extraire uniquement la réponse (ignorer texte cité)
                reponse = _extraire_reponse(emsg)
                print(f"  📩 Réponse reçue (EID:{eid}): '{reponse[:60]}'")

                upper = reponse.upper()
                if 'ACCEPTER' in upper:
                    m   = re.search(r'ACCEPTER\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_\-]+)',
                                    reponse, re.IGNORECASE)
                    nom = m.group(1).strip().capitalize() if m else None
                    if nom and nom.upper() not in ('PRENOM', 'NOM'):
                        _appliquer_decision(eid, nom, accepte=True)
                    else:
                        print(f"  ⚠ Nom invalide dans: '{reponse[:60]}'")

                elif 'REFUSER' in upper:
                    m   = re.search(r'REFUSER\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_\-]+)',
                                    reponse, re.IGNORECASE)
                    nom = (m.group(1).strip().capitalize()
                           if m else f"Refuse_{eid[-6:]}")
                    _appliquer_decision(eid, nom, accepte=False)

            except Exception as e:
                logging.error(f"Traitement email: {e}")
                print(f"  ✗ Traitement email: {e}")
            finally:
                try: mail.store(num, '+FLAGS', '\\Seen')
                except: pass


def _extraire_reponse(msg) -> str:
    """
    Extrait uniquement la réponse de l'admin,
    avant le texte cité par le client mail.
    Identique à camex8._extract_email_body() mais avec filtre de citation.
    """
    texte = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                try:
                    texte = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
                except: pass
    else:
        try:
            texte = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        except:
            texte = str(msg.get_payload() or '')

    # Garder uniquement les lignes avant le texte cité
    lignes = []
    for ligne in texte.splitlines():
        l = ligne.strip()
        if l.startswith('>'): break
        if re.match(r'^(Le |On ).{5,50}(a écrit|wrote)\s*:', l): break
        if re.match(r'^-{3,}', l): break
        if re.match(r'^_{3,}', l): break
        lignes.append(ligne)

    return '\n'.join(lignes).strip()


# ════════════════════════════════════════════════════════════
#  DÉCISION ADMIN — cœur du système
#
#  Inspiré de camex8._process_face() :
#    1. Sauvegarder image dans authorized_temp / denied_temp
#       avec le nom donné par l'admin (comme camex8 fait f"{name}.jpg")
#    2. Appeler _update_modele_en_memoire() → update immédiat
#       (= camex8: known_face_encodings.append(face_encoding))
#    3. Email de confirmation
#    4. Notification dashboard
# ════════════════════════════════════════════════════════════
def _appliquer_decision(eid: str, nom: str, accepte: bool):
    global _stats

    with _pending_lock:
        if eid not in _pending:
            print(f"  ⚠ EID {eid} déjà traité ou système redémarré")
            return
        chemin   = _pending[eid]['chemin']
        gray_roi = _pending[eid].get('gray_roi')

    action = "ACCEPTÉ" if accepte else "REFUSÉ"
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n  → Décision admin : {nom} {action} (EID:{eid})")

    try:
        # 1. Copier l'image dans authorized_temp ou denied_temp
        #    Nom du fichier = nom donné par l'admin (comme camex8 : f"{name}.jpg")
        dst_dir = ('data/authorized/authorized_temp'
                   if accepte else 'data/denied/denied_temp')
        os.makedirs(dst_dir, exist_ok=True)
        # Nom = prenom_timestamp.jpg (évite les écrasements)
        dst = os.path.join(dst_dir, f"{nom}_{ts}.jpg")

        if os.path.exists(chemin):
            shutil.copy2(chemin, dst)
            print(f"  ✓ Image copiée → {dst}")
        else:
            print(f"  ⚠ Image source introuvable: {chemin}")

        # 2. Mise à jour IMMÉDIATE du modèle en mémoire
        #    (comme camex8: known_face_encodings.append(face_encoding))
        _update_modele_en_memoire(
            nom, 'autorise' if accepte else 'refuse', gray_roi)

        # 3. Log + stats
        db_log(nom, f"ADMIN_{'ACCEPTE' if accepte else 'REFUSE'}")
        logging.info(f"Admin: {nom} {action} (EID:{eid})")

        # 4. Notification dashboard
        push_notif(
            'success' if accepte else 'danger',
            f"{'✅' if accepte else '❌'} {nom} {action}",
            (f"Capture → authorized_temp/{nom}_{ts}.jpg. "
             if accepte else
             f"Capture → denied_temp/{nom}_{ts}.jpg. ") +
            "Reconnaissance mise à jour. ⚠ Recapturez + train.py pour persistance."
        )

        print(f"  ✓ {nom} {action} — stream mis à jour immédiatement\n")

        # 5. Email de confirmation (thread séparé)
        threading.Thread(
            target=_envoyer_confirmation,
            args=(nom, accepte, dst),
            daemon=True).start()

    except Exception as e:
        logging.error(f"Décision {eid}: {e}")
        print(f"  ✗ Erreur décision: {e}")
    finally:
        # Supprimer de _pending dans tous les cas
        with _pending_lock:
            entry = _pending.pop(eid, None)
            if entry:
                try: os.remove(entry['chemin'])
                except: pass


def _envoyer_confirmation(nom: str, accepte: bool, chemin_dst: str):
    """Email de confirmation simple (inspiré de camex8._send_confirmation())."""
    try:
        action = "ACCEPTÉ" if accepte else "REFUSÉ"
        icone  = "✅" if accepte else "❌"

        msg            = MIMEMultipart('alternative')
        msg['From']    = EMAIL['expediteur']
        msg['To']      = EMAIL['destinataire']
        msg['Subject'] = f"{icone} Confirmation — {nom} {action}"

        html = f"""<html><body style="font-family:Arial,sans-serif">
<h2 style="color:{'#27ae60' if accepte else '#c0392b'}">{icone} {nom} {action}</h2>
<p>La reconnaissance a ete mise a jour immediatement.</p>
<p><b>{nom}</b> est desormais {'autorise' if accepte else 'bloque'} sur le flux video.</p>
<p>Capture enregistree :<br><code>{chemin_dst}</code></p>
<hr>
<p><b>Pour meilleure precision :</b><br>
<code>python3 capture.py</code> puis <code>python3 train.py</code></p>
</body></html>"""

        msg.attach(MIMEText(
            f"{nom} {action}. Reconnaissance mise a jour. Fichier: {chemin_dst}", 'plain'))
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL['expediteur'], EMAIL['mot_de_passe'])
            smtp.send_message(msg)

        print(f"  📧 Confirmation envoyée : {nom} {action}")
        logging.info(f"Confirmation: {nom} {action}")

    except Exception as e:
        logging.error(f"Email confirmation: {e}")
        print(f"  ✗ Email confirmation: {e}")


# ════════════════════════════════════════════════════════════
#  ROUTES FLASK
# ════════════════════════════════════════════════════════════
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with _ev_lock: return jsonify(_event.copy())

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'total_tentatives': _stats['total'],
        'autorises':        _stats['autorises'],
        'refuses':          _stats['refuses'],
        'inconnus':         _stats['inconnus'],
    })

@app.route('/api/logs')
def api_logs():
    try:
        c = sqlite3.connect(DB)
        rows = c.execute(
            'SELECT id,horodatage,nom,statut,confiance FROM acces ORDER BY id DESC LIMIT 100'
        ).fetchall()
        c.close()
        return jsonify([{'id':r[0],'heure':r[1],'nom':r[2],'statut':r[3],'confiance':r[4]}
                        for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/personnes')
def api_personnes():
    def ls(d):
        if not os.path.exists(d): return []
        return [x for x in os.listdir(d)
                if os.path.isdir(os.path.join(d, x))
                and x not in ('authorized_temp', 'denied_temp')]
    with _model_lock:
        auto = [v for v in _labels_map.values() if v not in _personnes_ref]
        ref  = list(_personnes_ref)
    return jsonify({'autorisees': auto, 'refusees': ref})

@app.route('/api/notifications')
def api_notifications():
    with _notifs_lock: return jsonify(list(_notifs))

@app.route('/api/notifications/clear', methods=['POST'])
def api_clear_notifs():
    with _notifs_lock: _notifs.clear()
    return jsonify({'ok': True})

@app.route('/api/health')
def api_health():
    with _model_lock:
        ok = _modele_ok
    return jsonify({'status': 'ok', 'camera': camera_ok,
                    'modele': ok, 'heure': datetime.now().strftime('%H:%M:%S')})




# ════════════════════════════════════════════════════════════
#  ROUTES ADMIN — Capture & Entraînement sans écran
# ════════════════════════════════════════════════════════════

_cap = {'active':False,'nom':'','type':'','compteur':0,'objectif':50,
        'lock':threading.Lock()}

@app.route('/api/admin/capture/start', methods=['POST'])
def api_capture_start():
    from flask import request
    d    = request.get_json(silent=True) or {}
    nom  = (d.get('nom') or '').strip().capitalize()
    typ  = d.get('type','authorized')
    obj  = int(d.get('objectif', 50))
    if not nom:
        return jsonify({'ok':False,'msg':'Nom manquant'}), 400
    with _cap['lock']:
        if _cap['active']:
            return jsonify({'ok':False,'msg':f"Session en cours : {_cap['nom']}"}), 409
        _cap.update({'active':True,'nom':nom,'type':typ,'compteur':0,'objectif':obj})
    push_notif('info',f"Capture : {nom}",f"Placez {nom} devant la camera. Objectif : {obj} photos.")
    return jsonify({'ok':True,'msg':f"Capture demarree pour {nom}"})

@app.route('/api/admin/capture/snap', methods=['POST'])
def api_capture_snap():
    with _cap['lock']:
        if not _cap['active']:
            return jsonify({'ok':False,'msg':'Aucune session'}), 400
        nom=_cap['nom']; typ=_cap['type']; cpt=_cap['compteur']; obj=_cap['objectif']
    with _ann_lock:
        frame = _ann_frame
    if frame is None:
        return jsonify({'ok':False,'msg':'Camera non prete'}), 503
    det = cv2.CascadeClassifier(HAAR_PATH)
    try:
        g = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = det.detectMultiScale(g, 1.15, 5, minSize=(70,70))
    except:
        return jsonify({'ok':False,'msg':'Erreur detection'}), 500
    if len(faces) == 0:
        return jsonify({'ok':False,'msg':'Aucun visage — approchez-vous'}), 400
    x,y,w,h = faces[0]
    roi = cv2.resize(cv2.equalizeHist(g[y:y+h, x:x+w]), (200,200))
    dossier = os.path.join(f"data/{typ}", nom)
    os.makedirs(dossier, exist_ok=True)
    cv2.imwrite(os.path.join(dossier, f"{nom}_{cpt:04d}.jpg"), roi)
    with _cap['lock']:
        _cap['compteur'] += 1
        n = _cap['compteur']
    if n >= obj:
        with _cap['lock']: _cap['active'] = False
        push_notif('success',f"Capture terminee : {nom}",
                   f"{obj} photos. Lancez l'entrainement.")
        return jsonify({'ok':True,'compteur':n,'termine':True,
                        'msg':f"Termine ({obj} photos) — lancez Train"})
    return jsonify({'ok':True,'compteur':n,'termine':False,'msg':f"Photo {n}/{obj}"})

@app.route('/api/admin/capture/stop', methods=['POST'])
def api_capture_stop():
    with _cap['lock']:
        nom=_cap['nom']; nb=_cap['compteur']; _cap['active']=False
    push_notif('warning',f"Capture arretee : {nom}",f"{nb} photos.")
    return jsonify({'ok':True,'msg':f"Arretee ({nb} photos)"})

@app.route('/api/admin/capture/status')
def api_capture_status():
    with _cap['lock']:
        return jsonify({k:v for k,v in _cap.items() if k!='lock'})

@app.route('/api/admin/train', methods=['POST'])
def api_train():
    def _do():
        try:
            push_notif('info','Entrainement en cours...','Patientez 30-60 secondes.')
            import subprocess, sys
            r = subprocess.run([sys.executable,'train.py'],
                               capture_output=True, text=True, timeout=180,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
            if r.returncode == 0:
                _charger_modele()
                push_notif('success','Entrainement termine',
                           'Modele rechargé. Nouvelles donnees actives.')
                logging.info("Train dashboard OK")
            else:
                push_notif('danger','Erreur entrainement', r.stderr[:300])
                logging.error(f"Train: {r.stderr}")
        except Exception as e:
            push_notif('danger','Erreur entrainement', str(e))
    threading.Thread(target=_do, daemon=True).start()
    return jsonify({'ok':True,'msg':'Entrainement lance'})

# ════════════════════════════════════════════════════════════
#  LANCEMENT
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import socket

    print()
    print("=" * 60)
    print("  CONTRÔLE D'ACCÈS — v8 FINAL")
    print("  Fadi Hammemi · ENIT · Raspberry Pi 3B+")
    print("=" * 60)
    print(f"  Seuil LBPH   : {CFG['seuil']}  (↑ si faux INCONNU, ↓ si faux positif)")
    print(f"  Stabilisateur: {CFG['votes']} votes | {int(CFG['ratio']*100)}%")
    print(f"  Stream       : {CFG['stream_fps']} FPS · JPEG q{CFG['stream_jpeg']}")
    print(f"  Email        : {EMAIL['destinataire']}")
    print(f"  Poll inbox   : toutes les {CFG['email_poll_sec']}s")

    init_db()

    if not (EMAIL['expediteur'] and EMAIL['mot_de_passe'] and EMAIL['destinataire']):
        print("  ⚠ Email non configuré : copiez .env.example en .env et remplissez-le")
        print("    (les alertes email et la validation par l'admin seront inactives)")

    # Charger le modèle en mémoire partagée
    _charger_modele()

    print()
    threading.Thread(target=thread_capture,
                     daemon=True, name="T1-Capture").start()
    threading.Thread(target=thread_reconnaissance,
                     daemon=True, name="T2-Reco").start()
    threading.Thread(target=thread_email_poll,
                     daemon=True, name="T3-EmailPoll").start()

    for _ in range(24):
        if camera_ok: break
        time.sleep(0.5)

    try:   ip = socket.gethostbyname(socket.gethostname())
    except: ip = "127.0.0.1"

    print()
    print("=" * 60)
    print(f"  ✓ Système prêt !")
    print(f"  🖥  http://{ip}:{CFG['port']}  (PC)")
    print(f"  🖥  http://localhost:{CFG['port']}  (Pi)")
    print(f"  📧  Répondez :  ACCEPTER Prénom  ou  REFUSER Prénom")
    print("=" * 60)
    print()

    app.run(host='0.0.0.0', port=CFG['port'],
            debug=False, threaded=True, use_reloader=False)
