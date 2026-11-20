#!/usr/bin/env python3
"""
ÉTAPE 2 — Entraînement du modèle LBPH
Projet de Fin d'Année — ENIT — Fadi Hammemi

Usage: python3 train.py
Durée: ~30 secondes pour 50 photos × 2 personnes

SCÉNARIOS SUPPORTÉS:
  • Personnes autorisées uniquement
  • Personnes refusées uniquement   (test du scénario refus)
  • Mix autorisées + refusées        (production)
"""

import cv2, os, pickle, numpy as np, time

DOSSIER_AUTO  = "data/authorized"
DOSSIER_REFUS = "data/denied"
MODEL_YML     = "models/model.yml"
MODEL_PKL     = "models/labels.pkl"
TAILLE        = (200, 200)


def compter_images(dossier: str) -> int:
    total = 0
    if os.path.exists(dossier):
        for p in os.listdir(dossier):
            d = os.path.join(dossier, p)
            if os.path.isdir(d) and p not in ('authorized_temp', 'denied_temp'):
                total += sum(1 for f in os.listdir(d) if f.lower().endswith('.jpg'))
    return total


def charger_dossier(dossier: str, label_debut: int,
                    exclure: tuple = ()) -> tuple:
    """
    Charge toutes les images d'un dossier.
    Retourne (visages, labels, mapping {label: nom}).
    """
    visages, labels, mapping = [], [], {}
    label = label_debut

    if not os.path.exists(dossier):
        return visages, labels, mapping

    for nom in sorted(os.listdir(dossier)):
        chemin_personne = os.path.join(dossier, nom)
        if not os.path.isdir(chemin_personne) or nom in exclure:
            continue

        images = [f for f in os.listdir(chemin_personne)
                  if f.lower().endswith('.jpg')]
        if not images:
            print(f"  ⚠ {nom} : aucune image, ignoré")
            continue

        mapping[label] = nom
        nb = 0
        for fichier in images:
            img = cv2.imread(os.path.join(chemin_personne, fichier),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, TAILLE)
            img = cv2.equalizeHist(img)
            visages.append(img)
            labels.append(label)
            nb += 1

        print(f"  ✓ {nom} : {nb} photo(s) — label #{label}")
        label += 1

    return visages, labels, mapping


def main():
    print()
    print("=" * 55)
    print("  ENTRAÎNEMENT DU MODÈLE — LBPH")
    print("=" * 55)

    os.makedirs("models", exist_ok=True)

    # ── Comptage ─────────────────────────────────────────────
    nb_auto  = compter_images(DOSSIER_AUTO)
    nb_refus = compter_images(DOSSIER_REFUS)

    print(f"\n  Images trouvées :")
    print(f"    Autorisées : {nb_auto}")
    print(f"    Refusées   : {nb_refus}")
    print(f"    Total      : {nb_auto + nb_refus}")

    if nb_auto == 0 and nb_refus == 0:
        print("\n✗ ERREUR : Aucune image trouvée.")
        print("  → Lancez d'abord : python3 capture.py")
        # Supprimer l'ancien modèle obsolète
        for f in [MODEL_YML, MODEL_PKL]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  ⚠ Ancien modèle supprimé : {f}")
        return

    if nb_auto == 0:
        print("\n  ⚠ ATTENTION : Aucune personne autorisée.")
        print("    Le système refusera les personnes connues et signalera")
        print("    tout autre visage comme INCONNU. (Mode test scénario refus)")

    # ── Chargement autorisés ──────────────────────────────────
    print("\n[1/2] Chargement personnes autorisées...")
    v_auto, l_auto, map_auto = charger_dossier(
        DOSSIER_AUTO, label_debut=0,
        exclure=('authorized_temp',))

    # ── Chargement refusés ────────────────────────────────────
    print("\n[2/2] Chargement personnes refusées...")
    label_debut_refus = max(map_auto.keys(), default=-1) + 1
    v_ref, l_ref, map_ref = charger_dossier(
        DOSSIER_REFUS, label_debut=label_debut_refus,
        exclure=('denied_temp',))

    # ── Fusion ────────────────────────────────────────────────
    tous_visages = v_auto + v_ref
    tous_labels  = l_auto + l_ref
    labels_map   = {**map_auto, **map_ref}
    personnes_refusees = list(map_ref.values())

    if not tous_visages:
        print("\n✗ ERREUR : Aucune image valide chargée.")
        return

    print(f"\n  Total : {len(tous_visages)} image(s) | {len(labels_map)} personne(s)")
    print(f"  Autorisées : {list(map_auto.values())}")
    print(f"  Refusées   : {personnes_refusees}")

    # ── Entraînement LBPH ────────────────────────────────────
    print("\n  Entraînement en cours...")
    debut = time.time()

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(tous_visages, np.array(tous_labels))
    recognizer.save(MODEL_YML)

    data = {
        'labels':  labels_map,
        'denied':  personnes_refusees,
        'nb_images': len(tous_visages),
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(data, f)

    duree = time.time() - debut
    print(f"\n{'='*55}")
    print(f"  ✓ Modèle entraîné en {duree:.1f}s")
    print(f"  ✓ {MODEL_YML}")
    print(f"  ✓ {MODEL_PKL}")
    print(f"\n  Prochaine étape : python3 app.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
