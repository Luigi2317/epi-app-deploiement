"""
Construire le corpus d'evenements qui alimente le tableau de bord.

Ce que ce fichier produit, et ce qu'il ne produit pas
-----------------------------------------------------
Le tableau de bord demande des indicateurs, une chronologie, des filtres par
periode et par camera. Or le systeme n'a jamais tourne en production : il
n'existe AUCUN historique d'alertes, aucune date, aucune camera.

On ne peut pas inventer cela en silence. Le corpus separe donc strictement
deux natures d'information :

    REEL, produit par le modele sur de vraies images
        les detections, les verdicts, les confiances, la position et la
        taille de chaque personne dans le champ, son statut de surveillance

    SIMULE, attribue par ce script et DECLARE comme tel
        l'horodatage, la camera, la zone

Les chiffres qui engagent, taux de detection, volume d'alertes, part de
personnes non jugeables : sont donc mesures. Seul le contexte de deploiement
est reconstitue, et l'application l'affiche en clair.

    Un tableau de bord entierement fictif serait indefendable.
    Un tableau de bord qui declare ce qui est simule ne l'est pas.

La colonne qui vaut de l'or
---------------------------
`x_relatif` et `y_relatif` donnent la position de chaque personne DANS LE
CHAMP, en fraction de l'image. C'est du reel, et c'est ce qui permet la
carte de chaleur du cadrage : le J12 a mesure que 94 % des fausses alertes
viennent du haut du cadre (tetes coupees) et du fond (personnes trop
petites). La carte le rend visible, et elle dit ou repositionner la camera.

Usage :
    python src/app/corpus.py                          les 1 454 images
    python src/app/corpus.py --limite 200             un echantillon
    python src/app/corpus.py --source data/hors_domaine
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.app.detection import Detecteur, Statut          # noqa: E402

SORTIE = RACINE / "resultats" / "tableau_de_bord"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SURVEILLES = ["helmet", "glasses", "gloves", "safety-vest"]

# --------------------------------------------------------------------------
# CONTEXTE SIMULE : tout ce qui suit est invente, et declare comme tel.
#
# Quatre cameras, une par zone type d'un chantier. Le choix des zones vient
# des personas du J6, pas d'un tirage au hasard : ce sont les endroits ou un
# responsable HSE veut savoir ce qui se passe.
# --------------------------------------------------------------------------
CAMERAS = [
    {"identifiant": "CAM-01", "zone": "Acces chantier"},
    {"identifiant": "CAM-02", "zone": "Zone de levage"},
    {"identifiant": "CAM-03", "zone": "Coffrage niveau 2"},
    {"identifiant": "CAM-04", "zone": "Aire de stockage"},
]

# Fenetre simulee : 14 jours, en journee de travail seulement.
DEBUT = datetime(2026, 8, 10, 7, 0, 0)
JOURS = 14
HEURE_DEBUT, HEURE_FIN = 7, 18


def horodatage_simule(rang: int, total: int) -> datetime:
    """
    Repartit les evenements sur la fenetre, en heures ouvrees uniquement.

    Un chantier ne travaille pas la nuit : etaler uniformement sur 24 h
    produirait une chronologie que personne ne croirait, et fausserait la
    lecture des pointes d'activite.
    """
    heures_par_jour = HEURE_FIN - HEURE_DEBUT
    position = rang / max(1, total)
    minutes_totales = position * JOURS * heures_par_jour * 60
    jour = int(minutes_totales // (heures_par_jour * 60))
    reste = minutes_totales % (heures_par_jour * 60)
    return DEBUT + timedelta(days=jour, hours=HEURE_DEBUT - DEBUT.hour,
                             minutes=reste)


ENTETE = [
    # --- contexte SIMULE ---
    "horodatage", "camera", "zone",
    # --- reel, produit par le modele ---
    "image", "identifiant", "statut", "alerte",
    "x_relatif", "y_relatif", "hauteur_relative", "confiance_personne",
    *[f"conf_{c}" for c in SURVEILLES],
]


def evenements(detecteur: Detecteur, fichiers: list[Path]):
    """Une ligne par personne detectee. Genere au fil de l'eau."""
    import cv2

    for rang, chemin in enumerate(fichiers):
        image = cv2.imread(str(chemin))
        if image is None:
            continue
        resultat = detecteur.analyser_image(image)
        camera = CAMERAS[rang % len(CAMERAS)]
        instant = horodatage_simule(rang, len(fichiers))

        for personne in resultat.personnes:
            x1, y1, x2, y2 = personne.boite
            yield {
                "horodatage": instant.isoformat(timespec="seconds"),
                "camera": camera["identifiant"],
                "zone": camera["zone"],
                "image": chemin.name,
                "identifiant": personne.identifiant,
                "statut": personne.statut.value,
                "alerte": int(personne.manque_casque),
                "x_relatif": round((x1 + x2) / 2 / resultat.largeur, 4),
                "y_relatif": round((y1 + y2) / 2 / resultat.hauteur, 4),
                "hauteur_relative": round((y2 - y1) / resultat.hauteur, 4),
                "confiance_personne": round(personne.confiance, 4),
                **{f"conf_{c}": round(personne.equipements.get(c, 0.0), 4)
                   for c in SURVEILLES},
            }


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--source", default="data/echantillon_app")
    a.add_argument("--modele", default="yolov8m")
    a.add_argument("--limite", type=int, default=None)
    a.add_argument("--sortie", default="evenements",
                   help="nom du corpus, sans extension : permet de garder plusieurs corpus cote a cote")
    args = a.parse_args()

    source = RACINE / args.source
    fichiers = sorted(f for f in source.iterdir()
                      if f.suffix.lower() in EXTENSIONS)
    if args.limite:
        fichiers = fichiers[:args.limite]
    if not fichiers:
        print(f"  aucune image dans {source}")
        return 1

    SORTIE.mkdir(parents=True, exist_ok=True)
    fichier = SORTIE / f"{args.sortie}.csv"

    detecteur = Detecteur(poids=RACINE / "models" / f"{args.modele}.pt")
    print(f"\n  {len(fichiers)} images · modele {args.modele}")
    print(f"  contexte simule : {len(CAMERAS)} cameras, {JOURS} jours, "
          f"{HEURE_DEBUT}h-{HEURE_FIN}h\n")

    depart = time.perf_counter()
    compte = {s.value: 0 for s in Statut}
    lignes = 0

    with fichier.open("w", encoding="utf-8", newline="") as flux:
        redacteur = csv.DictWriter(flux, fieldnames=ENTETE)
        redacteur.writeheader()
        for evenement in evenements(detecteur, fichiers):
            redacteur.writerow(evenement)
            compte[evenement["statut"]] += 1
            lignes += 1
            if lignes % 500 == 0:
                print(f"    {lignes} personnes…")

    duree = time.perf_counter() - depart
    surveillees = compte[Statut.SURVEILLEE.value]
    alertes = sum(1 for _ in [])  # recompte ci-dessous, sans tout garder

    with fichier.open(encoding="utf-8", newline="") as flux:
        alertes = sum(int(l["alerte"]) for l in csv.DictReader(flux))

    print(f"\n  {lignes} personnes sur {len(fichiers)} images "
          f", {duree:.0f} s\n")
    print(f"    surveillees        {surveillees:6d}")
    print(f"    hors perimetre     {compte[Statut.HORS_PERIMETRE.value]:6d}")
    print(f"    tete hors champ    {compte[Statut.TETE_HORS_CHAMP.value]:6d}")
    print(f"    ALERTES            {alertes:6d}"
          f"   ({alertes / surveillees * 100:.1f} % des surveillees)"
          if surveillees else "")
    print(f"\n  Ecrit : {fichier}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
