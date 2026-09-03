"""
Expliquer un verdict, detection par detection.

Pourquoi cet outil existe
-------------------------
Quand le systeme se trompe, l'ecran ne dit pas POURQUOI. Une personne
signalee « casque non detecte » alors qu'elle en porte un peut relever de
trois causes tres differentes :

    1. le casque n'a pas ete detecte du tout
       -> le modele est en cause, aucun reglage ne le corrige

    2. il a ete detecte, mais sous le seuil calibre de sa classe
       -> question de calibrage, donc de compromis assume

    3. il a ete detecte au-dessus du seuil, mais attribue a QUELQU'UN
       D'AUTRE, ou refuse par la contrainte de hauteur
       -> defaut de la regle metier, donc de notre code

Les trois produisent le meme affichage. Sans cet outil, on choisit une
explication au lieu de la constater, et c'est exactement l'erreur commise
le 20 aout sur les cellules vides d'un tableau, ou deux conclusions
opposees ont ete tirees successivement de la meme donnee ambigue.

Il sert aussi de socle au « drill-down » demande par la grille (C3.2-3) :
partir d'un chiffre agrege et descendre jusqu'au cas individuel.

Usage :
    python src/app/expliquer.py data/demonstration/extrait_0060.jpg
    python src/app/expliquer.py image.jpg --part-haute 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.app.detection import Detecteur, charger_seuils    # noqa: E402
from src.app.regles import (                                              # noqa: E402
    EQUIPEMENTS_TETE,
    PART_HAUTE,
    centre,
    dans_la_boite,
    dans_la_zone_haute,
)

SURVEILLES = ["helmet", "glasses", "gloves", "safety-vest"]


def aire(boite) -> float:
    return float((boite[2] - boite[0]) * (boite[3] - boite[1]))


def expliquer(chemin: Path, part_haute: float = PART_HAUTE,
              modele: str = "yolov8m") -> None:
    import cv2

    image = cv2.imread(str(chemin))
    if image is None:
        print(f"  image illisible : {chemin}")
        return

    detecteur = Detecteur(poids=RACINE / "models" / f"{modele}.pt")
    seuils = charger_seuils(modele)
    sortie = detecteur.modele.predict(image, conf=detecteur.confiance_brute,
                                      verbose=False)[0]
    personnes, equipements = detecteur._separer(sortie)

    hauteur, largeur = sortie.orig_shape
    print(f"\n  {chemin.name}, {largeur} x {hauteur} px")
    print(f"  seuil de detection brute {detecteur.confiance_brute:.4f} · "
          f"part haute {part_haute:.0%}\n")

    # ------------------------------------------------- 1. ce que le modele voit
    par_classe: dict[str, list[float]] = {}
    for e in equipements:
        par_classe.setdefault(e["classe"], []).append(e["confiance"])

    print("  CE QUE LE MODELE PROPOSE, avant toute regle")
    print(f"      {'person':14s} {len(personnes):3d}")
    for classe in sorted(par_classe, key=lambda c: -len(par_classe[c])):
        confiances = sorted(par_classe[classe], reverse=True)
        seuil = seuils.get(classe)
        au_dessus = (sum(1 for c in confiances if c >= seuil)
                     if seuil else None)
        detail = f"  dont {au_dessus} au-dessus du seuil {seuil:.3f}" if seuil else ""
        apercu = ", ".join(f"{c:.2f}" for c in confiances[:6])
        print(f"      {classe:14s} {len(confiances):3d}{detail}")
        print(f"      {'':14s}     confiances : {apercu}")

    if not personnes:
        print("\n  Aucune personne detectee : aucun verdict possible.\n")
        return

    # -------------------------------------- 2. le sort de chaque equipement
    print(f"\n  LE SORT DE CHAQUE EQUIPEMENT SURVEILLE")
    orphelins = 0
    for e in sorted(equipements, key=lambda x: -x["confiance"]):
        classe = e["classe"]
        if classe not in SURVEILLES:
            continue
        seuil = seuils.get(classe, 0.5)
        point = centre(e["boite"])
        tete = classe in EQUIPEMENTS_TETE

        contenants, refuses_hauteur = [], []
        for p in personnes:
            if not dans_la_boite(point, p["boite"]):
                continue
            if tete and not dans_la_zone_haute(point, p["boite"], part_haute):
                refuses_hauteur.append(p["identifiant"])
                continue
            contenants.append((abs(point[1] - p["boite"][1]), p["identifiant"]))

        etat = "RETENU " if e["confiance"] >= seuil else "sous seuil"
        print(f"\n      {classe} {e['confiance']:.3f}  [{etat}]  "
              f"centre ({point[0]:.0f}, {point[1]:.0f})")

        if contenants:
            _, gagnant = min(contenants)
            autres = [i for _, i in sorted(contenants) if i != gagnant]
            print(f"          -> attribue a la personne #{gagnant}")
            if autres:
                print(f"          (etait aussi dans les boites {autres}, "
                      f"depart au haut de boite le plus proche)")
        elif refuses_hauteur:
            orphelins += 1
            print(f"          -> REFUSE : hors du haut de boite des personnes "
                  f"{refuses_hauteur}")
            print(f"          (assouplir --part-haute le rattacherait)")
        else:
            orphelins += 1
            print(f"          -> ORPHELIN : dans aucune boite de personne")

    # ------------------------------------------------- 3. verdict par personne
    resultat = detecteur.analyser_image(image)
    print(f"\n  VERDICT PAR PERSONNE")
    for p in sorted(resultat.personnes, key=lambda x: -aire(x.boite)):
        surface = aire(p.boite) / (largeur * hauteur) * 100
        marque = "!" if p.manque_casque else " "
        print(f"    {marque} #{p.identifiant:<3d} {surface:5.1f} % de l'image"
              f"   personne {p.confiance:.2f}")
        for classe in SURVEILLES:
            confiance = p.equipements.get(classe, 0.0)
            vu = p.verdicts.get(classe, False)
            if confiance > 0 or vu:
                print(f"          {classe:12s} {confiance:.3f}  "
                      f"{'retenu' if vu else 'sous le seuil'}")

    sans = resultat.nombre_sans_casque
    print(f"\n  {len(resultat.personnes)} personnes · "
          f"{len(resultat.personnes) - sans} avec casque · {sans} sans · "
          f"{orphelins} equipements non rattaches\n")


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("image")
    a.add_argument("--part-haute", type=float, default=PART_HAUTE)
    a.add_argument("--modele", default="yolov8m")
    args = a.parse_args()

    chemin = Path(args.image)
    if not chemin.is_file():
        print(f"  introuvable : {chemin}")
        return 1
    expliquer(chemin, args.part_haute, args.modele)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
