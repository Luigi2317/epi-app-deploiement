"""
Regles metier : relier un equipement a une personne.

Pourquoi ce module existe
-------------------------
Le detecteur ne dit pas « cette personne porte un casque ». Il dit :

    « rectangle (340, 120, 420, 180), casque, 0,73 »
    « rectangle (300,  90, 520, 700), personne, 0,91 »

Deux boites, aucun lien. C'est ce module qui decide si le casque appartient
a cette personne — et c'est une REGLE METIER, ecrite a la main, pas une
sortie du modele.

    Le modele detecte des objets. La regle produit le verdict.

C'est la phrase la plus importante a tenir devant un jury : elle separe ce
que l'apprentissage a produit de ce que l'ingenieur a decide.

La regle retenue
----------------
Un casque appartient a une personne si :

    1. le CENTRE du casque tombe dans la boite de la personne
    2. et il se situe dans le TIERS SUPERIEUR de cette boite

Sur une personne dont la boite va de y=100 (tete) a y=700 (pieds) :

    hauteur = 600 px
    zone acceptee : de 100 a 100 + 0,35 x 600 = 310

    casque a y=150  ->  dans la zone  ->  attribue
    casque a y=450  ->  hors zone     ->  refuse (c'est un casque tenu
                                          a la main, ou celui d'un voisin)

Pourquoi le tiers superieur et non le quart
--------------------------------------------
Un casque se porte sur la tete, donc tres haut. Un seuil severe (20 %)
refuserait des cas legitimes : personne penchee, accroupie, vue en plongee
— or le J5 a montre que les cameras seront souvent en hauteur. Un seuil
laxiste (50 %) attribuerait a une personne le casque de celle qui se tient
derriere elle.

35 % est un compromis, et il est REGLABLE : `PART_HAUTE`. Il n'a pas ete
calibre sur des donnees, faute d'annotations reliant casques et personnes
dans SH17 — cette limite doit etre declaree.

Le cas des personnes qui se chevauchent
----------------------------------------
Un casque peut tomber dans DEUX boites de personnes superposees. Il est
alors attribue a celle dont le HAUT de boite est le plus proche du casque :
c'est la personne sur la tete de qui il repose le plus vraisemblablement.

La confusion connue, et ce qu'elle impose
------------------------------------------
La matrice de confusion du J7 a mesure 44 mains etiquetees « gants », et le
test hors domaine du J9 a confirme une sur-detection de 24 %. Un systeme de
detection d'ABSENCE qui voit des gants sur des mains nues ne declenche pas
d'alerte : c'est une alerte ratee, silencieuse.

Ce module ne corrige pas cette confusion — elle est dans le modele. Il la
DOCUMENTE, pour que le tableau de bord puisse afficher un niveau de
confiance different selon la classe.
"""

from __future__ import annotations

import numpy as np

# Part haute de la boite personne dans laquelle un equipement de tete est
# accepte. Voir l'en-tete pour la justification et la limite.
PART_HAUTE = 0.35

# Equipements portes sur la tete : la contrainte de hauteur s'y applique.
EQUIPEMENTS_TETE = {"helmet", "glasses", "face-mask", "ear-mufs", "face-guard"}

# Equipements portes ailleurs : seule l'inclusion dans la boite compte.
EQUIPEMENTS_CORPS = {"safety-vest", "gloves", "safety-suit", "medical-suit", "shoes"}


# --------------------------------------------------------------------------
# PERIMETRE DE SURVEILLANCE — ajoute le 24 aout, apres mesure.
#
# Sur quatre images d'une video de chantier ou les douze personnes portent
# un casque, le systeme produisait 16 fausses alertes sur 23 personnes
# detectees. Chacune a ete attribuee a une cause :
#
#     trop LOIN — moins de 20 % de la hauteur d'image ....  9   56 %
#     trop PRES — tete coupee par le bord superieur ......  6   38 %
#     autre .............................................  1    6 %
#
# LES DEUX CAUSES DOMINANTES SONT LA MEME : LA CAMERA EST MAL PLACEE.
# Elle filme a hauteur d'homme, au milieu de la scene. Ceux du premier plan
# ont la tete hors champ, ceux du fond font cent pixels.
#
# Ni le modele, ni les seuils, ni la logique de decision ne sont en cause.
# Les deux regles ci-dessous ne corrigent donc pas une erreur : elles
# DECLARENT ce que le systeme peut honnetement juger.
# --------------------------------------------------------------------------

# Hauteur minimale d'une personne, en fraction de la hauteur de l'image.
# Mesure : les personnes correctement jugees font 28 % ou plus ; celles qui
# produisaient des fausses alertes font 13 % ou moins. Le seuil de 20 %
# separe les deux groupes sans couper au milieu d'aucun.
HAUTEUR_MINIMALE = 0.20

# Distance au bord superieur, en pixels, en deca de laquelle on considere
# que la tete peut etre hors champ. Mesure : six des sept echecs sur des
# personnes de grande taille avaient leur boite a 3 pixels ou moins du bord.
MARGE_BORD = 5


def hors_perimetre(boite: np.ndarray, hauteur_image: int,
                   minimum: float = HAUTEUR_MINIMALE) -> bool:
    """
    La personne est-elle trop petite pour qu'on juge son equipement ?

    Sur une image de 1080 px de haut, un seuil de 20 % rejette toute
    personne mesurant moins de 216 px :

        personne de 524 px  ->  48,5 %  ->  surveillee
        personne de 142 px  ->  13,2 %  ->  hors perimetre

    Ce n'est pas un filtre de confort. A vingt metres, un casque fait
    quelques pixels : le modele ne peut pas le voir, et un systeme qui
    alerte sur ce qu'il ne peut pas voir n'informe personne.

    LE COUT EST ASSUME : les ouvriers eloignes ne sont pas surveilles. Cela
    devient une exigence de placement des cameras, a inscrire au cahier des
    charges d'installation plutot qu'a masquer dans le code.
    """
    return (boite[3] - boite[1]) < minimum * hauteur_image


def tete_hors_champ(boite: np.ndarray, marge: int = MARGE_BORD) -> bool:
    """
    La tete de la personne sort-elle par le haut de l'image ?

    Si la boite touche le bord superieur, le casque n'est pas dans le champ.
    Dire « casque non detecte » serait alors un MENSONGE : le systeme n'a
    pas regarde, il ne peut rien affirmer.

        boite de y=0 a y=1080 sur une image de 1080  ->  tete hors champ
        boite de y=219 a y=526                       ->  tete visible

    Le verdict correct est « indetermine ». C'est plus honnete qu'une
    alerte, et plus utile : cela dit a l'exploitant de rehausser sa camera.
    """
    return boite[1] <= marge


def centre(boite: np.ndarray) -> tuple[float, float]:
    """Centre d'une boite (x1, y1, x2, y2)."""
    return (boite[0] + boite[2]) / 2, (boite[1] + boite[3]) / 2


def dans_la_boite(point: tuple[float, float], boite: np.ndarray) -> bool:
    x, y = point
    return boite[0] <= x <= boite[2] and boite[1] <= y <= boite[3]


def dans_la_zone_haute(point: tuple[float, float], boite: np.ndarray,
                       part: float = PART_HAUTE) -> bool:
    """Le point est-il dans la partie superieure de la boite ?"""
    _, y = point
    hauteur = boite[3] - boite[1]
    return boite[1] <= y <= boite[1] + part * hauteur


def associer(personnes: list[dict], equipements: list[dict],
             part_haute: float = PART_HAUTE) -> dict[int, dict]:
    """
    Attribue chaque equipement a une personne.

    `personnes`    [{"identifiant": 3, "boite": array([x1,y1,x2,y2])}, ...]
    `equipements`  [{"classe": "helmet", "confiance": 0.73,
                     "boite": array([...])}, ...]

    Renvoie, par identifiant de personne, la meilleure confiance obtenue
    pour chaque classe d'equipement :

        {3: {"helmet": 0.73, "gloves": 0.0, ...}, ...}

    Une classe absente vaut 0.0 : une absence de detection EST une
    information, ce n'est pas une donnee manquante.
    """
    resultat = {p["identifiant"]: {} for p in personnes}

    for equipement in equipements:
        point = centre(equipement["boite"])
        tete = equipement["classe"] in EQUIPEMENTS_TETE

        candidats = []
        for personne in personnes:
            boite = personne["boite"]
            if not dans_la_boite(point, boite):
                continue
            if tete and not dans_la_zone_haute(point, boite, part_haute):
                continue
            # Distance du casque au HAUT de la boite : celui sur la tete de
            # qui il repose le plus vraisemblablement, si deux personnes se
            # chevauchent.
            candidats.append((abs(point[1] - boite[1]), personne["identifiant"]))

        if not candidats:
            continue
        _, identifiant = min(candidats)
        classe = equipement["classe"]
        ancienne = resultat[identifiant].get(classe, 0.0)
        resultat[identifiant][classe] = max(ancienne, float(equipement["confiance"]))

    # Toute classe non vue vaut zero, explicitement.
    for identifiant in resultat:
        for classe in EQUIPEMENTS_TETE | EQUIPEMENTS_CORPS:
            resultat[identifiant].setdefault(classe, 0.0)
    return resultat
