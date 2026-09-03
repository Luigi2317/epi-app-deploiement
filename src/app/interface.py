"""
Interface Streamlit, detection d'EPI sur chantier.

Ce que ce fichier contient, et ce qu'il ne contient pas
-------------------------------------------------------
Il ne contient QUE de l'affichage. Aucune regle metier, aucun seuil, aucun
calcul de verdict : tout cela vit dans `detection.py`, `regles.py` et
`decision.py`, qui sont testables sans navigateur.

    Si une decision se prend ici, elle est au mauvais endroit.

C'est ce qui permet aux 56 tests de couvrir le comportement du systeme sans
jamais lancer Streamlit, et au tableau de bord du J13 de se brancher sur la
meme logique sans la reecrire.

Lancement :
    streamlit run src/app/interface.py
"""

from __future__ import annotations

import sys
import tempfile
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import streamlit as st

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.app.decision import Reglages                      # noqa: E402
from src.app.detection import (                            # noqa: E402
    PERIMETRE_ALERTE,
    Detecteur,
    ResultatImage,
    Statut,
    calibrage_disponible,
    charger_seuils,
    resume,
)
from src.app import langue                # noqa: E402
from src.app.langue import t              # noqa: E402
from src.app.regles import HAUTEUR_MINIMALE                 # noqa: E402
from src.app import tableau_bord                           # noqa: E402

# --------------------------------------------------------------------------
# PALETTE ACCESSIBLE, revue le 26 aout.
#
# La version precedente opposait ROUGE et VERT. C'est le pire choix
# possible : la deuterananopie et la protanopie, les deux formes les plus
# repandues de daltonisme, environ 8 % des hommes, rendent ces deux
# couleurs presque identiques. Un ouvrier signale et un ouvrier conforme
# auraient porte la meme couleur pour une personne sur douze.
#
# Palette retenue : Okabe-Ito, concue pour rester distinguable sous les
# trois formes de daltonisme. Bleu contre vermillon au lieu de vert contre
# rouge, les deux se distinguent aussi bien en vision normale.
#
# ET SURTOUT : LA COULEUR NE PORTE JAMAIS L'INFORMATION SEULE.
# Chaque boite affiche un symbole et un texte. Une capture d'ecran en noir
# et blanc, un ecran mal regle ou un oeil daltonien lisent la meme chose.
# --------------------------------------------------------------------------
BLEU = (178, 114, 0)        # #0072B2, casque detecte
VERMILLON = (0, 94, 213)    # #D55E00, casque non detecte, alerte possible
JAUNE = (66, 228, 240)      # #F0E442, tete hors champ, verdict impossible
#   Le jaune plutot que l ambre : verifie par simulation. Avec l ambre,
#   la pire paire de la palette tombait a 46 sous tritanopie ; avec le
#   jaune elle remonte a 89. Deux couleurs chaudes voisines se
#   confondent, une claire et une saturee non.
GRIS = (153, 153, 153)      # #999999, hors perimetre, non juge

# Le symbole double la couleur. C'est lui qui porte l'information quand la
# couleur ne peut pas.
SYMBOLES = {"detecte": "[OK]", "absent": "[!]",
            "indetermine": "[?]", "hors": "[-]"}

LIBELLES = {
    "helmet": t("casque"), "glasses": t("lunettes"),
    "gloves": t("gants"), "safety-vest": t("gilet"),
}

# Une phrase entiere par classe, et non « %s non detecte » : le participe
# s'accorde. « lunettes » est feminin pluriel, « gants » masculin pluriel.
# Composer la phrase a partir du nom produirait « lunettes non detecte ».
#
# Le libelle du casque est impose mot pour mot par D-032 : le systeme
# constate une absence de detection, il ne prononce jamais de
# non-conformite. Les trois autres suivent la meme forme.
NON_DETECTE = {
    "helmet": "casque non détecté",
    "safety-vest": "gilet non détecté",
    "glasses": "lunettes non détectées",
    "gloves": "gants non détectés",
}


# --------------------------------------------------------------------------
# Chargement mis en cache : le modele coute plusieurs secondes a charger,
# et Streamlit re-execute tout le script a chaque interaction.
# --------------------------------------------------------------------------

# Le modele retenu en D-037, apres comparaison des six a budget de calcul
# egal. Ce n'est pas un reglage : c'est un resultat.
MODELE_RETENU = "yolov8m"

# Le repli, et lui seul. Il ne s'active pas au choix de l'utilisateur mais
# parce que le modele retenu n'a pas pu etre charge, typiquement faute de
# memoire sur un hebergement gratuit. Un responsable securite n'a aucun
# element pour arbitrer entre deux architectures ; lui poser la question
# revient a lui repasser une decision deja tranchee.
MODELE_REPLI = "yolov8n"


@st.cache_resource(show_spinner="Chargement du modele…")
def obtenir_detecteur(nom_modele: str) -> Detecteur:
    """
    Charge le modele, puis le PRECHAUFFE sur une image vide.

    Mesure du J12 : la premiere inference coute 2 061 ms, les suivantes
    301 ms. Ce surcout n'est pas du calcul utile : c'est l'allocation des
    tampons et la compilation des noyaux, payee une fois.

    Sans prechauffage, c'est le PREMIER UTILISATEUR qui l'encaisse, et il
    en tire une impression de lenteur que le service n'a pas. On la paie
    donc ici, pendant que le message « Chargement du modele » est affiche
    et qu'elle passe inapercue.
    """
    detecteur = Detecteur(poids=RACINE / "models" / f"{nom_modele}.pt")
    detecteur.analyser_image(np.zeros((640, 640, 3), dtype=np.uint8))
    return detecteur


def charger_le_detecteur() -> tuple[Detecteur, str, str | None]:
    """
    Charge le modele retenu, ou bascule sur le repli s'il ne passe pas.

    Retourne (detecteur, nom du modele actif, motif du repli ou None).

    Le motif est remonte jusqu'a l'ecran : une degradation silencieuse est
    pire qu'une panne, parce qu'on continue de lire les resultats comme
    s'ils venaient du systeme mesure.
    """
    try:
        detecteur, nom, motif = obtenir_detecteur(MODELE_RETENU), MODELE_RETENU, None
    except Exception as erreur:                       # noqa: BLE001
        repli = RACINE / "models" / f"{MODELE_REPLI}.pt"
        if not repli.is_file():
            raise
        detecteur, nom, motif = obtenir_detecteur(MODELE_REPLI), MODELE_REPLI, str(erreur)

    # On memorise ce qui tourne REELLEMENT, pour que la barre laterale
    # puisse l'afficher sans avoir a charger le modele elle-meme.
    st.session_state["_modele_actif"] = nom
    st.session_state["_motif_repli"] = motif
    return detecteur, nom, motif


def afficher_image(image, legende: str = "") -> None:
    """
    Affiche une image en pleine largeur, quelle que soit la version.

    `st.image` a change de parametre en cours de route : `use_column_width`
    jusqu'a Streamlit 1.39, `use_container_width` ensuite. Le poste de
    travail et la plateforme d'hebergement n'ont pas forcement la meme
    version : on essaie donc le nom recent, puis l'ancien.
    """
    try:
        st.image(image, caption=legende, use_container_width=True)
    except TypeError:
        st.image(image, caption=legende, use_column_width=True)


def mention_perimetre(perimetre: set[str]) -> str:
    """
    Comment nommer le perimetre dans une etiquette dessinee sur l'image.

    Une etiquette trop longue est tronquee par le bord droit de l'image :
    `cv2.putText` n'enroule pas, il coupe. On nomme donc l'equipement quand
    il est seul, et on le compte au-dela. Le detail des quatre reste lisible
    dans le tableau sous l'image, qui lui n'a pas cette contrainte.
    """
    if len(perimetre) == 1:
        classe = next(iter(perimetre))
        return LIBELLES.get(classe, classe)
    return "%d %s" % (len(perimetre), t("EPI"))


@lru_cache(maxsize=8)
def police(taille: int):
    """
    Une police TrueType capable d'ecrire « périmètre ».

    Pourquoi cette fonction existe
    ------------------------------
    `cv2.putText` ne sait dessiner que de l'ASCII : les polices Hershey
    d'OpenCV sont des traces vectoriels definis caractere par caractere, et
    la table s'arrete a 127. Tout le reste sort en « ? ». C'est pour ca que
    les etiquettes de cette application etaient ecrites sans accents.

    Le 2 septembre, la passe de reaccentuation a casse ca sans le savoir :
    « conforme · perimetre casque » s'affichait « conforme ?? p??rim??tre ».
    Le probleme n'etait pas l'accent, c'etait l'outil de dessin.

    On prend DejaVuSans, livree avec matplotlib, donc deja presente en local
    comme dans les dependances du deploiement. Aucun fichier a embarquer.
    Retourne None si elle est introuvable : l'appelant retombe alors sur
    OpenCV et un texte replie en ASCII.
    """
    try:
        import matplotlib
        from PIL import ImageFont

        chemin = (Path(matplotlib.get_data_path()) / "fonts" / "ttf"
                  / "DejaVuSans.ttf")
        return ImageFont.truetype(str(chemin), taille)
    except Exception:                                  # noqa: BLE001
        return None


def sans_accent(texte: str) -> str:
    """
    Replie un texte en ASCII, pour le repli OpenCV.

    « périmètre » devient « perimetre », et le point median un tiret. C'est
    laid, et c'est le prix a payer quand la police TrueType manque. Mieux
    vaut un texte sans accents qu'un texte troue de points d'interrogation.
    """
    plie = unicodedata.normalize("NFD", texte.replace("·", "-"))
    return "".join(c for c in plie if unicodedata.category(c) != "Mn")


def contraste(avant: tuple, arriere: tuple) -> float:
    """
    Rapport de contraste WCAG entre deux couleurs RGB, de 1 a 21.

    Formule officielle : (L_clair + 0,05) / (L_sombre + 0,05), ou L est la
    luminance relative, calculee sur des composantes LINEARISEES. La
    linearisation compte : un canal a 50 % de valeur n'emet pas 50 % de
    lumiere, l'ecran applique une courbe. L'ignorer donnerait des rapports
    faux d'un facteur deux sur les tons moyens.

    Reperes WCAG : 4,5 pour du texte courant, 3,0 pour du gros texte.
    """
    def luminance(couleur):
        canaux = []
        for valeur in couleur:
            v = valeur / 255
            canaux.append(v / 12.92 if v <= 0.03928
                          else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * canaux[0] + 0.7152 * canaux[1] + 0.0722 * canaux[2]

    a, b = luminance(avant), luminance(arriere)
    clair, sombre = max(a, b), min(a, b)
    return (clair + 0.05) / (sombre + 0.05)


def couleur_du_texte(fond_bgr: tuple) -> tuple:
    """
    Noir ou blanc sur ce fond, celui des deux qui contraste le plus.

    Pourquoi ce n'est pas un detail
    -------------------------------
    Le blanc etait ecrit en dur. Sur le bleu et le vermillon il passe, sur
    le JAUNE il donne 1,2 de contraste : l'etiquette « tete hors champ »
    etait illisible. Or c'est justement celle du troisieme etat, le
    resultat central du 24 aout.

    La palette Okabe-Ito de D-047 est choisie pour que les FONDS restent
    distinguables entre eux. Elle ne dit rien de ce qu'on ecrit dessus.
    C'est le complement, et il se mesure au lieu de se choisir a l'oeil.
    """
    rgb = tuple(reversed(fond_bgr))
    blanc, noir = (255, 255, 255), (17, 17, 17)
    return blanc if contraste(blanc, rgb) >= contraste(noir, rgb) else noir


def dessiner_etiquettes(image: np.ndarray, etiquettes: list) -> np.ndarray:
    """
    Ecrit les etiquettes des boites, avec accents si c'est possible.

    Les rectangles restent dessines par OpenCV, qui n'a aucun probleme avec
    eux. Seul le TEXTE passe par Pillow, et en une seule conversion pour
    toute l'image plutot qu'une par personne.

    L'image est en BGR, convention d'OpenCV ; Pillow attend du RGB. D'ou
    l'aller-retour, qui est aussi la raison de ne le faire qu'une fois.
    """
    import cv2

    if not etiquettes:
        return image

    hauteur, largeur = image.shape[:2]
    taille = max(13, int(min(hauteur, largeur) / 45))
    fonte = police(taille)

    if fonte is None:
        # Repli : OpenCV, donc ASCII seulement.
        echelle = max(0.4, taille / 24)
        for x, y, texte, couleur in etiquettes:
            texte = sans_accent(texte)
            (l_txt, h_txt), _ = cv2.getTextSize(
                texte, cv2.FONT_HERSHEY_SIMPLEX, echelle, 1)
            cv2.rectangle(image, (x, max(0, y - h_txt - 8)),
                          (x + l_txt + 6, y), couleur, -1)
            cv2.putText(image, texte, (x + 3, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, echelle,
                        tuple(reversed(couleur_du_texte(couleur))),
                        1, cv2.LINE_AA)
        return image

    from PIL import Image, ImageDraw

    toile = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    crayon = ImageDraw.Draw(toile)
    for x, y, texte, couleur in etiquettes:
        gauche, haut, droite, bas = crayon.textbbox((0, 0), texte, font=fonte)
        l_txt, h_txt = droite - gauche, bas - haut
        y_fond = max(0, y - h_txt - 8)
        # `couleur` est en BGR : on la retourne pour Pillow.
        crayon.rectangle([x, y_fond, x + l_txt + 8, y_fond + h_txt + 8],
                         fill=tuple(reversed(couleur)))
        crayon.text((x + 4, y_fond + 3), texte, font=fonte,
                    fill=couleur_du_texte(couleur))
    return cv2.cvtColor(np.array(toile), cv2.COLOR_RGB2BGR)


def annoter(image: np.ndarray, resultat: ResultatImage,
            perimetre: set[str] | None = None) -> np.ndarray:
    """
    Dessine une boite par personne, coloree selon le VERDICT D'ALERTE.

    Ce que la boite montre, et ce qu'elle ne montre pas
    ---------------------------------------------------
    La boite ne dit pas quels equipements ont ete detectes : elle dit s'il
    faut intervenir. Une personne, une boite, un verdict. Le detail des
    quatre equipements, avec leur confiance, vit dans le tableau affiche
    sous l'image.

    Le libelle le dit desormais explicitement, « conforme · perimetre
    casque » plutot que « casque detecte ». L'ancienne formulation parlait
    d'un equipement la ou la boite parle d'un verdict, et faisait conclure
    que le systeme ne traitait que le casque.

    Le perimetre est un ARGUMENT, pas une constante : il vient des cases
    cochees dans le tableau de bord (D-048). Par defaut, le casque seul.

    Vocabulaire impose par D-032 : le systeme constate une absence de
    detection, il ne juge pas une personne. D'ou « casque non detecte », et
    jamais « ouvrier non conforme ».
    """
    import cv2

    if perimetre is None:
        perimetre = set(PERIMETRE_ALERTE)
    annotee = image.copy()
    epaisseur = max(2, int(min(annotee.shape[:2]) / 400))
    etiquettes: list[tuple[int, int, str, tuple]] = []

    for personne in resultat.personnes:
        x1, y1, x2, y2 = [int(v) for v in personne.boite]
        manquants = personne.manques(perimetre)

        if personne.statut is Statut.HORS_PERIMETRE:
            couleur, mention, cle = GRIS, t("hors périmètre"), "hors"
        elif personne.statut is Statut.TETE_HORS_CHAMP:
            couleur, mention, cle = JAUNE, t("tête hors champ"), "indetermine"
        elif not perimetre:
            # Aucun equipement surveille : le systeme n'a rien a dire, et le
            # taire vaut mieux que d'afficher un « conforme » qui ne repose
            # sur aucun critere.
            couleur, cle = GRIS, "hors"
            mention = t("aucun équipement surveillé")
        elif manquants:
            couleur, cle = VERMILLON, "absent"
            mention = (t(NON_DETECTE[manquants[0]]) if len(manquants) == 1
                       else "%d %s" % (len(manquants), t("EPI non détectés")))
        else:
            couleur, cle = BLEU, "detecte"
            mention = "%s · %s %s" % (t("conforme"), t("périmètre"),
                                      mention_perimetre(perimetre))

        # Le trait double lui aussi l'information : plein et epais pour ce
        # qui est juge, fin pour ce qui ne l'est pas. Troisieme canal, apres
        # la couleur et le symbole.
        trait = max(1, epaisseur // 2) if cle == "hors" else epaisseur
        cv2.rectangle(annotee, (x1, y1), (x2, y2), couleur, trait)

        etiquettes.append((x1, y1,
                           f"{SYMBOLES[cle]} #{personne.identifiant} {mention}",
                           couleur))

    return dessiner_etiquettes(annotee, etiquettes)


def tableau_personnes(resultat: ResultatImage) -> list[dict]:
    """Une ligne par personne, une colonne par equipement surveille."""
    etats = {Statut.SURVEILLEE: t("[OK] surveillée"),
             Statut.HORS_PERIMETRE: t("[-] hors périmètre"),
             Statut.TETE_HORS_CHAMP: t("[?] tête hors champ")}
    lignes = []
    for personne in resultat.personnes:
        ligne = {t("Personne"): f"#{personne.identifiant}",
                 t("État"): etats[personne.statut]}
        for classe, libelle in LIBELLES.items():
            confiance = personne.equipements.get(classe, 0.0)
            vu = personne.verdicts.get(classe, False)
            marque = "[OK] oui" if vu else "[!] non"
            ligne[libelle] = f"{marque} ({confiance:.2f})"
        lignes.append(ligne)
    return lignes


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def page_image() -> None:
    st.subheader(t("Analyse d'une image"))
    fichier = st.file_uploader(t("Photographie de chantier"),
                               type=["jpg", "jpeg", "png", "bmp"])
    if fichier is None:
        st.info(t("Dépose une image pour lancer l'analyse."))
        return

    # Le detecteur se charge ICI, apres le depot d'un fichier : tant que
    # personne n'analyse rien, aucun modele n'occupe la memoire.
    detecteur, _, _ = charger_le_detecteur()

    import cv2
    donnees = np.frombuffer(fichier.getvalue(), np.uint8)
    image = cv2.imdecode(donnees, cv2.IMREAD_COLOR)
    if image is None:
        st.error(t("Image illisible."))
        return

    depart = time.perf_counter()
    resultat = detecteur.analyser_image(image)
    duree = time.perf_counter() - depart

    chiffres = resume(resultat)

    # Deux rangees de trois plutot qu'une de six : au-dela de trois colonnes,
    # une tablette en portrait comprime les nombres jusqu'a les tronquer.
    # Streamlit ne replie pas les colonnes tout seul.
    haut = st.columns(3)
    haut[0].metric(t("Personnes vues"), chiffres[t("personnes")])
    haut[1].metric(t("Surveillées"), chiffres["surveillees"])
    haut[2].metric(t("Non jugeables"),
                   chiffres["hors_perimetre"] + chiffres["tete_hors_champ"],
                   help="trop eloignees, ou tete hors du champ")
    bas = st.columns(3)
    bas[0].metric(t("[OK] Casque détecté"), chiffres["casque_detecte"])
    bas[1].metric(t("[!] Casque non détecté"), chiffres["casque_non_detecte"])
    bas[2].metric(t("Durée"), f"{duree * 1000:.0f} ms")

    # `t(...) % valeur` et non une f-string : la traduction est indexee par
    # le texte francais, or une f-string produit une chaine differente a
    # chaque appel selon le nombre. Le gabarit a trous, lui, est constant,
    # donc utilisable comme cle. C'est le meme motif que le message de repli
    # de modele.
    if chiffres["tete_hors_champ"]:
        st.warning(
            t("%d personne(s) ont la tête hors du champ : le casque n'est "
              "pas dans l'image, aucun verdict n'est rendu. **Rehausser la "
              "caméra** corrigerait ces cas.")
            % chiffres["tete_hors_champ"])
    if chiffres["hors_perimetre"]:
        st.info(
            t("%d personne(s) occupent moins de %.0f %% de la hauteur "
              "d'image : trop éloignées pour que leur équipement soit "
              "jugeable. Affichées, jamais alertées.")
            % (chiffres["hors_perimetre"], HAUTEUR_MINIMALE * 100))

    # Disposition : l'image bornee a gauche, la legende a droite, le tableau
    # des quatre equipements en pleine largeur dessous.
    #
    # Deux contraintes qui se contredisent, et l'ordre dans lequel elles ont
    # ete tranchees :
    #
    #   1. En pleine largeur, une photographie de chantier occupe tout
    #      l'ecran en hauteur. Les indicateurs sortent par le haut, les
    #      verdicts par le bas, et on ne peut plus lire un verdict et
    #      l'image qui le produit d'un seul regard.
    #   2. Le tableau porte SIX colonnes, dont les quatre equipements. Dans
    #      une colonne etroite il se tronque : « gilet » disparait, et le
    #      panneau censé prouver que les quatre sont mesures ne montre plus
    #      que deux.
    #
    # La legende occupe donc l'espace a droite de l'image, et le tableau
    # garde la pleine largeur. Sur ecran etroit, Streamlit empile les
    # colonnes : image, legende, tableau, dans cet ordre.
    gauche, droite = st.columns([3, 2], gap="medium")
    with gauche:
        afficher_image(
            annoter(image, resultat, tableau_bord.perimetre_actif())[:, :, ::-1],
            f"{resultat.largeur} × {resultat.hauteur} px")
    with droite:
        st.markdown(
            t("**Légende.** Une boîte par personne, et elle porte le "
              "**verdict d'alerte**, pas la liste des équipements : celle-ci "
              "est dans le tableau ci-dessous, avec les confiances des "
              "quatre. La couleur ne porte jamais l'information seule, "
              "chaque boîte affiche aussi un symbole.")
        )
        st.markdown(
            t("`[OK]` conforme sur le périmètre surveillé  \n"
              "`[!]` un équipement du périmètre n'est pas détecté  \n"
              "`[?]` tête hors champ, verdict impossible  \n"
              "`[-]` trop loin pour être jugée")
        )
        st.caption(
            t("Le périmètre surveillé se règle dans le panneau **Périmètre "
              "d'alerte** du tableau de bord. Par défaut le casque seul, "
              "décision D-038, avec le coût affiché à côté de chaque case."))

    if resultat.personnes:
        st.dataframe(tableau_personnes(resultat), use_container_width=True,
                     hide_index=True)
    else:
        st.warning(t("Aucune personne détectée : aucun verdict ne peut "
                     "être rendu."))

    st.caption(
        t("Sur une image isolée, il n'y a pas d'historique : le verdict est"
          " rendu par comparaison au seuil calibré de chaque classe. "
          "L'hystérésis ne s'applique qu'à la vidéo.")
    )


def page_video(reglages: Reglages) -> None:
    st.subheader(t("Analyse d'une vidéo"))
    st.caption(
        t("C'est ici qu'agissent les trois mécanismes du J11 : deux seuils "
          "au lieu d'un, une confirmation sur images consécutives, et une "
          "alerte par personne et par épisode.")
    )

    fichier = st.file_uploader(t("Séquence de chantier"), type=["mp4", "avi", "mov"])
    cadence = st.slider(t("Analyser une image sur"), 1, 5, 3,
                        help=t("3 correspond au réglage utilisé pour les "
                               "mesures du J11."))
    if fichier is None:
        st.info(t("Dépose une vidéo pour lancer l'analyse."))
        return

    # Meme principe que pour l'image : chargement a la demande.
    detecteur, _, _ = charger_le_detecteur()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(fichier.name).suffix) as tampon:
        tampon.write(fichier.getvalue())
        chemin = tampon.name

    barre = st.progress(0.0, text=t("Analyse en cours…"))
    journal, analysees = [], 0

    for numero, resultat, alertes in detecteur.analyser_video(
            chemin, cadence=cadence, reglages=reglages):
        analysees += 1
        for alerte in alertes:
            journal.append({t("Image"): alerte["image"],
                            t("Personne"): f"#{alerte['identifiant']}",
                            "Images consecutives": alerte["images_consecutives"],
                            "Message": alerte["message"]})
        if analysees % 10 == 0:
            barre.progress(min(0.99, analysees / 400),
                           text=f"{analysees} images analysees, {len(journal)} alertes")

    barre.progress(1.0, text=f"{analysees} images analysees")
    statistiques = getattr(detecteur, "derniere_statistique", {})

    colonnes = st.columns(3)
    colonnes[0].metric(t("Personnes suivies"), statistiques.get("personnes_suivies", ","))
    colonnes[1].metric(t("Basculements de verdict"),
                       statistiques.get("basculements_de_verdict", ","))
    colonnes[2].metric(t("Alertes"), len(journal))

    if journal:
        st.dataframe(journal, use_container_width=True, hide_index=True)
    else:
        st.success(t("Aucune alerte confirmée sur cette séquence."))


def page_guide() -> None:
    """
    Le guide d'utilisation, exige au §6.2 du cahier des charges.

    Il vit DANS l'application et non a cote : un guide qu'il faut aller
    chercher ailleurs n'est pas lu. Il est bilingue comme le reste de
    l'interface.
    """
    st.subheader(t("Guide d'utilisation"))
    st.markdown(GUIDE[st.session_state.get("langue", "fr")])


# --------------------------------------------------------------------------
# Le guide, exige au §6.2. Ecrit en Markdown : c'est le format le plus
# lisible dans le code source comme a l'ecran.
# --------------------------------------------------------------------------
GUIDE = {
    "fr": """
### À quoi sert cet outil

Il mesure le port des équipements de protection sur des images ou des séquences
de chantier, et signale les situations qui appellent une vérification.

**Il ne remplace pas un contrôle humain.** Il produit une trace continue là où
l'observation est ponctuelle.

---

### Les autres onglets

| Onglet | Ce qu'il fait |
|---|---|
| **Tableau de bord** | Les chiffres agrégés : taux de détection, chronologie, carte du champ, descente au cas |
| **Image** | Dépose une photographie, obtiens un verdict par personne |
| **Vidéo** | Dépose une séquence : hystérésis, confirmation et agrégation s'appliquent |
| **Limites** | Ce que le système ne garantit pas. **À lire avant de s'en servir** |

---

### Lire une image annotée

Chaque personne détectée reçoit une boîte, un symbole et un texte. **La couleur
ne porte jamais l'information seule** : une capture en noir et blanc reste
lisible.

| Symbole | Signification | Ce qu'il faut faire |
|---|---|---|
| `[OK]` | Casque détecté | rien |
| `[!]` | Casque non détecté | **vérifier** : deux fois sur trois, l'ouvrier est en règle |
| `[?]` | Tête hors du champ | le système n'a pas regardé : **rehausser la caméra** |
| `[-]` | Hors périmètre | personne trop éloignée pour être jugée |

> **`[?]` et `[-]` ne sont pas des alertes.** Le système dit qu'il ne peut pas
> juger, ce qui n'est pas la même chose que « l'équipement manque ».

---

### Les chiffres du tableau de bord

**Personnes vues** : tout ce que le détecteur a trouvé.

**Surveillées** : celles que le système peut effectivement juger : assez
grandes, et tête dans le champ.

**Taux de détection du casque**, la part des personnes surveillées chez qui un
casque a été retenu. *Ce n'est pas un taux de conformité* : le système atteste
une présence, pas le respect d'une norme.

**Non jugeables**, trop loin, ou tête coupée. **Un chiffre élevé ne signale pas
une panne : il signale une caméra mal placée.**

---

### La carte du champ de vision

Elle montre **où, dans l'image**, les alertes se produisent. Elle se lit ainsi :

| Ce que tu vois | Ce que ça veut dire |
|---|---|
| Concentration **en haut** | têtes coupées par le bord, caméra trop basse ou trop proche |
| Concentration **diffuse et faible** | personnes trop lointaines, champ trop large |
| Bande **horizontale** | caméra à hauteur d'homme : toutes les têtes s'alignent |
| Point chaud **au centre** | sur des photographies composées, c'est la règle des tiers, **pas une zone à risque** |

Dans les trois premiers cas, la correction est le **placement de la caméra**,
non un réglage du logiciel.

---

### Le périmètre d'alerte

Par défaut, **seul le casque déclenche une alerte**. C'est le seul équipement
obligatoire partout, et le seul dont les taux d'erreur mesurés le permettent.

Les trois autres sont détectés, associés et comptés : ils n'interrompent
simplement pas l'opérateur.

Tu peux les activer par les cases du tableau de bord. **Chaque case affiche ce
qu'elle coûterait** sur le corpus courant, et le choix se répercute aussitôt
sur les boîtes dessinées dans l'onglet Image.

Deux choses ne suivent pas ce réglage, et c'est voulu. Les **compteurs agrégés**
du tableau de bord viennent d'un corpus calculé une fois hors ligne, au casque
seul : les recalculer supposerait de rejouer le modèle sur les 3 028 personnes
du corpus. Et l'**alerte vidéo** reste au casque, parce que sa zone morte
d'hystérésis a été déduite du frémissement mesuré sur la confiance du casque,
et de lui seul. L'appliquer aux gants reviendrait à réutiliser une mesure qui
n'a pas été faite.

Attention : lunettes, gants et gilet ne sont pas exigés pour toutes les tâches.
Le Code du travail impose une évaluation par poste, et le système ignore quelle
tâche exécute la personne qu'il regarde.

---

### Ce que montre le panneau de gauche

**La langue est le seul réglage.** Le modèle, lui, n'est pas un choix offert :
il est affiché.

Le modèle est **yolov8m**, retenu après comparaison de six architectures à
budget de calcul égal. Ce n'est pas un réglage, c'est un résultat, et le
soumettre à l'utilisateur reviendrait à lui repasser une décision déjà tranchée
et mesurée. Un repli automatique sur **yolov8n** existe, mais il ne se déclenche
que si yolov8m ne peut pas être chargé, typiquement faute de mémoire sur
l'hébergement. Dans ce cas l'écran le dit, avec le motif technique : une
dégradation silencieuse serait pire qu'une panne.

Les seuils affichés en dessous sont **calibrés par classe**, calculés sur les
données de validation. Ce ne sont pas des valeurs rondes choisies à la main.

---

### Ce qu'il ne faut pas faire

- **Lire le taux de détection comme un taux de conformité.** Le système atteste
  la présence d'un objet ressemblant à un casque, pas le respect de la norme
  EN 397.
- **Utiliser une alerte pour sanctionner quelqu'un.** Deux alertes sur trois
  sont fausses au réglage actuel.
- **Ignorer les personnes non jugeables.** Elles ne sont pas conformes : elles
  ne sont pas regardées.
""",
    "en": """
### What this tool is for

It measures the wearing of protective equipment on construction site images or
footage, and flags situations that call for a check.

**It does not replace human inspection.** It produces a continuous record where
observation is occasional.

---

### The other tabs

| Tab | What it does |
|---|---|
| **Dashboard** | Aggregated figures: detection rate, timeline, frame map, drill-down |
| **Image** | Drop a photograph, get a verdict per person |
| **Video** | Drop footage: hysteresis, confirmation and aggregation apply |
| **Limitations** | What the system does not guarantee. **Read before use** |

---

### Reading an annotated image

Every detected person gets a box, a symbol and a text label. **Colour never
carries the information alone** : a black-and-white screenshot stays readable.

| Symbol | Meaning | What to do |
|---|---|---|
| `[OK]` | Helmet detected | nothing |
| `[!]` | Helmet not detected | **check**, two times out of three the worker is compliant |
| `[?]` | Head out of frame | the system did not look: **raise the camera** |
| `[-]` | Out of range | person too far away to be judged |

> **`[?]` and `[-]` are not alerts.** The system is saying it cannot judge,
> which is not the same as "the equipment is missing".

---

### The dashboard figures

**People seen**, everything the detector found.

**Monitored**, those the system can actually judge: large enough, head in frame.

**Helmet detection rate** : the share of monitored people with a helmet
retained. *This is not a compliance rate*: the system establishes presence, not
conformity to a standard.

**Cannot be judged**, too far, or head cut off. **A high figure does not signal
a malfunction: it signals a badly placed camera.**

---

### The field-of-view map

It shows **where in the frame** alerts occur:

| What you see | What it means |
|---|---|
| Concentration **at the top** | heads cut off by the edge, camera too low or too close |
| **Faint, spread out** | people too far away, field of view too wide |
| **Horizontal band** | camera at eye level: every head lines up |
| **Central hot spot** | on composed photographs this is the rule of thirds, **not a risk zone** |

In the first three cases the fix is **camera placement**, not a software setting.

---

### The alert scope

By default, **only the helmet raises an alert**. It is the only item mandatory
everywhere, and the only one whose measured error rates allow it.

The other three are detected, linked and counted: they simply do not interrupt
the operator.

You can enable them from the dashboard checkboxes. **Each box shows what it
would cost** on the current corpus, and the choice immediately carries over to
the boxes drawn in the Image tab.

Two things do not follow this setting, deliberately. The **aggregated counters**
on the dashboard come from a corpus computed once offline, helmet only:
recomputing them would mean replaying the model over all 3,028 people in the
corpus. And the **video alert** stays on the helmet, because its hysteresis
dead zone was derived from the flicker measured on helmet confidence, and on
that alone. Applying it to gloves would mean reusing a measurement that was
never made.

Note: eye protection, gloves and vests are not required for every task. French
labour law requires a per-role assessment, and the system does not know which
task the person is performing.

---

### What the left-hand panel shows

**Language is the only setting.** The model is not a choice on offer: it is
displayed.

The model is **yolov8m**, retained after comparing six architectures at equal
compute budget. That is not a setting, it is a result, and handing it back to
the user would mean handing back a decision already made and measured. An
automatic fallback to **yolov8n** exists, but it only fires when yolov8m cannot
be loaded, typically for lack of memory on the host. When that happens the
screen says so, with the technical reason: a silent degradation would be worse
than an outage.

The thresholds shown below are **calibrated per class**, computed on validation
data. They are not round numbers chosen by hand.

---

### What not to do

- **Read the detection rate as a compliance rate.** The system establishes the
  presence of an object resembling a helmet, not conformity to EN 397.
- **Use an alert to sanction someone.** Two alerts out of three are false at the
  current setting.
- **Ignore people who cannot be judged.** They are not compliant: they are not
  being looked at.
""",
}


def page_limites() -> None:
    st.subheader(t("Ce que ce système ne garantit pas"))
    st.markdown(LIMITES[st.session_state.get("langue", "fr")])


# La page la plus importante de l'outil, et la seule que le sujet ne demande
# pas. Un systeme de securite qui n'enonce pas ses limites laisse croire
# qu'il n'en a pas : c'est le mode de defaillance le plus dangereux, parce
# qu'il est invisible.
LIMITES = {
    "fr": """
Ces limites sont **mesurees**, et elles doivent etre connues de qui utilise
l'outil.

**Il atteste une presence, jamais une conformite.** La classe `helmet` du
jeu de donnees ne distingue pas un casque de chantier d'un casque de velo.
Le systeme ne peut donc pas certifier le respect de la norme EN 397 : il
detecte la presence d'un objet ressemblant a un casque.

**Seul le casque declenche une alerte.** Les trois autres equipements sont
detectes et affiches, sans interrompre l'operateur. Au meilleur reglage
possible, une alerte sur les gants signalerait a tort pres d'un ouvrier
conforme sur deux. Chaque classe a une condition de retour chiffree.

**Le detecteur manque souvent le casque.** Sur une video de chantier reel,
53 % des paires d'images consecutives ne contiennent aucune detection de
casque, alors que quinze des dix-sept personnes filmees en portent un. La
cause est identifiee : le jeu d'entrainement ne contient que **773 casques**
au total, sur des photographies de stock. Aucune logique de decision ne
compense un objet que le modele ne voit pas.

**Une personne qui traverse le champ en moins de deux secondes ne sera pas
signalee.** C'est le prix, assume, de la fenetre de confirmation qui
supprime la majorite des fausses alertes.

**Deux situations ne sont pas jugees, et le systeme le dit au lieu
d'alerter.** Une personne occupant moins de 20 % de la hauteur de l'image
est trop eloignee pour qu'un casque soit visible. Une personne dont la tete
sort par le haut du cadre n'a tout simplement pas ete regardee. Sur une
sequence de chantier reel, ces deux situations expliquaient **94 % des
fausses alertes**, et toutes deux relevent du **placement de la camera**,
non du modele.
""",
    "en": """
These limitations are **measured**, and anyone using the tool must know
them.

**It establishes presence, never compliance.** The `helmet` class in the
dataset does not distinguish a construction hard hat from a bicycle helmet.
The system therefore cannot certify compliance with standard EN 397: it
detects the presence of an object resembling a helmet.

**Only the helmet raises an alert.** The other three items are detected and
displayed without interrupting the operator. At the best possible operating
point, a glove alert would wrongly flag close to one compliant worker in
two. Each class has a measured condition for returning to the alert scope.

**The detector frequently misses the helmet.** On real construction footage,
53 % of consecutive frame pairs contain no helmet detection at all, even
though fifteen of the seventeen people filmed are wearing one. The cause is
identified: the training set contains only **773 helmets** in total, on
stock photographs. No decision logic can compensate for an object the model
does not see.

**Anyone crossing the field of view in under two seconds will not be
flagged.** That is the accepted price of the confirmation window that
removes most false alerts.

**Two situations are not judged, and the system says so instead of raising
an alert.** A person occupying less than 20 % of the image height is too far
away for a helmet to be visible. A person whose head leaves the top of the
frame simply has not been looked at. On real site footage, these two
situations accounted for **94 % of false alerts**, and both stem from
**camera placement**, not from the model.
""",
}

# --------------------------------------------------------------------------

def main() -> None:
    # `expanded` et non `auto`. En mode auto, Streamlit replie la barre
    # laterale des que la fenetre est etroite : portable, fenetre partagee,
    # tablette. Or TOUS les filtres du tableau de bord vivent dans cette
    # barre. Un visiteur qui ouvre l'application sur un ecran etroit voit
    # alors six sections figees et conclut qu'il n'y a pas d'interactivite.
    # Le repli reste possible, il est simplement demande a l'utilisateur au
    # lieu d'etre impose par la largeur.
    st.set_page_config(page_title="Detection EPI : chantier",
                       page_icon="🦺", layout="wide",
                       initial_sidebar_state="expanded")
    # Le selecteur de langue est lu AVANT tout affichage. Streamlit rejoue
    # le script de haut en bas : si `choisir()` etait appele plus bas, le
    # titre serait rendu avec la langue precedente, un render de retard,
    # visible a chaque bascule.
    with st.sidebar:
        langue.choisir()

    st.title(t("Détection d'équipements de protection"))
    st.caption(t("Le modèle détecte des objets. Les règles produisent le "
                 "verdict."))

    # Le modele n'est PLUS charge au demarrage. Il l'etait, et cela suffisait
    # a faire echouer le deploiement : lire 45 Mo de poids puis prechauffer
    # sur un processeur partage depasse le delai d'attente de l'hebergeur,
    # qui tue le processus avant que Streamlit n'ouvre son port, sans
    # message, puisque le noyau n'attend pas que Python s'explique.
    #
    # Il se charge desormais a la premiere analyse reelle. Le tableau de
    # bord, qui lit un corpus deja calcule, n'en a jamais besoin.
    nom_modele = st.session_state.get("_modele_actif", MODELE_RETENU)
    motif_repli = st.session_state.get("_motif_repli")

    with st.sidebar:
        st.divider()
        st.header(t("Réglages"))
        # Le modele n'est plus un choix offert : il est AFFICHE.
        st.text(f"{t('Modèle')}  {nom_modele}")
        if motif_repli:
            st.warning(t("Repli automatique sur %s : le modèle retenu n'a "
                         "pas pu être chargé. Les résultats sont ceux d'un "
                         "modèle plus léger et moins précis.") % nom_modele)
            with st.expander(t("Motif technique")):
                st.code(motif_repli)
        else:
            st.caption(t("Modèle retenu après comparaison de six "
                         "architectures à budget de calcul égal."))

        st.divider()
        if calibrage_disponible(nom_modele):
            st.caption(t("**Seuils calibrés par classe** : mesurés, non "
                         "choisis."))
        else:
            st.caption(t("**Seuils par défaut** : ce modèle n'a pas été "
                         "calibré. Valeurs de repli, non mesurées."))
        for classe, seuil in sorted(charger_seuils(nom_modele).items()):
            marque = " ⚠ alerte" if classe in PERIMETRE_ALERTE else ""
            st.text(f"{LIBELLES.get(classe, classe):9s} {seuil:.3f}{marque}")

        st.divider()
        defauts = Reglages()
        st.caption(t("**Décision en vidéo**"))
        st.text(f"seuil haut  {defauts.seuil_haut:.3f}")
        st.text(f"seuil bas   {defauts.seuil_bas:.3f}")
        st.text(f"attente     {defauts.images_avant_alerte} images")


    (onglet_bord, onglet_image, onglet_video,
     onglet_guide, onglet_limites) = st.tabs(
        [t("Tableau de bord"), t("Image"), t("Vidéo"),
         t("Guide"), t("Limites")])
    with onglet_bord:
        tableau_bord.page(nom_modele)
    with onglet_image:
        page_image()
    with onglet_video:
        page_video(Reglages())
    with onglet_guide:
        page_guide()
    with onglet_limites:
        page_limites()


if __name__ == "__main__":
    main()
