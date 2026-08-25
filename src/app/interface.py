"""
Interface Streamlit — detection d'EPI sur chantier.

Ce que ce fichier contient, et ce qu'il ne contient pas
-------------------------------------------------------
Il ne contient QUE de l'affichage. Aucune regle metier, aucun seuil, aucun
calcul de verdict : tout cela vit dans `detection.py`, `regles.py` et
`decision.py`, qui sont testables sans navigateur.

    Si une decision se prend ici, elle est au mauvais endroit.

C'est ce qui permet aux 56 tests de couvrir le comportement du systeme sans
jamais lancer Streamlit — et au tableau de bord du J13 de se brancher sur la
meme logique sans la reecrire.

Lancement :
    streamlit run src/app/interface.py
"""

from __future__ import annotations

import sys
import tempfile
import time
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
# PALETTE ACCESSIBLE — revue le 26 aout.
#
# La version precedente opposait ROUGE et VERT. C'est le pire choix
# possible : la deuterananopie et la protanopie — les deux formes les plus
# repandues de daltonisme, environ 8 % des hommes — rendent ces deux
# couleurs presque identiques. Un ouvrier signale et un ouvrier conforme
# auraient porte la meme couleur pour une personne sur douze.
#
# Palette retenue : Okabe-Ito, concue pour rester distinguable sous les
# trois formes de daltonisme. Bleu contre vermillon au lieu de vert contre
# rouge — les deux se distinguent aussi bien en vision normale.
#
# ET SURTOUT : LA COULEUR NE PORTE JAMAIS L'INFORMATION SEULE.
# Chaque boite affiche un symbole et un texte. Une capture d'ecran en noir
# et blanc, un ecran mal regle ou un oeil daltonien lisent la meme chose.
# --------------------------------------------------------------------------
BLEU = (178, 114, 0)        # #0072B2 — casque detecte
VERMILLON = (0, 94, 213)    # #D55E00 — casque non detecte, alerte possible
JAUNE = (66, 228, 240)      # #F0E442 — tete hors champ, verdict impossible
#   Le jaune plutot que l ambre : verifie par simulation. Avec l ambre,
#   la pire paire de la palette tombait a 46 sous tritanopie ; avec le
#   jaune elle remonte a 89. Deux couleurs chaudes voisines se
#   confondent, une claire et une saturee non.
GRIS = (153, 153, 153)      # #999999 — hors perimetre, non juge

# Le symbole double la couleur. C'est lui qui porte l'information quand la
# couleur ne peut pas.
SYMBOLES = {"detecte": "[OK]", "absent": "[!]",
            "indetermine": "[?]", "hors": "[-]"}

LIBELLES = {
    "helmet": t("casque"), "glasses": t("lunettes"),
    "gloves": t("gants"), "safety-vest": t("gilet"),
}


# --------------------------------------------------------------------------
# Chargement mis en cache : le modele coute plusieurs secondes a charger,
# et Streamlit re-execute tout le script a chaque interaction.
# --------------------------------------------------------------------------

# Le modele retenu en D-037, apres comparaison des six a budget de calcul
# egal. Ce n'est pas un reglage : c'est un resultat.
MODELE_RETENU = "yolov8m"

# Le repli, et lui seul. Il ne s'active pas au choix de l'utilisateur mais
# parce que le modele retenu n'a pas pu etre charge — typiquement faute de
# memoire sur un hebergement gratuit. Un responsable securite n'a aucun
# element pour arbitrer entre deux architectures ; lui poser la question
# revient a lui repasser une decision deja tranchee.
MODELE_REPLI = "yolov8n"


@st.cache_resource(show_spinner="Chargement du modele…")
def obtenir_detecteur(nom_modele: str) -> Detecteur:
    """
    Charge le modele, puis le PRECHAUFFE sur une image vide.

    Mesure du J12 : la premiere inference coute 2 061 ms, les suivantes
    301 ms. Ce surcout n'est pas du calcul utile — c'est l'allocation des
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
        return obtenir_detecteur(MODELE_RETENU), MODELE_RETENU, None
    except Exception as erreur:                       # noqa: BLE001
        repli = RACINE / "models" / f"{MODELE_REPLI}.pt"
        if not repli.is_file():
            raise
        return obtenir_detecteur(MODELE_REPLI), MODELE_REPLI, str(erreur)


def afficher_image(image, legende: str = "") -> None:
    """
    Affiche une image en pleine largeur, quelle que soit la version.

    `st.image` a change de parametre en cours de route : `use_column_width`
    jusqu'a Streamlit 1.39, `use_container_width` ensuite. Le poste de
    travail et la plateforme d'hebergement n'ont pas forcement la meme
    version — on essaie donc le nom recent, puis l'ancien.
    """
    try:
        st.image(image, caption=legende, use_container_width=True)
    except TypeError:
        st.image(image, caption=legende, use_column_width=True)


def annoter(image: np.ndarray, resultat: ResultatImage) -> np.ndarray:
    """
    Dessine une boite par personne, coloree selon le casque.

    Le libelle dit « casque non detecte », jamais « non conforme » : le
    systeme constate une absence de detection, il ne juge pas (D-032).
    """
    import cv2

    annotee = image.copy()
    epaisseur = max(2, int(min(annotee.shape[:2]) / 400))

    for personne in resultat.personnes:
        x1, y1, x2, y2 = [int(v) for v in personne.boite]

        if personne.statut is Statut.HORS_PERIMETRE:
            couleur, mention, cle = GRIS, "hors perimetre", "hors"
        elif personne.statut is Statut.TETE_HORS_CHAMP:
            couleur, mention, cle = JAUNE, "tete hors champ", "indetermine"
        elif personne.manque_casque:
            couleur, mention, cle = VERMILLON, "casque non detecte", "absent"
        else:
            couleur, mention, cle = BLEU, "casque detecte", "detecte"

        # Le trait double lui aussi l'information : plein et epais pour ce
        # qui est juge, fin pour ce qui ne l'est pas. Troisieme canal, apres
        # la couleur et le symbole.
        trait = max(1, epaisseur // 2) if cle == "hors" else epaisseur
        cv2.rectangle(annotee, (x1, y1), (x2, y2), couleur, trait)

        texte = f"{SYMBOLES[cle]} #{personne.identifiant} {mention}"
        echelle = max(0.4, epaisseur * 0.22)
        (largeur_txt, hauteur_txt), _ = cv2.getTextSize(
            texte, cv2.FONT_HERSHEY_SIMPLEX, echelle, 1)
        cv2.rectangle(annotee, (x1, max(0, y1 - hauteur_txt - 8)),
                      (x1 + largeur_txt + 6, y1), couleur, -1)
        cv2.putText(annotee, texte, (x1 + 3, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, echelle, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return annotee


def tableau_personnes(resultat: ResultatImage) -> list[dict]:
    """Une ligne par personne, une colonne par equipement surveille."""
    etats = {Statut.SURVEILLEE: t("[OK] surveillee"),
             Statut.HORS_PERIMETRE: t("[-] hors perimetre"),
             Statut.TETE_HORS_CHAMP: t("[?] tete hors champ")}
    lignes = []
    for personne in resultat.personnes:
        ligne = {t("Personne"): f"#{personne.identifiant}",
                 t("Etat"): etats[personne.statut]}
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

def page_image(detecteur: Detecteur) -> None:
    st.subheader(t("Analyse d'une image"))
    fichier = st.file_uploader(t("Photographie de chantier"),
                               type=["jpg", "jpeg", "png", "bmp"])
    if fichier is None:
        st.info(t("Depose une image pour lancer l'analyse."))
        return

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
    haut[1].metric(t("Surveillees"), chiffres["surveillees"])
    haut[2].metric(t("Non jugeables"),
                   chiffres["hors_perimetre"] + chiffres["tete_hors_champ"],
                   help="trop eloignees, ou tete hors du champ")
    bas = st.columns(3)
    bas[0].metric(t("[OK] Casque detecte"), chiffres["casque_detecte"])
    bas[1].metric(t("[!] Casque non detecte"), chiffres["casque_non_detecte"])
    bas[2].metric(t("Duree"), f"{duree * 1000:.0f} ms")

    if chiffres["tete_hors_champ"]:
        st.warning(
            f"{chiffres['tete_hors_champ']} personne(s) ont la tete hors du "
            "champ : le casque n'est pas dans l'image, aucun verdict n'est "
            "rendu. **Rehausser la camera** corrigerait ces cas."
        )
    if chiffres["hors_perimetre"]:
        st.info(
            f"{chiffres['hors_perimetre']} personne(s) occupent moins de "
            f"{HAUTEUR_MINIMALE:.0%} de la hauteur d'image : trop eloignees "
            "pour que leur equipement soit jugeable. Affichees, jamais alertees."
        )

    afficher_image(annoter(image, resultat)[:, :, ::-1],
                   f"{resultat.largeur} × {resultat.hauteur} px")

    if resultat.personnes:
        st.dataframe(tableau_personnes(resultat), use_container_width=True,
                     hide_index=True)
    else:
        st.warning(t("Aucune personne detectee : aucun verdict ne peut etre rendu."))

    st.caption(
        t("Sur une image isolee, il n'y a pas d'historique : le verdict est "
        "rendu par comparaison au seuil calibre de chaque classe. "
        "L'hysteresis ne s'applique qu'a la video.")
    )
    st.markdown(
        t("**Legende** — la couleur ne porte jamais l'information seule : "
        "chaque boite affiche aussi un symbole. "
        "`[OK]` casque detecte · `[!]` casque non detecte · "
        "`[?]` tete hors champ, verdict impossible · "
        "`[-]` hors perimetre de surveillance.")
    )


def page_video(detecteur: Detecteur, reglages: Reglages) -> None:
    st.subheader(t("Analyse d'une video"))
    st.caption(
        t("C'est ici qu'agissent les trois mecanismes du J11 : deux seuils au "
        "lieu d'un, une confirmation sur images consecutives, et une alerte "
        "par personne et par episode.")
    )

    fichier = st.file_uploader(t("Sequence de chantier"), type=["mp4", "avi", "mov"])
    cadence = st.slider(t("Analyser une image sur"), 1, 5, 3,
                        help=t("3 correspond au reglage utilise pour les mesures du J11."))
    if fichier is None:
        st.info(t("Depose une video pour lancer l'analyse."))
        return

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
    colonnes[0].metric(t("Personnes suivies"), statistiques.get("personnes_suivies", "—"))
    colonnes[1].metric(t("Basculements de verdict"),
                       statistiques.get("basculements_de_verdict", "—"))
    colonnes[2].metric(t("Alertes"), len(journal))

    if journal:
        st.dataframe(journal, use_container_width=True, hide_index=True)
    else:
        st.success(t("Aucune alerte confirmee sur cette sequence."))


def page_guide() -> None:
    """
    Le guide d'utilisation, exige au §6.2 du cahier des charges.

    Il vit DANS l'application et non a cote : un guide qu'il faut aller
    chercher ailleurs n'est pas lu. Il est bilingue comme le reste de
    l'interface.
    """
    st.subheader(t("Guide d'utilisation"))
    st.markdown(GUIDE[st.session_state.get("langue", "fr")])


def page_limites() -> None:
    st.subheader(t("Ce que ce systeme ne garantit pas"))
    st.markdown(LIMITES[st.session_state.get("langue", "fr")])


# La page la plus importante de l'outil, et la seule que le sujet ne demande
# pas. Un systeme de securite qui n'enonce pas ses limites laisse croire
# qu'il n'en a pas — c'est le mode de defaillance le plus dangereux, parce
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
fausses alertes** — et toutes deux relevent du **placement de la camera**,
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
situations accounted for **94 % of false alerts** — and both stem from
**camera placement**, not from the model.
""",
}

# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Detection EPI — chantier",
                       page_icon="🦺", layout="wide",
                       initial_sidebar_state="auto")
    # Le selecteur de langue est lu AVANT tout affichage. Streamlit rejoue
    # le script de haut en bas : si `choisir()` etait appele plus bas, le
    # titre serait rendu avec la langue precedente — un render de retard,
    # visible a chaque bascule.
    with st.sidebar:
        langue.choisir()

    st.title(t("Detection d'equipements de protection"))
    st.caption(t("Le modele detecte des objets. Les regles produisent le verdict."))

    detecteur, nom_modele, motif_repli = charger_le_detecteur()

    with st.sidebar:
        st.divider()
        st.header(t("Reglages"))
        # Le modele n'est plus un choix offert : il est AFFICHE.
        st.text(f"{t('Modele')}  {nom_modele}")
        if motif_repli:
            st.warning(t("Repli automatique sur %s : le modele retenu n'a pas "
                         "pu etre charge. Les resultats sont ceux d'un modele "
                         "plus leger et moins precis.") % nom_modele)
            with st.expander(t("Motif technique")):
                st.code(motif_repli)
        else:
            st.caption(t("Modele retenu apres comparaison de six "
                         "architectures a budget de calcul egal."))

        st.divider()
        if calibrage_disponible(nom_modele):
            st.caption(t("**Seuils calibres par classe** — mesures, non choisis."))
        else:
            st.caption(t("**Seuils par defaut** — ce modele n'a pas ete calibre. "
                         "Valeurs de repli, non mesurees."))
        for classe, seuil in sorted(charger_seuils(nom_modele).items()):
            marque = " ⚠ alerte" if classe in PERIMETRE_ALERTE else ""
            st.text(f"{LIBELLES.get(classe, classe):9s} {seuil:.3f}{marque}")

        st.divider()
        defauts = Reglages()
        st.caption("**Decision en video**")
        st.text(f"seuil haut  {defauts.seuil_haut:.3f}")
        st.text(f"seuil bas   {defauts.seuil_bas:.3f}")
        st.text(f"attente     {defauts.images_avant_alerte} images")


    (onglet_bord, onglet_image, onglet_video,
     onglet_guide, onglet_limites) = st.tabs(
        [t("Tableau de bord"), t("Image"), t("Video"),
         t("Guide"), t("Limites")])
    with onglet_bord:
        tableau_bord.page(nom_modele)
    with onglet_image:
        page_image(detecteur)
    with onglet_video:
        page_video(detecteur, Reglages())
    with onglet_guide:
        page_guide()
    with onglet_limites:
        page_limites()


if __name__ == "__main__":
    main()


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

### Les quatre onglets

| Onglet | Ce qu'il fait |
|---|---|
| **Tableau de bord** | Les chiffres agrégés : taux de détection, chronologie, carte du champ, descente au cas |
| **Image** | Dépose une photographie, obtiens un verdict par personne |
| **Vidéo** | Dépose une séquence : hystérésis, confirmation et agrégation s'appliquent |
| **Limites** | Ce que le système ne garantit pas. **À lire avant de s'en servir** |

---

### Lire une image annotée

Chaque personne détectée reçoit une boîte, un symbole et un texte. **La couleur
ne porte jamais l'information seule** — une capture en noir et blanc reste
lisible.

| Symbole | Signification | Ce qu'il faut faire |
|---|---|---|
| `[OK]` | Casque détecté | rien |
| `[!]` | Casque non détecté | **vérifier** — deux fois sur trois, l'ouvrier est en règle |
| `[?]` | Tête hors du champ | le système n'a pas regardé : **rehausser la caméra** |
| `[-]` | Hors périmètre | personne trop éloignée pour être jugée |

> **`[?]` et `[-]` ne sont pas des alertes.** Le système dit qu'il ne peut pas
> juger, ce qui n'est pas la même chose que « l'équipement manque ».

---

### Les chiffres du tableau de bord

**Personnes vues** — tout ce que le détecteur a trouvé.

**Surveillées** — celles que le système peut effectivement juger : assez
grandes, et tête dans le champ.

**Taux de détection du casque** — la part des personnes surveillées chez qui un
casque a été retenu. *Ce n'est pas un taux de conformité* : le système atteste
une présence, pas le respect d'une norme.

**Non jugeables** — trop loin, ou tête coupée. **Un chiffre élevé ne signale pas
une panne : il signale une caméra mal placée.**

---

### La carte du champ de vision

Elle montre **où, dans l'image**, les alertes se produisent. Elle se lit ainsi :

| Ce que tu vois | Ce que ça veut dire |
|---|---|
| Concentration **en haut** | têtes coupées par le bord — caméra trop basse ou trop proche |
| Concentration **diffuse et faible** | personnes trop lointaines — champ trop large |
| Bande **horizontale** | caméra à hauteur d'homme : toutes les têtes s'alignent |
| Point chaud **au centre** | sur des photographies composées, c'est la règle des tiers, **pas une zone à risque** |

Dans les trois premiers cas, la correction est le **placement de la caméra**,
non un réglage du logiciel.

---

### Le périmètre d'alerte

Par défaut, **seul le casque déclenche une alerte**. C'est le seul équipement
obligatoire partout, et le seul dont les taux d'erreur mesurés le permettent.

Les trois autres sont détectés, associés et comptés — ils n'interrompent
simplement pas l'opérateur.

Tu peux les activer par les cases du tableau de bord. **Chaque case affiche ce
qu'elle coûterait** sur le corpus courant. Attention : lunettes, gants et gilet
ne sont pas exigés pour toutes les tâches — le Code du travail impose une
évaluation par poste, et le système ignore quelle tâche exécute la personne
qu'il regarde.

---

### Régler la langue et le modèle

Panneau de gauche. **yolov8m** est le modèle retenu ; **yolov8n** est le repli
si la mémoire d'hébergement est insuffisante, au prix de la qualité.

Les seuils affichés sous le sélecteur sont **calibrés par classe**, calculés sur
les données de validation. Ce ne sont pas des valeurs rondes choisies à la main.

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

### The four tabs

| Tab | What it does |
|---|---|
| **Dashboard** | Aggregated figures: detection rate, timeline, frame map, drill-down |
| **Image** | Drop a photograph, get a verdict per person |
| **Video** | Drop footage: hysteresis, confirmation and aggregation apply |
| **Limitations** | What the system does not guarantee. **Read before use** |

---

### Reading an annotated image

Every detected person gets a box, a symbol and a text label. **Colour never
carries the information alone** — a black-and-white screenshot stays readable.

| Symbol | Meaning | What to do |
|---|---|---|
| `[OK]` | Helmet detected | nothing |
| `[!]` | Helmet not detected | **check** — two times out of three the worker is compliant |
| `[?]` | Head out of frame | the system did not look: **raise the camera** |
| `[-]` | Out of range | person too far away to be judged |

> **`[?]` and `[-]` are not alerts.** The system is saying it cannot judge,
> which is not the same as "the equipment is missing".

---

### The dashboard figures

**People seen** — everything the detector found.

**Monitored** — those the system can actually judge: large enough, head in frame.

**Helmet detection rate** — the share of monitored people with a helmet
retained. *This is not a compliance rate*: the system establishes presence, not
conformity to a standard.

**Cannot be judged** — too far, or head cut off. **A high figure does not signal
a malfunction: it signals a badly placed camera.**

---

### The field-of-view map

It shows **where in the frame** alerts occur:

| What you see | What it means |
|---|---|
| Concentration **at the top** | heads cut off by the edge — camera too low or too close |
| **Faint, spread out** | people too far away — field of view too wide |
| **Horizontal band** | camera at eye level: every head lines up |
| **Central hot spot** | on composed photographs this is the rule of thirds, **not a risk zone** |

In the first three cases the fix is **camera placement**, not a software setting.

---

### The alert scope

By default, **only the helmet raises an alert**. It is the only item mandatory
everywhere, and the only one whose measured error rates allow it.

The other three are detected, linked and counted — they simply do not interrupt
the operator.

You can enable them from the dashboard checkboxes. **Each box shows what it
would cost** on the current corpus. Note: eye protection, gloves and vests are
not required for every task — French labour law requires a per-role assessment,
and the system does not know which task the person is performing.

---

### Language and model settings

Left-hand panel. **yolov8m** is the selected model; **yolov8n** is the fallback
if hosting memory is insufficient, at the cost of quality.

The thresholds shown are **calibrated per class**, computed on validation data.
They are not round numbers chosen by hand.

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
