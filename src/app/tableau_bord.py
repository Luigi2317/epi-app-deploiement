"""
Tableau de bord metier — indicateurs, chronologie, carte du champ, detail.

Ce que ce fichier affiche, et sur quoi il s'appuie
--------------------------------------------------
Tout vient de `resultats/tableau_de_bord/evenements.csv`, produit par
`corpus.py`. Deux natures d'information y cohabitent, et l'ecran le dit :

    REEL    detections, verdicts, confiances, position dans le champ
    SIMULE  horodatage, camera, zone

Aucun calcul metier ici : les statuts, les seuils et les verdicts ont ete
decides ailleurs et sont lus tels quels. Ce module agrege et dessine.

La carte de chaleur, et pourquoi celle-ci
------------------------------------------
Le sujet demande une « heatmap des zones a risque ». Une image ne contient
aucune notion de zone : la zone est une propriete de la camera, pas de la
photographie. Fabriquer une carte geographique reviendrait donc a inventer.

Il existe une carte REELLE et plus utile : la repartition des alertes DANS
LE CHAMP DE LA CAMERA. Le J12 a mesure que 94 % des fausses alertes venaient
du haut du cadre (tetes coupees) et du fond (personnes trop petites). La
carte le rend visible d'un coup d'oeil, et elle repond a une question
operationnelle precise :

    ou faut-il repositionner la camera ?

Le drill-down
-------------
Nomme explicitement dans la grille (C3.2-3) : partir d'un chiffre agrege et
descendre jusqu'au cas individuel qui l'a produit. Ici, une alerte de la
table mene a son image annotee et a l'explication detection par detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.app.langue import t              # noqa: E402
from src.app.regles import HAUTEUR_MINIMALE               # noqa: E402

DOSSIER = RACINE / "resultats" / "tableau_de_bord"
EVENEMENTS = DOSSIER / "evenements.csv"

# Le corpus a ete calcule UNE FOIS, hors ligne, par ce modele
# (`corpus.py --modele yolov8m`). Changer de modele dans le panneau de
# gauche ne le recalcule pas : les chiffres agreges restent ceux-ci. Le
# selecteur n'agit que sur ce qui est analyse EN DIRECT — onglets Image et
# Video, et descente au cas.
MODELE_DU_CORPUS = "yolov8m"

# Chaque corpus dit ce qu'il est. La carte du champ ne signifie pas la meme
# chose selon la nature des images : sur des photographies composees, elle
# revele la regle des tiers ; sur un flux de camera fixe, elle revele le
# cadrage. Confondre les deux ferait passer une convention de cadrage
# photographique pour une zone a risque.
CORPUS = {
    "evenements": {
        "titre": "Validation SH17 — photographies de stock",
        "nature": "photo",
        "avertissement":
            "Ces images sont des **photographies composees** (banque Pexels). "
            "Le point chaud de la carte tombe au centre du cadre parce que "
            "c'est la qu'un photographe place son sujet — **c'est la regle "
            "des tiers, pas une zone a risque**. Sur ce corpus, la carte "
            "mesure une convention photographique.",
    },
    "flux_camera": {
        "titre": "Flux de camera — sequences de chantier reel",
        "nature": "camera",
        "avertissement":
            "Ces images sont des **captures de camera fixe**. Ici, la carte "
            "mesure ce qu'elle doit mesurer : le cadrage. Une concentration "
            "en haut signale des tetes coupees, une concentration diffuse et "
            "faible signale des personnes trop lointaines.",
    },
}


def corpus_disponibles() -> dict:
    return {c: CORPUS.get(c, {"titre": c, "nature": "inconnue",
                              "avertissement": ""})
            for c in sorted(f.stem for f in DOSSIER.glob("*.csv"))}


SURVEILLES = {"helmet": t("casque"), "glasses": t("lunettes"),
              "gloves": t("gants"), "safety-vest": t("gilet")}
# Les CLES sont des valeurs de donnees : elles ne se traduisent jamais.
# Les valeurs sont des libelles, traduits au moment de l'affichage par
# `etiquette()` — pas ici, ou la langue choisie n'est pas encore connue.
ETATS = {"surveillee": "surveillée",
         "hors_perimetre": "hors périmètre",
         "tete_hors_champ": "tête hors champ"}


def etiquette(statut: str) -> str:
    """Le libelle affichable d'un statut, dans la langue courante."""
    return t(ETATS.get(statut, statut))


@st.cache_data(show_spinner="Lecture du corpus…")
def charger(chemin: str, signature: float) -> pd.DataFrame:
    """
    Lit le corpus. `signature` est la date de modification du fichier :
    elle force la relecture quand le corpus est regenere.
    """
    donnees = pd.read_csv(chemin, parse_dates=["horodatage"])
    donnees["jour"] = donnees["horodatage"].dt.date
    donnees["heure"] = donnees["horodatage"].dt.hour
    # Aucune traduction ici : cette table est mise en cache, et un texte
    # traduit y resterait fige dans la langue du premier chargement.
    return donnees


def bandeau_declaration() -> None:
    """La phrase qui rend ce tableau de bord defendable."""
    st.info(
        t("**Ce que ces chiffres valent.** Les détections, verdicts, confiances "
        "et positions sont **réels** — produits par le modèle sur de vraies "
        "images. L'**horodatage, la caméra et la zone sont simulés** : le "
        "système n'a jamais été déployé, il n'existe donc aucun historique. "
        "Le contexte est reconstitué pour démontrer les mécanismes ; les "
        "chiffres qui engagent, eux, sont mesurés."),
        icon="ℹ️")


def indicateurs(donnees: pd.DataFrame) -> None:
    surveillees = donnees[donnees["statut"] == "surveillee"]
    alertes = int(surveillees["alerte"].sum())
    n = len(surveillees)
    non_jugeables = len(donnees) - n

    # Trois puis deux, jamais cinq de front : au-dela de trois colonnes, une
    # tablette en portrait tronque les nombres.
    haut = st.columns(3)
    haut[0].metric(t("Personnes vues"), f"{len(donnees):,}".replace(",", " "))
    haut[1].metric(t("Surveillées"), f"{n:,}".replace(",", " "),
                   help="assez grandes, et tête dans le champ")
    haut[2].metric(t("Non jugeables"), f"{non_jugeables:,}".replace(",", " "),
                   f"{non_jugeables / len(donnees):.0%} du total"
                   if len(donnees) else None, delta_color="off")
    bas = st.columns(2)
    bas[0].metric(t("Taux de détection du casque"),
                  f"{(n - alertes) / n:.1%}" if n else "—")
    bas[1].metric(t("[!] Alertes"), f"{alertes:,}".replace(",", " "))

    if n and (n - alertes) / n < 0.5:
        st.warning(
            t("**Un taux bas est ici attendu, et il est juste.** Le jeu "
            "d'images de validation ne contient que 773 casques pour 11 063 "
            "personnes : la plupart des gens photographiés n'en portent "
            "réellement pas. Ce taux décrit la composition du corpus, non "
            "un dysfonctionnement du système."))


def quatre_epi(donnees: pd.DataFrame) -> None:
    """
    Les quatre equipements du sujet, detectes et affiches.

    Le §2 du cahier des charges demande de DETECTER les quatre. Le §5.1
    demande d'identifier ceux a surveiller en premier. Ce sont deux
    exigences distinctes, et ce panneau les separe a l'ecran :

        les quatre sont mesures et affiches
        un seul interrompt l'operateur

    Sans ce panneau, le tableau de bord ne montrait que le casque, et
    laissait croire que les trois autres n'etaient pas traites.
    """
    from src.app.detection import (PERIMETRE_DEFAUT, SELON_LA_TACHE,
                                   charger_seuils)

    st.subheader(t("Les quatre equipements du sujet"))
    surveillees = donnees[donnees["statut"] == "surveillee"]
    if surveillees.empty:
        st.info(t("Aucune donnée."))
        return

    seuils = charger_seuils()
    lignes = []
    for classe, nom in SURVEILLES.items():
        colonne = f"conf_{classe}"
        if colonne not in surveillees:
            continue
        seuil = seuils.get(classe, 0.5)
        rattaches = int((surveillees[colonne] > 0).sum())
        retenus = int((surveillees[colonne] >= seuil).sum())
        lignes.append({
            t("Equipement"): t(nom),
            t("Seuil calibré"): f"{seuil:.3f}",
            t("Rattachés"): rattaches,
            t("Au-dessus du seuil"): retenus,
            t("Part des personnes"): f"{retenus / len(surveillees):.1%}",
            t("Alerte par défaut"):
                t("oui") if classe in PERIMETRE_DEFAUT else t("non"),
        })

    st.dataframe(lignes, hide_index=True, use_container_width=True)

    # ---- le perimetre d'alerte, choisi par l'exploitant ------------------
    st.markdown("**" + t("Périmètre d'alerte") + "**")
    st.caption(t("Chaque case indique ce qu'elle coûterait sur ce corpus. "
                 "Le casque est coché par défaut : c'est le seul équipement "
                 "obligatoire partout, et le seul dont les taux d'erreur "
                 "mesurés le permettent."))

    choisis = set()
    cases = st.columns(len(SURVEILLES))
    for colonne, (classe, nom) in zip(cases, SURVEILLES.items()):
        seuil = seuils.get(classe, 0.5)
        manquants = int((surveillees[f"conf_{classe}"] < seuil).sum())
        part = manquants / len(surveillees)
        etiquette_case = "%s — %d (%.0f %%)" % (t(nom), manquants, part * 100)
        with colonne:
            if st.checkbox(etiquette_case, value=classe in PERIMETRE_DEFAUT,
                           key=f"perimetre_{classe}"):
                choisis.add(classe)
            if classe in SELON_LA_TACHE:
                st.caption("⚠ " + t("requis selon la tâche"))

    total = sum(int((surveillees[f"conf_{c}"] < seuils.get(c, 0.5)).sum())
                for c in choisis)
    if not choisis:
        st.error(t("Aucun équipement sélectionné : le système n'alerterait "
                   "sur rien."))
    else:
        par_personne = total / len(surveillees)
        message = "**%d %s** — %.1f %s" % (
            total, t("alertes sur ce corpus"), par_personne,
            t("par personne surveillée"))
        (st.info if par_personne <= 0.5 else st.warning)(message)

    if SELON_LA_TACHE & choisis:
        st.warning(t(
            "Lunettes, gants et gilet ne sont pas exigés pour toutes les "
            "tâches : le Code du travail impose une évaluation par poste, "
            "consignée au document unique. Le système ne sait pas quelle "
            "tâche exécute la personne qu'il regarde. Sur le flux de "
            "chantier mesuré, alerter sur les lunettes signalerait **98 % "
            "des ouvriers** — un chiffre qui ne décrit pas une "
            "non-conformité, mais une exigence qui ne s'applique pas là."))

    graphe = pd.DataFrame({
        t("équipement"): [l[t("Equipement")] for l in lignes],
        t("personnes"): [l[t("Au-dessus du seuil")] for l in lignes],
    })
    st.bar_chart(graphe, x=t("équipement"), y=t("personnes"), height=220)

    st.caption(
        t("Les quatre équipements sont détectés, associés à une personne et "
          "comptés. Un seul — le casque — déclenche une alerte en phase "
          "pilote : c'est le seul dont les taux d'erreur mesurés le "
          "permettent. Les trois autres ont chacun une condition de retour "
          "chiffrée."))


def chronologie(donnees: pd.DataFrame) -> None:
    st.subheader(t("Chronologie des alertes"))
    st.caption(t("Horodatage simulé — la forme démontre le mécanisme, "
               "les volumes sont réels."))

    par_jour = (donnees.groupby(["jour", "camera"])["alerte"]
                .sum().reset_index()
                .rename(columns={"alerte": t("alertes")}))
    if par_jour.empty:
        st.info(t("Aucune donnée."))
        return
    st.bar_chart(par_jour, x="jour", y=t("alertes"), color="camera",
                 height=260)

    st.caption(t("Répartition horaire — utile pour dimensionner la "
               "surveillance humaine."))
    par_heure = (donnees.groupby("heure")["alerte"].sum()
                 .reset_index().rename(columns={"alerte": t("alertes")}))
    st.bar_chart(par_heure, x="heure", y=t("alertes"), height=200)


def carte_du_champ(donnees: pd.DataFrame) -> None:
    """
    Carte de chaleur des alertes DANS LE CHAMP DE LA CAMERA.

    C'est du reel : `x_relatif` et `y_relatif` sont la position mesuree de
    chaque personne dans l'image, en fraction de largeur et de hauteur.
    """
    import matplotlib.pyplot as plt

    st.subheader(t("Où, dans le champ, les alertes se produisent-elles ?"))
    st.caption(t("**Donnée réelle** — position mesurée de chaque personne dans "
               "l'image. Aucune simulation ici."))

    nature = st.session_state.get("_nature_corpus")
    for cle, fiche in CORPUS.items():
        if fiche["nature"] == nature and fiche["avertissement"]:
            (st.warning if nature == "photo" else st.success)(
                t(fiche["avertissement"]))
            break

    alertes = donnees[(donnees["statut"] == "surveillee") &
                      (donnees["alerte"] == 1)]
    if alertes.empty:
        st.info(t("Aucune alerte à cartographier."))
        return

    grille, _, _ = np.histogram2d(
        alertes["y_relatif"], alertes["x_relatif"],
        bins=[12, 16], range=[[0, 1], [0, 1]])

    figure, axes = plt.subplots(figsize=(7, 3.6), layout="constrained")
    image = axes.imshow(grille, cmap="inferno", aspect="auto",
                        extent=[0, 1, 1, 0], interpolation="bilinear")
    axes.set_xlabel(t("largeur du champ"))
    axes.set_ylabel(t("hauteur du champ"))
    axes.set_title(f"{len(alertes)} alertes")
    figure.colorbar(image, ax=axes, label=t("alertes"))
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    # --- la lecture operationnelle, chiffree ------------------------------
    haut = float((alertes["y_relatif"] < 0.33).mean())
    petites = float((donnees["hauteur_relative"] < HAUTEUR_MINIMALE).mean())
    colonnes = st.columns(2)
    colonnes[0].metric(t("Alertes dans le tiers supérieur"), f"{haut:.0%}",
                       help="signe de têtes coupées par le bord du cadre")
    colonnes[1].metric(t("Personnes trop petites pour être jugées"),
                       f"{petites:.0%}",
                       help=f"moins de {HAUTEUR_MINIMALE:.0%} de la hauteur")

    st.markdown(
        t("**Ce que cette carte commande.** Une concentration en haut du cadre "
        "signale des têtes hors champ : la caméra est trop basse ou trop "
        "proche. Une concentration au loin signale des personnes de quelques "
        "dizaines de pixels : le champ est trop large. Dans les deux cas, la "
        "correction est **le placement de la caméra**, pas un réglage du "
        "logiciel — c'est le constat mesuré du 24 août."))


def repartition(donnees: pd.DataFrame) -> None:
    st.subheader(t("Ce que le système peut juger, et ce qu'il ne peut pas"))
    compte = donnees["statut"].value_counts().reset_index()
    compte.columns = ["statut", t("personnes")]
    compte["statut"] = compte["statut"].map(etiquette)
    compte = compte.rename(columns={"statut": t("état")})
    gauche, droite = st.columns([1, 1])
    with gauche:
        st.dataframe(compte, hide_index=True, use_container_width=True)
    with droite:
        st.bar_chart(compte, x=t("état"), y=t("personnes"), height=220)
    st.caption(
        t("« Je ne peux pas juger » n'est pas « l'équipement manque ». Séparer "
        "les deux a supprimé 88 % des fausses alertes sur séquence réelle."))


def detail(donnees: pd.DataFrame, nom_modele: str = MODELE_DU_CORPUS) -> None:
    """
    Le drill-down : d'un chiffre agrege au cas individuel (C3.2-3).

    Le modele est passe en argument. Avant le 25/08 la descente au cas
    construisait `Detecteur()` sans argument, donc TOUJOURS yolov8m : on
    pouvait selectionner yolov8n et voir s'afficher les boites d'un autre
    modele. Le selecteur mentait sur ce qu'il commandait.
    """
    st.subheader(t("Descendre jusqu'au cas"))
    st.caption(t("Choisis une alerte : le système montre l'image et explique "
                 "sa décision, détection par détection."))
    alertes = donnees[(donnees["statut"] == "surveillee") &
                      (donnees["alerte"] == 1)].copy()
    if alertes.empty:
        st.info(t("Aucune alerte dans la sélection courante."))
        return

    alertes["libelle"] = (alertes["horodatage"].dt.strftime("%d/%m %H:%M")
                          + " · " + alertes["camera"]
                          + " · personne #" + alertes["identifiant"].astype(str)
                          + " · " + alertes["image"])
    choix = st.selectbox(t("Alerte"), alertes["libelle"].tolist())
    ligne = alertes[alertes["libelle"] == choix].iloc[0]

    gauche, droite = st.columns([2, 1], gap="medium")
    with droite:
        st.markdown(f"**Caméra** {ligne['camera']}  \n"
                    f"**Zone** {ligne['zone']}  \n"
                    f"**Image** `{ligne['image']}`  \n"
                    f"**Position** {ligne['x_relatif']:.2f} × "
                    f"{ligne['y_relatif']:.2f}  \n"
                    f"**Hauteur** {ligne['hauteur_relative']:.1%} du champ")
        st.markdown(t("**Confiances mesurées**"))
        for classe, nom in SURVEILLES.items():
            valeur = float(ligne.get(f"conf_{classe}", 0.0))
            st.text(f"{nom:9s} {valeur:.3f}" + ("  —" if valeur == 0 else ""))
        st.caption(t("Une confiance nulle signifie qu'aucun équipement de cette "
                   "classe n'a été rattaché à cette personne. C'est une "
                   "information, pas une donnée manquante."))

    with gauche:
        chemin = None
        for dossier in ["data/echantillon_app", "data/hors_domaine",
                        "data/demonstration"]:
            candidat = RACINE / dossier / str(ligne["image"])
            if candidat.is_file():
                chemin = candidat
                break
        if chemin is None:
            st.warning(f"Image introuvable : {ligne['image']}")
            return

        from src.app.detection import Detecteur
        from src.app.interface import afficher_image, annoter
        import cv2

        # Le cache est indexe sur le NOM du modele : sans cela, changer de
        # modele reutilisait le detecteur precedent, deja en memoire.
        cle_cache = f"_detecteur_detail_{nom_modele}"
        detecteur = st.session_state.get(cle_cache)
        if detecteur is None:
            detecteur = Detecteur(
                poids=RACINE / "models" / f"{nom_modele}.pt")
            st.session_state[cle_cache] = detecteur

        if nom_modele != MODELE_DU_CORPUS:
            st.info(t("La detection ci-dessous est rejouee avec %s, alors que "
                      "le corpus a ete produit par %s. Les boites peuvent "
                      "differer des confiances enregistrees a droite.")
                    % (nom_modele, MODELE_DU_CORPUS))

        image = cv2.imread(str(chemin))
        resultat = detecteur.analyser_image(image)
        afficher_image(annoter(image, resultat)[:, :, ::-1], chemin.name)

    st.code(f"python src/app/expliquer.py {chemin.relative_to(RACINE)}",
            language="bash")
    st.caption(t("La même explication, détection par détection, en ligne de "
               "commande."))


def page(nom_modele: str = MODELE_DU_CORPUS) -> None:
    disponibles = corpus_disponibles()
    if not disponibles:
        st.warning(t("Corpus absent. Génère-le d'abord :"))
        st.code("python src/app/corpus.py", language="bash")
        return

    if len(disponibles) > 1:
        cle = st.radio(t("Corpus"), list(disponibles),
                       format_func=lambda c: t(disponibles[c]["titre"]),
                       horizontal=True)
    else:
        cle = next(iter(disponibles))
    fiche = disponibles[cle]

    chemin = DOSSIER / f"{cle}.csv"
    donnees = charger(str(chemin), chemin.stat().st_mtime)
    bandeau_declaration()
    if fiche["avertissement"]:
        st.session_state["_nature_corpus"] = fiche["nature"]

    # --- filtres croises --------------------------------------------------
    with st.sidebar:
        st.divider()
        st.header(t("Filtres du tableau de bord"))
        toutes = sorted(donnees["camera"].unique())
        cameras = st.multiselect(t("Caméra"), toutes, default=toutes)
        # Les options sont les VALEURS de donnees ; seul l'affichage est
        # traduit, par `format_func`. C'est ce qui permet de changer de
        # langue sans casser le filtre.
        etats = st.multiselect(t("État"), list(ETATS),
                               format_func=lambda c: t(ETATS[c]),
                               default=list(ETATS))
        jours = sorted(donnees["jour"].unique())
        if len(jours) > 1:
            debut, fin = st.select_slider(
                t("Période"), options=jours, value=(jours[0], jours[-1]))
        else:
            debut = fin = jours[0]

    filtre = donnees[donnees["camera"].isin(cameras)
                     & donnees["statut"].isin(etats)
                     & (donnees["jour"] >= debut) & (donnees["jour"] <= fin)]
    if filtre.empty:
        st.warning(t("Aucun événement ne correspond aux filtres."))
        return

    if nom_modele != MODELE_DU_CORPUS:
        st.warning(t("Modele selectionne : %s. Les chiffres de cette page ont "
                     "ete calcules hors ligne avec %s et ne sont pas "
                     "recalcules. Le choix du modele n'agit que sur les "
                     "onglets Image et Video, et sur la descente au cas.")
                   % (nom_modele, MODELE_DU_CORPUS))

    indicateurs(filtre)
    st.divider()
    quatre_epi(filtre)
    st.divider()
    carte_du_champ(filtre)
    st.divider()
    chronologie(filtre)
    st.divider()
    repartition(filtre)
    st.divider()
    detail(filtre, nom_modele)
