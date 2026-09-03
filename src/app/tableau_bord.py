"""
Tableau de bord metier, indicateurs, chronologie, carte du champ, detail.

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
# (`corpus.py --modele yolov8m`). Les chiffres agreges de cette page en
# viennent et ne sont jamais recalcules a l'affichage.
#
# Le panneau de gauche n'offre plus de choix de modele : yolov8m est le
# resultat de la comparaison a budget egal, pas un reglage. Seul le repli
# automatique peut faire tourner autre chose, et seulement sur ce qui est
# analyse EN DIRECT : onglets Image et Video, et descente au cas. L'ecart
# est alors signale en tete de page.
MODELE_DU_CORPUS = "yolov8m"

# Chaque corpus dit ce qu'il est. La carte du champ ne signifie pas la meme
# chose selon la nature des images : sur des photographies composees, elle
# revele la regle des tiers ; sur un flux de camera fixe, elle revele le
# cadrage. Confondre les deux ferait passer une convention de cadrage
# photographique pour une zone a risque.
CORPUS = {
    "evenements": {
        "titre": "Validation SH17, photographies de stock",
        "nature": "photo",
        "avertissement":
            "Ces images sont des **photographies composées** (banque "
            "Pexels). Le point chaud de la carte tombe au centre du cadre "
            "parce que c'est là qu'un photographe place son sujet : **c'est"
            " la règle des tiers, pas une zone à risque**. Sur ce corpus, "
            "la carte mesure une convention photographique.",
    },
    "flux_camera": {
        "titre": "Flux de caméra, séquences de chantier réel",
        "nature": "camera",
        "avertissement":
            "Ces images sont des **captures de caméra fixe**. Ici, la carte"
            " mesure ce qu'elle doit mesurer : le cadrage. Une "
            "concentration en haut signale des têtes coupées, une "
            "concentration diffuse et faible signale des personnes trop "
            "lointaines.",
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
# `etiquette()`, pas ici, ou la langue choisie n'est pas encore connue.
ETATS = {"surveillee": "surveillée",
         "hors_perimetre": "hors périmètre",
         "tete_hors_champ": "tête hors champ"}


def etiquette(statut: str) -> str:
    """Le libelle affichable d'un statut, dans la langue courante."""
    return t(ETATS.get(statut, statut))


def perimetre_actif() -> set[str]:
    """
    Les equipements coches dans le panneau « Perimetre d'alerte ».

    Pourquoi lire `session_state` et non une variable
    -------------------------------------------------
    Les cases vivent dans `quatre_epi()`, sur l'onglet Tableau de bord, et
    le resultat doit servir dans `annoter()`, sur l'onglet Image. Passer la
    valeur d'une fonction a l'autre supposerait de connaitre leur ordre
    d'execution. Streamlit rejoue tout le script a chaque interaction, et
    cet ordre n'est pas garanti.

    Les cles des widgets, elles, survivent au rejeu et sont lisibles de
    partout. C'est le seul etat partage de l'application, et il est en
    lecture seule hors de `quatre_epi()`.

    Le defaut reste D-038, le casque seul : un exploitant qui ne touche a
    rien obtient exactement le comportement mesure et documente.
    """
    from src.app.detection import PERIMETRE_DEFAUT

    return {classe for classe in SURVEILLES
            if st.session_state.get(f"perimetre_{classe}",
                                    classe in PERIMETRE_DEFAUT)}


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


def bandeau_filtres(donnees: pd.DataFrame, filtre: pd.DataFrame,
                    cameras: list, zones: list, etats: list,
                    debut, fin) -> None:
    """
    Dire a l'ecran que les filtres existent, ou ils sont, et ce qu'ils font.

    Pourquoi ce bandeau
    -------------------
    Les quatre filtres vivent dans la barre laterale, et ils commandent les
    SIX sections d'un seul coup. Rien ne le disait. Un visiteur qui les
    apercoit peut les prendre pour des reglages locaux, et un visiteur dont
    la barre est repliee ne les voit pas du tout.

    Le bandeau affiche donc l'etat courant de la selection ET son effet sur
    le nombre de lignes retenues. Bouger un filtre change le chiffre sous
    les yeux : c'est la demonstration la plus courte que le croisement
    fonctionne.
    """
    total, retenu = len(donnees), len(filtre)
    morceaux = [
        "**%d/%d** %s" % (len(cameras), len(donnees["camera"].unique()),
                          t("caméras")),
        "**%d/%d** %s" % (len(zones), len(donnees["zone"].unique()),
                          t("zones")),
        "**%d/%d** %s" % (len(etats), len(ETATS), t("états")),
    ]
    if debut != fin:
        morceaux.append("%s **%s → %s**" % (t("du"), debut, fin))
    part = retenu / total if total else 0.0
    st.markdown(
        "🎛 **%s** (%s) : %s  \n%s **%s** %s **%s** %s (%.0f %%). %s"
        % (t("Filtres actifs"), t("barre latérale, à gauche"),
           " · ".join(morceaux),
           t("Sélection :"), f"{retenu:,}".replace(",", " "), t("personnes sur"),
           f"{total:,}".replace(",", " "), t("au total"), part * 100,
           t("Les six sections ci-dessous se recalculent ensemble.")))


def bandeau_declaration() -> None:
    """La phrase qui rend ce tableau de bord defendable."""
    st.info(
        t("**Ce que ces chiffres valent.** Les détections, verdicts, confiances "
        "et positions sont **réels**, produits par le modèle sur de vraies "
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
    # « de la selection » et non « du total » : `donnees` est la table DEJA
    # filtree. Avec les filtres grands ouverts les deux formulations disent
    # la meme chose, mais des qu'une zone est isolee le denominateur devient
    # celui de la selection. Ecrire « du total » y affichait un pourcentage
    # juste sous une phrase fausse.
    haut[2].metric(t("Non jugeables"), f"{non_jugeables:,}".replace(",", " "),
                   f"{non_jugeables / len(donnees):.0%} " + t("de la sélection")
                   if len(donnees) else None, delta_color="off")
    bas = st.columns(2)
    bas[0].metric(t("Taux de détection du casque"),
                  f"{(n - alertes) / n:.1%}" if n else ",")
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

    st.subheader(t("Les quatre équipements du sujet"))
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
            t("Équipement"): t(nom),
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
        etiquette_case = "%s, %d (%.0f %%)" % (t(nom), manquants, part * 100)
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
        message = "**%d %s**, %.1f %s" % (
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
            "des ouvriers**, un chiffre qui ne décrit pas une "
            "non-conformité, mais une exigence qui ne s'applique pas là."))

    graphe = pd.DataFrame({
        t("équipement"): [l[t("Équipement")] for l in lignes],
        t("personnes"): [l[t("Au-dessus du seuil")] for l in lignes],
    })
    st.bar_chart(graphe, x=t("équipement"), y=t("personnes"), height=220)

    st.caption(
        t("Les quatre équipements sont détectés, associés à une personne et "
          "comptés. Un seul (le casque) déclenche une alerte en phase "
          "pilote : c'est le seul dont les taux d'erreur mesurés le "
          "permettent. Les trois autres ont chacun une condition de retour "
          "chiffrée."))


def chronologie(donnees: pd.DataFrame) -> None:
    st.subheader(t("Chronologie des alertes"))
    st.caption(t("Horodatage simulé : la forme démontre le mécanisme, "
               "les volumes sont réels."))

    par_jour = (donnees.groupby(["jour", "camera"])["alerte"]
                .sum().reset_index()
                .rename(columns={"alerte": t("alertes")}))
    if par_jour.empty:
        st.info(t("Aucune donnée."))
        return
    st.bar_chart(par_jour, x="jour", y=t("alertes"), color="camera",
                 height=260)

    st.caption(t("Répartition horaire, utile pour dimensionner la "
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
    import altair as alt

    st.subheader(t("Où, dans le champ, les alertes se produisent-elles ?"))
    st.caption(t("**Donnée réelle**, position mesurée de chaque personne dans "
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

    # Le comptage reste identique : 12 bandes en hauteur, 16 en largeur.
    # Seul le rendu change. L'ancienne version passait par matplotlib, donc
    # par une IMAGE FIXE : on voyait la forme du point chaud sans jamais
    # pouvoir en lire la valeur. Ici chaque case est survolable et annonce
    # sa position dans le champ et son nombre d'alertes.
    #
    # L'interpolation bilineaire de matplotlib est perdue au passage, et
    # c'est un gain : elle lissait douze bandes en un degrade continu, ce
    # qui laissait croire a une mesure plus fine que les donnees.
    grille, bords_y, bords_x = np.histogram2d(
        alertes["y_relatif"], alertes["x_relatif"],
        bins=[12, 16], range=[[0, 1], [0, 1]])

    cases = pd.DataFrame([
        {"x1": bords_x[j], "x2": bords_x[j + 1],
         "y1": bords_y[i], "y2": bords_y[i + 1],
         "alertes": int(grille[i, j]),
         "bande": "%.0f à %.0f %% × %.0f à %.0f %%" % (
             bords_x[j] * 100, bords_x[j + 1] * 100,
             bords_y[i] * 100, bords_y[i + 1] * 100)}
        for i in range(grille.shape[0]) for j in range(grille.shape[1])])

    # `inferno` est repris tel quel de l'ancienne carte : c'est une echelle
    # perceptuellement uniforme, donc lisible en protanopie comme en
    # deutéranopie, et les figures du rapport restent comparables.
    carte = (
        alt.Chart(cases, height=260)
        .mark_rect()
        .encode(
            x=alt.X("x1:Q", title=t("largeur du champ"),
                    scale=alt.Scale(domain=[0, 1], nice=False),
                    axis=alt.Axis(format="%")),
            x2="x2:Q",
            # Repere image : l'origine est EN HAUT a gauche, comme les
            # coordonnees YOLO. Sans `reverse`, la carte serait retournee
            # et « le tiers superieur » designerait le bas du cadre.
            y=alt.Y("y1:Q", title=t("hauteur du champ"),
                    scale=alt.Scale(domain=[0, 1], nice=False, reverse=True),
                    axis=alt.Axis(format="%")),
            y2="y2:Q",
            color=alt.Color("alertes:Q", title=t("alertes"),
                            scale=alt.Scale(scheme="inferno")),
            tooltip=[alt.Tooltip("bande:N", title=t("zone du champ")),
                     alt.Tooltip("alertes:Q", title=t("alertes"))],
        )
    )
    st.altair_chart(carte, use_container_width=True)
    st.caption("%d %s" % (len(alertes), t("alertes cartographiées. "
               "Survole une case pour lire sa position et son compte.")))

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
        "logiciel : c'est le constat mesuré du 24 août."))


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

    Le modele est passe en argument, et non reconstruit ici. Historique de
    ce choix : jusqu'au 25/08, la descente au cas construisait `Detecteur()`
    sans argument, donc TOUJOURS yolov8m, alors qu'un selecteur permettait
    encore de demander yolov8n. L'ecran affichait les boites d'un modele et
    le nom d'un autre.

    Le selecteur a depuis ete retire, mais le passage par argument est
    conserve : le repli automatique peut encore faire tourner yolov8n, et
    l'image montree doit venir du modele qui tourne vraiment.
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
        # Les libelles passent par `t()`, les valeurs non : ce sont des
        # donnees. Une f-string melangeait les deux et laissait ce panneau
        # en francais dans l'interface anglaise, seul ecran a l'etre.
        st.markdown("**%s** %s  \n**%s** %s  \n**%s** `%s`  \n"
                    "**%s** %.2f × %.2f  \n**%s** %s"
                    % (t("Caméra"), ligne["camera"],
                       t("Zone"), ligne["zone"],
                       t("Image"), ligne["image"],
                       t("Position"), ligne["x_relatif"], ligne["y_relatif"],
                       t("Hauteur"),
                       t("%.1f %% du champ") % (ligne["hauteur_relative"] * 100)))
        st.markdown(t("**Confiances mesurées**"))
        for classe, nom in SURVEILLES.items():
            valeur = float(ligne.get(f"conf_{classe}", 0.0))
            st.text(f"{nom:9s} {valeur:.3f}" + (" ," if valeur == 0 else ""))
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
            st.info(t("La détection ci-dessous est rejouée avec %s, alors "
                      "que le corpus a été produit par %s. Les boîtes "
                      "peuvent différer des confiances enregistrées à "
                      "droite.")
                    % (nom_modele, MODELE_DU_CORPUS))

        image = cv2.imread(str(chemin))
        resultat = detecteur.analyser_image(image)
        afficher_image(annoter(image, resultat, perimetre_actif())[:, :, ::-1],
                       chemin.name)

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
        # Le §4.1.2 du sujet demande des filtres « par type d'EPI, par zone
        # du chantier, par periode ». La zone existait dans les donnees et
        # s'affichait dans la descente au cas, mais on ne pouvait pas
        # filtrer dessus. Or c'est la maille du chef de chantier : il pense
        # « zone de levage », pas « CAM-03 ».
        #
        # MAIS les deux filtres ne sont pas independants : dans ce corpus,
        # une camera couvre une zone et une seule. Les cacher l'un a l'autre
        # produirait une intersection vide des qu'on croise CAM-01 avec une
        # autre zone que la sienne, sans qu'on comprenne pourquoi. On affiche
        # donc l'appariement dans le libelle, et on le declare sous les
        # filtres. Deux entrees vers la meme selection, une par metier.
        appariement = (donnees.groupby("camera")["zone"]
                       .agg(lambda z: sorted(set(z))).to_dict())
        toutes = sorted(donnees["camera"].unique())
        cameras = st.multiselect(
            t("Caméra"), toutes, default=toutes,
            format_func=lambda c: "%s · %s" % (c, ", ".join(appariement[c])))
        zones_toutes = sorted(donnees["zone"].unique())
        zones = st.multiselect(t("Zone du chantier"), zones_toutes,
                               default=zones_toutes)
        if all(len(z) == 1 for z in appariement.values()):
            st.caption(t("Une caméra couvre ici une zone et une seule : ces "
                         "deux filtres sont deux entrées vers la même "
                         "sélection, l'une pour l'exploitant technique, "
                         "l'autre pour le chef de chantier. Les croiser sur "
                         "des valeurs qui ne se correspondent pas donne un "
                         "résultat vide, et c'est normal."))
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
                     & donnees["zone"].isin(zones)
                     & donnees["statut"].isin(etats)
                     & (donnees["jour"] >= debut) & (donnees["jour"] <= fin)]
    bandeau_filtres(donnees, filtre, cameras, zones, etats, debut, fin)
    if filtre.empty:
        st.warning(t("Aucun événement ne correspond aux filtres."))
        return

    # Ce cas n'est plus atteignable par un choix de l'utilisateur : le
    # selecteur de modele a ete retire, le modele est desormais affiche et
    # non reglable. Il reste atteignable par le REPLI AUTOMATIQUE, quand
    # yolov8m n'a pas pu etre charge. Le message est donc reformule : il ne
    # parle plus d'un choix, il constate un ecart.
    if nom_modele != MODELE_DU_CORPUS:
        st.warning(t("Le modèle actif est %s, alors que les chiffres de "
                     "cette page ont été calculés hors ligne avec %s. Ils "
                     "ne sont pas recalculés. Seuls les onglets Image et "
                     "Vidéo, et la descente au cas, tournent avec le modèle "
                     "actif.")
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
