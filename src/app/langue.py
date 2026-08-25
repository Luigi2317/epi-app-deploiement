"""
Bascule francais / anglais de l'interface.

Pourquoi cette forme, et pas une autre
---------------------------------------
La traduction est indexee par LE TEXTE FRANCAIS LUI-MEME, et non par des
cles abstraites du genre `page.image.titre`.

    t("Personnes vues")     ->  "People seen"

Consequence pratique : le code reste lisible sans dictionnaire sous les
yeux, et une chaine sans traduction s'affiche en francais au lieu de
provoquer une erreur ou d'afficher une cle technique a l'utilisateur.

    Une traduction manquante degrade l'affichage. Elle ne le casse pas.

Ce que couvre la traduction, et ce qu'elle ne couvre pas
--------------------------------------------------------
COUVERT : tout ce que lit un utilisateur de l'outil — titres, indicateurs,
messages, legendes, avertissements, page des limites.

NON COUVERT : les journaux techniques, les noms de fichiers et les
docstrings du code. Ils s'adressent a qui maintient le systeme, pas a qui
l'exploite, et les traduire creerait deux versions a maintenir.

Le brief demande des langues multiples « SI NECESSAIRE » (§4.1.2). La
justification retenue est inscrite dans le registre de decisions : sur un
chantier francais, une part significative des compagnons ne lit pas le
francais, et une consigne de securite mal comprise ne protege personne.
"""

from __future__ import annotations

import streamlit as st

LANGUES = {"fr": "Français", "en": "English"}
DEFAUT = "fr"

# --------------------------------------------------------------------------
# Francais -> anglais. Seules les chaines vues par l'utilisateur y figurent.
# --------------------------------------------------------------------------
TRADUCTIONS = {
    # --- titres et navigation ---
    "Detection d'equipements de protection": "Personal protective equipment detection",
    "Le modele detecte des objets. Les regles produisent le verdict.":
        "The model detects objects. The rules produce the verdict.",
    "Tableau de bord": "Dashboard",
    "Image": "Image",
    "Video": "Video",
    "Limites": "Limitations",
    "Reglages": "Settings",
    "Modele": "Model",
    "Corpus": "Corpus",

    # --- indicateurs ---
    "Personnes vues": "People seen",
    "Surveillees": "Monitored",
    "Surveillées": "Monitored",
    "Non jugeables": "Cannot be judged",
    "[OK] Casque detecte": "[OK] Helmet detected",
    "[!] Casque non detecte": "[!] Helmet not detected",
    "Duree": "Duration",
    "Taux de détection du casque": "Helmet detection rate",
    "[!] Alertes": "[!] Alerts",
    "Alerte": "Alert",
    "Personne": "Person",
    "Etat": "State",
    "État": "State",

    # --- etats ---
    "surveillee": "monitored",
    "surveillée": "monitored",
    "[-] hors perimetre": "[-] out of range",
    "hors périmètre": "out of range",
    "[?] tete hors champ": "[?] head out of frame",
    "tête hors champ": "head out of frame",

    # --- equipements ---
    "casque": "helmet", "lunettes": "eye protection",
    "gants": "gloves", "gilet": "hi-vis vest",

    # --- pages ---
    "Analyse d'une image": "Single image analysis",
    "Analyse d'une video": "Video analysis",
    "Photographie de chantier": "Construction site photograph",
    "Sequence de chantier": "Construction site footage",
    "Depose une image pour lancer l'analyse.":
        "Drop an image to start the analysis.",
    "Depose une video pour lancer l'analyse.":
        "Drop a video to start the analysis.",
    "Image illisible.": "Unreadable image.",
    "Analyse en cours…": "Analysing…",
    "Analyser une image sur": "Analyse one frame out of",
    "Personnes suivies": "People tracked",
    "Basculements de verdict": "Verdict flips",
    "Alertes": "Alerts",
    "Aucune alerte confirmee sur cette sequence.":
        "No confirmed alert in this sequence.",
    "Aucune personne detectee : aucun verdict ne peut etre rendu.":
        "No person detected: no verdict can be given.",
    "Ce que ce systeme ne garantit pas":
        "What this system does not guarantee",
    "Où, dans le champ, les alertes se produisent-elles ?":
        "Where in the frame do alerts occur?",
    "Chronologie des alertes": "Alert timeline",
    "Descendre jusqu'au cas": "Drill down to the individual case",
    "Ce que le système peut juger, et ce qu'il ne peut pas":
        "What the system can judge, and what it cannot",
    "Aucune alerte à cartographier.": "No alert to map.",
    "Aucune alerte dans la sélection courante.":
        "No alert in the current selection.",
    "Aucun événement ne correspond aux filtres.":
        "No event matches the filters.",
    "Filtres du tableau de bord": "Dashboard filters",
    "Caméra": "Camera", "Zone": "Zone", "Période": "Period",
    "largeur du champ": "frame width",
    "hauteur du champ": "frame height",
    "alertes": "alerts",
    "personnes": "people",
    "état": "state",

    # --- messages longs ---
    "Legende": "Key",
    "Alertes dans le tiers supérieur": "Alerts in the upper third",
    "Personnes trop petites pour être jugées":
        "People too small to be judged",
    "Corpus absent. Génère-le d'abord :": "Corpus missing. Generate it first:",

    # --- reperes a l'ecran le 26 aout, lors de la relecture bilingue ---
    "[OK] surveillee":
        "[OK] monitored",
    "Seuils calibres par classe — mesures, non choisis.":
        "Per-class calibrated thresholds — measured, not chosen.",
    "Decision en video":
        "Video decision",
    "Validation SH17 — photographies de stock":
        "SH17 validation — stock photographs",
    "Flux de camera — sequences de chantier reel":
        "Camera feed — real construction site footage",
    "Répartition horaire — utile pour dimensionner la surveillance humaine.":
        "Hourly distribution — useful for sizing human supervision.",
    "Horodatage simulé — la forme démontre le mécanisme, les volumes sont réels.":
        "Simulated timestamps — the shape demonstrates the mechanism, the volumes are real.",
    "Choisis une alerte : le système montre l'image et explique sa décision, détection par détection.":
        "Pick an alert: the system shows the image and explains its decision, detection by detection.",
    "Confiances mesurées":
        "Measured confidences",
    "Aucune donnée.":
        "No data.",
    "Corpus":
        "Corpus",

    # --- messages longs, traduits le 26 aout ---
    # Brouillon MyMemory, puis relecture terme a terme : la machine
    # rendait « confiance » par *trust*, « carte » par *card* et
    # « calibre » par *caliber*. Corrige a la main.
    "**Ce que ces chiffres valent.** Les détections, verdicts, confiances et positions sont **réels** — produits par le modèle sur de vraies images. L'**horodatage, la caméra et la zone sont simulés** : le système n'a jamais été déployé, il n'existe donc aucun historique. Le contexte est reconstitué pour démontrer les mécanismes ; les chiffres qui engagent, eux, sont mesurés.":
        "**What these figures are worth.** The detections, verdicts, confidence scores and positions are **real** — produced by the model on genuine images. The **timestamp, camera and zone are simulated**: the system has never been deployed, so no history exists. The context is reconstructed to demonstrate the mechanisms; the figures that commit us are measured.",
    "**Ce que cette carte commande.** Une concentration en haut du cadre signale des têtes hors champ : la caméra est trop basse ou trop proche. Une concentration au loin signale des personnes de quelques dizaines de pixels : le champ est trop large. Dans les deux cas, la correction est **le placement de la caméra**, pas un réglage du logiciel — c'est le constat mesuré du 24 août.":
        "**What this map calls for.** A concentration at the top of the frame indicates heads out of frame: the camera is too low or too close. A concentration far away indicates people only a few dozen pixels tall: the field of view is too wide. In both cases the fix is **camera placement**, not a software setting — this is the measured finding of 24 August.",
    "**Confiances mesurées**":
        "**Measured confidence scores**",
    "**Donnée réelle** — position mesurée de chaque personne dans l'image. Aucune simulation ici.":
        "**Real data** — measured position of each person in the image. No simulation here.",
    "**Legende** — la couleur ne porte jamais l'information seule : chaque boite affiche aussi un symbole. `[OK]` casque detecte · `[!]` casque non detecte · `[?]` tete hors champ, verdict impossible · `[-]` hors perimetre de surveillance.":
        "**Key** — colour never carries the information alone: every box also shows a symbol. `[OK]` helmet detected · `[!]` helmet not detected · `[?]` head out of frame, verdict impossible · `[-]` outside the monitoring range.",
    "**Un taux bas est ici attendu, et il est juste.** Le jeu d'images de validation ne contient que 773 casques pour 11 063 personnes : la plupart des gens photographiés n'en portent réellement pas. Ce taux décrit la composition du corpus, non un dysfonctionnement du système.":
        "**A low rate is expected here, and it is correct.** The validation image set contains only 773 helmets for 11,063 people: most of the people photographed genuinely are not wearing one. This rate describes the composition of the corpus, not a malfunction of the system.",
    "3 correspond au reglage utilise pour les mesures du J11.":
        "3 is the setting used for the day-11 measurements.",
    "C'est ici qu'agissent les trois mecanismes du J11 : deux seuils au lieu d'un, une confirmation sur images consecutives, et une alerte par personne et par episode.":
        "This is where the three day-11 mechanisms apply: two thresholds instead of one, confirmation over consecutive frames, and one alert per person and per episode.",
    "La même explication, détection par détection, en ligne de commande.":
        "The same explanation, detection by detection, from the command line.",
    "Sur une image isolee, il n'y a pas d'historique : le verdict est rendu par comparaison au seuil calibre de chaque classe. L'hysteresis ne s'applique qu'a la video.":
        "On a single image there is no history: the verdict is given by comparing against the calibrated threshold of each class. Hysteresis applies to video only.",
    "Une confiance nulle signifie qu'aucun équipement de cette classe n'a été rattaché à cette personne. C'est une information, pas une donnée manquante.":
        "A zero confidence score means no equipment of that class was linked to this person. That is information, not missing data.",
    "yolov8m est le modele retenu. yolov8n est le repli si la memoire d'hebergement est insuffisante.":
        "yolov8m is the selected model. yolov8n is the fallback if hosting memory is insufficient.",
    "« Je ne peux pas juger » n'est pas « l'équipement manque ». Séparer les deux a supprimé 88 % des fausses alertes sur séquence réelle.":
        "« I cannot judge » is not « the equipment is missing ». Separating the two removed 88 % of false alerts on real footage.",

    # --- avertissements de lecture des corpus ---
    "Ces images sont des **photographies composees** (banque Pexels). Le point chaud de la carte tombe au centre du cadre parce que c'est la qu'un photographe place son sujet — **c'est la regle des tiers, pas une zone a risque**. Sur ce corpus, la carte mesure une convention photographique.":
        "These images are **composed photographs** (Pexels stock library). The hot spot falls at the centre of the frame because that is where a photographer places the subject — **this is the rule of thirds, not a risk zone**. On this corpus the map measures a photographic convention, nothing more.",
    "Ces images sont des **captures de camera fixe**. Ici, la carte mesure ce qu'elle doit mesurer : le cadrage. Une concentration en haut signale des tetes coupees, une concentration diffuse et faible signale des personnes trop lointaines.":
        "These images are **fixed-camera captures**. Here the map measures what it should: framing. A concentration at the top indicates heads cut off by the frame; a faint, spread-out concentration indicates people too far away.",


    # --- panneau des quatre EPI ---
    "Les quatre equipements du sujet":
        "The four items of equipment",
    "Equipement":
        "Equipment",
    "Seuil calibré":
        "Calibrated threshold",
    "Rattachés":
        "Linked",
    "Au-dessus du seuil":
        "Above threshold",
    "Part des personnes":
        "Share of people",
    "Déclenche une alerte":
        "Raises an alert",
    "oui":
        "yes",
    "non":
        "no",
    "équipement":
        "equipment",
    "Les quatre équipements sont détectés, associés à une personne et comptés. Un seul — le casque — déclenche une alerte en phase pilote : c'est le seul dont les taux d'erreur mesurés le permettent. Les trois autres ont chacun une condition de retour chiffrée.":
        "All four items are detected, linked to a person and counted. Only one — the helmet — raises an alert during the pilot phase: it is the only one whose measured error rates allow it. Each of the other three has a measured condition for returning to the alert scope.",


    # --- perimetre d'alerte configurable ---
    "Périmètre d'alerte":
        "Alert scope",
    "Alerte par défaut":
        "Alerts by default",
    "requis selon la tâche":
        "required depending on the task",
    "alertes sur ce corpus":
        "alerts on this corpus",
    "par personne surveillée":
        "per monitored person",
    "Aucun équipement sélectionné : le système n'alerterait sur rien.":
        "No equipment selected: the system would alert on nothing.",
    "Chaque case indique ce qu'elle coûterait sur ce corpus. Le casque est coché par défaut : c'est le seul équipement obligatoire partout, et le seul dont les taux d'erreur mesurés le permettent.":
        "Each box shows what it would cost on this corpus. The helmet is ticked by default: it is the only item mandatory everywhere, and the only one whose measured error rates allow it.",
    "Lunettes, gants et gilet ne sont pas exigés pour toutes les tâches : le Code du travail impose une évaluation par poste, consignée au document unique. Le système ne sait pas quelle tâche exécute la personne qu'il regarde. Sur le flux de chantier mesuré, alerter sur les lunettes signalerait **98 % des ouvriers** — un chiffre qui ne décrit pas une non-conformité, mais une exigence qui ne s'applique pas là.":
        "Eye protection, gloves and hi-vis vests are not required for every task: French labour law requires a per-role risk assessment recorded in the site risk register. The system does not know which task the person it is looking at is performing. On the measured site footage, alerting on eye protection would flag **98 % of workers** — a figure that describes not non-compliance, but a requirement that does not apply there.",


    # --- guide integre ---
    "Guide":
        "Guide",
    "Guide d'utilisation":
        "User guide",


    # --- selecteur de modele et origine des seuils (D-051) ---
    "Aucun poids de modele trouve dans le dossier models.":
        "No model weights found in the models folder.",
    "**Seuils calibres par classe** — mesures, non choisis.":
        "**Per-class calibrated thresholds** — measured, not chosen.",
    "**Seuils par defaut** — ce modele n'a pas ete calibre. Valeurs de repli, non mesurees.":
        "**Default thresholds** — this model was not calibrated. Fallback values, not measured.",
    "La detection ci-dessous est rejouee avec %s, alors que le corpus a ete produit par %s. Les boites peuvent differer des confiances enregistrees a droite.":
        "The detection below is replayed with %s, whereas the corpus was produced by %s. Boxes may differ from the confidence scores recorded on the right.",
    "Modele selectionne : %s. Les chiffres de cette page ont ete calcules hors ligne avec %s et ne sont pas recalcules. Le choix du modele n'agit que sur les onglets Image et Video, et sur la descente au cas.":
        "Selected model: %s. The figures on this page were computed offline with %s and are not recalculated. The model choice only affects the Image and Video tabs, and the drill-down.",


    # --- repli automatique de modele (D-052) ---
    "Repli automatique sur %s : le modele retenu n'a pas pu etre charge. Les resultats sont ceux d'un modele plus leger et moins precis.":
        "Automatic fallback to %s: the selected model could not be loaded. These results come from a lighter, less accurate model.",
    "Motif technique":
        "Technical reason",
    "Modele retenu apres comparaison de six architectures a budget de calcul egal.":
        "Model retained after comparing six architectures at equal compute budget.",

}


def choisir() -> str:
    """Le selecteur, a placer dans le panneau lateral."""
    if "langue" not in st.session_state:
        st.session_state["langue"] = DEFAUT
    st.session_state["langue"] = st.radio(
        "Langue / Language", list(LANGUES),
        format_func=lambda c: LANGUES[c], horizontal=True,
        index=list(LANGUES).index(st.session_state["langue"]))
    return st.session_state["langue"]


def t(texte: str) -> str:
    """
    Traduit si l'anglais est actif et si la traduction existe.

    Le repli sur le francais est volontaire : une chaine oubliee reste
    lisible. Lever une exception ou afficher une cle technique serait pire
    pour l'utilisateur que de lire une phrase dans l'autre langue.
    """
    if st.session_state.get("langue", DEFAUT) == "fr":
        return texte
    return TRADUCTIONS.get(texte, texte)


def couverture() -> tuple[int, int]:
    """Nombre de chaines traduites — affiche dans la page des limites."""
    return len(TRADUCTIONS), len(TRADUCTIONS)
