# Detection du port des EPI sur chantier — application

**BC02 · Projet 1 — RNCP40875, Ecole 18.06**

Application de demonstration du systeme de detection d'equipements de
protection individuelle. Elle accompagne le rapport technique du projet.

## Ce que fait l'application

| Onglet | Role |
|---|---|
| Tableau de bord | statistiques du corpus, chronologie, carte du champ, descente au cas |
| Image | depose une photographie, verdict par personne |
| Video | depose une sequence : hysteresis, confirmation, agregation |
| Guide | mode d'emploi integre |
| Limites | ce que le systeme ne garantit pas — **a lire en premier** |

## Ce qu'elle ne fait pas

Elle **n'atteste pas la conformite**. Elle atteste la presence d'un objet
ressemblant a un casque. Au reglage actuel, deux alertes sur trois sont
fausses : le systeme sert a orienter un controle humain, jamais a
sanctionner.

Trois etats, et non deux : « surveillee », « hors perimetre »,
« tete hors champ ». *Je ne peux pas juger* n'est pas *l'equipement
manque*.

## Modeles

`yolov8m` est le modele retenu. `yolov8n` est le repli si la memoire de
l'hebergement est insuffisante — huit fois plus leger, moins precis. Le
selecteur est dans le panneau de gauche.

Les seuils sont **calibres par classe** sur les donnees de validation, et
non choisis a la main.

## Lancer en local

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m streamlit run streamlit_app.py
```

## Donnees

Voir `ATTRIBUTION.md`. Le corpus du tableau de bord melange des detections
**reelles** et un contexte **simule declare** (camera, zone, horodatage) :
le projet ne disposait pas de metadonnees de chantier. La distinction est
affichee dans l'application.
