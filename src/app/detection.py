"""
Le pont entre le modele et la decision.

Ce que ce module relie
----------------------
Trois briques existent deja, chacune verifiee separement :

    le modele YOLO ......... produit des boites et des confiances
    regles.py .............. relie un equipement a une personne
    decision.py ............ transforme une suite de confiances en verdict

Aucune ne parle aux deux autres. Ce module est le seul endroit ou elles se
rencontrent, et il ne contient AUCUNE regle metier propre : il orchestre.

    Si une regle apparait ici, c'est qu'elle est au mauvais endroit.

C'est ce qui permet a l'interface Streamlit de ne contenir que de
l'affichage, et a la logique de rester testable sans navigateur.

La difference entre une image et une video, et pourquoi elle compte
--------------------------------------------------------------------
UNE IMAGE ISOLEE N'A PAS D'HISTORIQUE. L'hysteresis du J11 compare la
confiance actuelle au verdict precedent ; sur une photographie, il n'y a pas
de verdict precedent. Appliquer deux seuils n'aurait aucun sens.

    image  ->  seuil calibre par classe (J10), une decision immediate
    video  ->  hysteresis + confirmation + agregation (J11)

Ce sont donc deux chemins distincts, et c'est voulu. Presenter le meme
mecanisme dans les deux cas serait plus simple a expliquer, et faux.

Le perimetre d'alerte
---------------------
Les quatre equipements sont detectes, associes et comptes. Le casque seul
alerte PAR DEFAUT (D-038), parce qu'il est le seul obligatoire partout et le
seul dont les taux d'erreur mesures le permettent.

Depuis le 26 aout (D-048), le perimetre est REGLABLE par l'exploitant, avec
le cout mesure affiche a cote de chaque equipement. Deux raisons :

    1. les seuils des quatre classes existent — les figer dans le code
       reviendrait a decider a la place de qui connait le chantier ;

    2. lunettes, gants et gilet ne sont pas exiges pour toutes les taches.
       Le Code du travail impose une evaluation par poste, consignee au
       document unique. Le systeme ignore quelle tache execute la personne
       qu'il regarde : alerter partout serait faux MEME AVEC UN MODELE
       PARFAIT.

Mesure a l'appui : sur le flux de chantier, alerter sur les lunettes
signalerait 98 % des ouvriers. Ce chiffre ne decrit pas une
non-conformite — il decrit une exigence qui ne s'applique pas la.

Vocabulaire
-----------
Le systeme constate une ABSENCE DE DETECTION, il ne prononce pas une
non-conformite (D-032). Les libelles produits ici disent « casque non
detecte », jamais « ouvrier non conforme ».
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from src.app.decision import MoteurDecision, Reglages
from src.app.regles import (EQUIPEMENTS_CORPS, EQUIPEMENTS_TETE, associer,
                            hors_perimetre, tete_hors_champ)

RACINE = Path(__file__).resolve().parents[2]

# Plancher du seuil de detection brut. Volontairement tres bas : on veut
# que le modele PROPOSE et que la decision TRANCHE.
#
# Corrige le 24 aout, apres echec d'un test. La valeur precedente — 0,20 —
# etait SUPERIEURE au seuil calibre du gilet (0,1456) : toute detection de
# gilet entre 0,1456 et 0,20 etait jetee avant d'atteindre la decision.
# C'etait exactement le defaut corrige au J10, reintroduit par la porte de
# derriere, et invisible dans les resultats.
#
# Le detecteur calcule desormais son seuil brut a partir des seuils
# calibres eux-memes (voir `Detecteur.__init__`) : si un recalibrage
# abaisse une classe, le filtre suit tout seul.
CONFIANCE_BRUTE = 0.10

# Le suivi, lui, exige un minimum de solidite : nourri de bruit, il
# fabrique une identite par fausse detection. Mesure au J11 : 96 identites
# pour 17 personnes reelles a 0,001, contre 65 a 0,25.
CONFIANCE_SUIVI = 0.25

# Perimetre d'alerte PAR DEFAUT (D-038, revu par D-048).
#
# Le casque seul, parce qu'il est le seul obligatoire partout sur un
# chantier et le seul dont les taux d'erreur mesures le permettent.
#
# Ce n'est plus une constante figee : l'exploitant peut elargir le
# perimetre, avec le cout affiche a cote de chaque case. Le systeme ne
# decide pas a la place d'un responsable HSE qui, lui, connait les taches
# de son chantier — mais il ne le laisse pas choisir a l aveugle.
PERIMETRE_DEFAUT = {"helmet"}
PERIMETRE_ALERTE = set(PERIMETRE_DEFAUT)      # compatibilite ascendante

# Ce que le Code du travail impose PARTOUT, contre ce qui depend de la
# tache evaluee au document unique. Cette distinction n'est pas technique :
# alerter sur les lunettes partout serait faux meme avec un modele parfait.
OBLIGATOIRE_PARTOUT = {"helmet"}
SELON_LA_TACHE = {"glasses", "gloves", "safety-vest"}

# Repli si le fichier de calibrage est absent : les seuils mesures au J10.
SEUILS_DEFAUT = {
    "helmet": 0.4653,
    "glasses": 0.4516,
    "gloves": 0.4193,
    "safety-vest": 0.1456,
}


def charger_seuils(modele: str = "yolov8m") -> dict[str, float]:
    """
    Lit les seuils calibres par classe produits au J10.

    Un seuil unique pour toutes les classes etait le premier defaut du
    systeme (« il fallait regler un curseur a la main »). Chaque classe a
    donc le sien, calcule sur les donnees de validation.
    """
    fichier = RACINE / "resultats" / "evaluation" / f"calibrage_{modele}.json"
    if not fichier.is_file():
        return dict(SEUILS_DEFAUT)

    mesure = json.loads(fichier.read_text(encoding="utf-8"))
    seuils = dict(SEUILS_DEFAUT)
    for classe, donnees in mesure.get("classes", {}).items():
        point = donnees.get("points", {}).get("f1_max")
        if point and "seuil" in point:
            seuils[classe] = float(point["seuil"])
    return seuils


def calibrage_disponible(modele: str) -> bool:
    """
    Ce modele a-t-il ses propres seuils mesures ?

    Sert a dire la verite dans l'interface : afficher « seuils calibres »
    au-dessus de valeurs de repli serait un mensonge d'affichage.
    """
    fichier = RACINE / "resultats" / "evaluation" / f"calibrage_{modele}.json"
    return fichier.is_file()


class Statut(str, Enum):
    """
    Ce que le systeme peut honnetement dire d'une personne.

    Trois etats, et non deux : « je ne peux pas juger » n'est pas la meme
    chose que « l'equipement manque ». Les confondre produit des alertes
    sur des situations que le systeme n'a tout simplement pas regardees.
    """
    SURVEILLEE = "surveillee"              # juge normalement
    HORS_PERIMETRE = "hors_perimetre"      # trop loin pour etre jugee
    TETE_HORS_CHAMP = "tete_hors_champ"    # tete coupee par le bord haut


@dataclass
class PersonneVue:
    """Une personne sur une image, et l'etat de ses equipements."""
    identifiant: int
    boite: np.ndarray
    confiance: float
    equipements: dict[str, float] = field(default_factory=dict)
    verdicts: dict[str, bool] = field(default_factory=dict)
    statut: Statut = Statut.SURVEILLEE

    @property
    def alertable(self) -> bool:
        """Seule une personne effectivement surveillee peut declencher."""
        return self.statut is Statut.SURVEILLEE

    def manque(self, classe: str, seuil: float) -> bool:
        """Cet equipement manque-t-il, pour cette personne ?"""
        return self.alertable and self.equipements.get(classe, 0.0) < seuil

    @property
    def manque_casque(self) -> bool:
        """
        Vrai uniquement si le systeme a REGARDE et n'a rien vu.

        Une personne hors perimetre ou dont la tete sort du cadre ne
        « manque » pas de casque : on n'en sait rien.
        """
        return self.alertable and not self.verdicts.get("helmet", False)

    def libelle(self) -> str:
        """Formulation prudente : une absence de detection, pas un jugement."""
        if self.statut is Statut.HORS_PERIMETRE:
            return (f"hors perimetre — personne {self.identifiant} "
                    f"trop eloignee pour etre jugee")
        if self.statut is Statut.TETE_HORS_CHAMP:
            return (f"verdict impossible — tete de la personne "
                    f"{self.identifiant} hors du champ")
        if self.manque_casque:
            return f"casque non detecte — personne {self.identifiant}"
        return f"casque detecte — personne {self.identifiant}"


@dataclass
class ResultatImage:
    """Ce qu'une image produit, une fois tout assemble."""
    personnes: list[PersonneVue]
    equipements_orphelins: int          # detectes, rattaches a personne
    largeur: int
    hauteur: int

    @property
    def nombre_sans_casque(self) -> int:
        return sum(1 for p in self.personnes if p.manque_casque)

    @property
    def surveillees(self) -> list:
        return [p for p in self.personnes if p.alertable]

    @property
    def nombre_hors_perimetre(self) -> int:
        return sum(1 for p in self.personnes
                   if p.statut is Statut.HORS_PERIMETRE)

    @property
    def nombre_tete_hors_champ(self) -> int:
        return sum(1 for p in self.personnes
                   if p.statut is Statut.TETE_HORS_CHAMP)


class Detecteur:
    """
    Charge le modele une fois, puis analyse images et videos.

    Le chargement est couteux (plusieurs secondes) : dans Streamlit, cette
    classe doit etre construite une seule fois et mise en cache.
    """

    def __init__(self, poids: str | Path | None = None,
                 seuils: dict[str, float] | None = None):
        from ultralytics import YOLO          # import tardif : lourd

        self.poids = Path(poids) if poids else RACINE / "models" / "yolov8m.pt"
        if not self.poids.is_file():
            raise FileNotFoundError(f"poids introuvables : {self.poids}")

        self.modele = YOLO(str(self.poids))
        self.noms = self.modele.names
        # Les seuils suivent le modele REELLEMENT charge. Sans cette ligne,
        # selectionner yolov8n appliquait silencieusement les seuils
        # calibres pour yolov8m : l'application affirmait des valeurs
        # mesurees pour un modele jamais calibre.
        self.nom = self.poids.stem
        self.seuils_calibres = calibrage_disponible(self.nom)
        self.seuils = (seuils if seuils is not None
                       else charger_seuils(self.nom))

        # Le filtre brut doit rester SOUS le plus bas des seuils calibres,
        # faute de quoi il decide a leur place. La moitie du minimum laisse
        # une marge confortable sans noyer la suite sous le bruit.
        self.confiance_brute = min(CONFIANCE_BRUTE,
                                   min(self.seuils.values()) / 2)

    # ------------------------------------------------------------ image --

    def analyser_image(self, image) -> ResultatImage:
        """
        Analyse une image isolee.

        Pas d'historique, donc pas d'hysteresis : chaque equipement est
        compare au seuil calibre de sa classe, et le verdict est immediat.
        """
        sortie = self.modele.predict(image, conf=self.confiance_brute, verbose=False)[0]
        personnes, equipements = self._separer(sortie)

        association = associer(personnes, equipements)
        vues = []
        for p in personnes:
            trouves = association.get(p["identifiant"], {})
            verdicts = {classe: confiance >= self.seuils.get(classe, 0.5)
                        for classe, confiance in trouves.items()}
            vues.append(PersonneVue(identifiant=p["identifiant"],
                                    boite=p["boite"],
                                    confiance=p["confiance"],
                                    equipements=trouves,
                                    verdicts=verdicts,
                                    statut=self._statut(p["boite"],
                                                        sortie.orig_shape[0])))

        rattaches = sum(1 for c in association.values()
                        for v in c.values() if v > 0)
        hauteur, largeur = sortie.orig_shape
        return ResultatImage(personnes=vues,
                             equipements_orphelins=max(0, len(equipements) - rattaches),
                             largeur=largeur, hauteur=hauteur)

    # ------------------------------------------------------------ video --

    def analyser_video(self, chemin: str | Path, cadence: int = 1,
                       reglages: Reglages | None = None):
        """
        Analyse une video, image par image, avec memoire.

        Genere un tuple `(numero, ResultatImage, alertes)` par image
        analysee. `cadence=3` n'analyse qu'une image sur trois — c'est le
        reglage utilise pour les mesures du J11.

        C'est ici, et seulement ici, qu'interviennent l'hysteresis, la
        confirmation temporelle et l'agregation par episode.
        """
        moteur = MoteurDecision(reglages or Reglages())
        numero = 0

        flux = self.modele.track(
            source=str(chemin), stream=True, persist=True,
            conf=CONFIANCE_SUIVI, verbose=False,
            tracker=str(RACINE / "src" / "app" / "suivi_epi.yaml"),
        )

        for sortie in flux:
            numero += 1
            if cadence > 1 and numero % cadence:
                continue

            personnes, equipements = self._separer(sortie, avec_suivi=True)
            association = associer(personnes, equipements)

            hauteur, largeur = sortie.orig_shape
            vues = [PersonneVue(identifiant=p["identifiant"], boite=p["boite"],
                                confiance=p["confiance"],
                                equipements=association.get(p["identifiant"], {}),
                                statut=self._statut(p["boite"], hauteur))
                    for p in personnes]

            # Seules les personnes effectivement surveillees alimentent le
            # moteur. Une personne hors perimetre ou dont la tete sort du
            # cadre ne doit pas accumuler d'images « sans casque » : le
            # systeme ne l'a pas regardee.
            observations = [
                {"identifiant": v.identifiant,
                 "confiance_epi": v.equipements.get("helmet", 0.0)}
                for v in vues if v.alertable
            ]
            alertes = moteur.traiter(numero, observations)
            yield numero, ResultatImage(vues, 0, largeur, hauteur), alertes

        self.derniere_statistique = moteur.statistiques()

    # ---------------------------------------------------------- interne --

    def _statut(self, boite: np.ndarray, hauteur_image: int) -> Statut:
        """
        Ce que le systeme peut dire de cette personne, avant tout verdict.

        L'ordre compte : une personne minuscule collee au bord haut est
        d'abord hors perimetre. Le message « rehaussez la camera » ne
        s'adresse qu'aux personnes assez grandes pour etre jugees.
        """
        if hors_perimetre(boite, hauteur_image):
            return Statut.HORS_PERIMETRE
        if tete_hors_champ(boite):
            return Statut.TETE_HORS_CHAMP
        return Statut.SURVEILLEE

    def _separer(self, sortie, avec_suivi: bool = False):
        """
        Coupe les detections en deux listes : les personnes, et le reste.

        Une detection sans boite exploitable est ignoree plutot que de
        propager une valeur douteuse dans la suite du calcul.
        """
        personnes, equipements = [], []
        boites = sortie.boxes
        if boites is None or len(boites) == 0:
            return personnes, equipements

        coords = boites.xyxy.cpu().numpy()
        confiances = boites.conf.cpu().numpy()
        classes = boites.cls.cpu().numpy().astype(int)

        if avec_suivi and boites.id is not None:
            identifiants = boites.id.cpu().numpy().astype(int)
        else:
            identifiants = np.arange(len(classes))

        connues = EQUIPEMENTS_TETE | EQUIPEMENTS_CORPS
        for boite, confiance, classe, identifiant in zip(
                coords, confiances, classes, identifiants):
            nom = self.noms.get(int(classe), str(classe))
            if nom == "person":
                personnes.append({"identifiant": int(identifiant),
                                  "boite": boite, "confiance": float(confiance)})
            elif nom in connues:
                equipements.append({"classe": nom, "boite": boite,
                                    "confiance": float(confiance)})
        return personnes, equipements


def resume(resultat: ResultatImage) -> dict:
    """
    Chiffres prets a afficher, sans logique d'interface.

    Le taux se calcule sur les seules personnes SURVEILLEES. Compter les
    autres gonflerait artificiellement le denominateur et produirait un
    taux de conformite rassurant sur des gens que le systeme n'a pas
    regardes.
    """
    surveillees = resultat.surveillees
    total = len(surveillees)
    sans = resultat.nombre_sans_casque
    return {
        "personnes": len(resultat.personnes),
        "surveillees": total,
        "casque_detecte": total - sans,
        "casque_non_detecte": sans,
        "hors_perimetre": resultat.nombre_hors_perimetre,
        "tete_hors_champ": resultat.nombre_tete_hors_champ,
        "taux": round((total - sans) / total, 3) if total else None,
        "equipements_non_rattaches": resultat.equipements_orphelins,
    }
