"""
Moteur de decision : transformer des detections en verdicts, puis en alertes.

Ce que ce module corrige
------------------------
Le detecteur produit, pour chaque image et chaque objet, une boite et une
confiance. Il ne dit RIEN sur la conformite : c'est ce module qui decide.

Deux defauts de la premiere version du systeme sont traites ici.

DEFAUT 1 — LE VERDICT CLIGNOTAIT
    Un seuil unique transforme une confiance continue en decision binaire.
    Si la confiance oscille autour du seuil, le verdict oscille avec elle :

        image 1   0,47  >  0,465   ->  CONFORME
        image 2   0,46  <  0,465   ->  NON CONFORME
        image 3   0,48  >  0,465   ->  CONFORME

    Rien n'a bouge sur le chantier. A 25 images par seconde, cela produit
    une douzaine de basculements par seconde.

DEFAUT 2 — LE VOLUME D'ALERTES EST INTENABLE
    Mesure du J10 : le scenario de deploiement produirait environ 8 400
    fausses alertes par heure, la norme EEMUA 191 en autorisant SIX.
    Il faut diviser par ~1 400.

Les trois mecanismes, et ce que chacun apporte
-----------------------------------------------
1. HYSTERESIS — deux seuils au lieu d'un

       au-dessus de HAUT  ->  l'equipement est considere present
       en dessous de BAS  ->  il est considere absent
       entre les deux     ->  ON GARDE LE VERDICT PRECEDENT

   Un thermostat qui chauffe jusqu'a 20 degres et ne redemarre qu'a 18 :
   entre les deux, il ne fait rien. Sans cela il claquerait sans cesse
   autour de 19.

   L'ecart entre HAUT et BAS n'est pas choisi : il est DEDUIT du bruit
   reellement mesure sur la video (voir `mesure_clignotement.py`).

2. CONFIRMATION TEMPORELLE — n'alerter qu'apres N images consecutives

   Une fausse detection isolee ne franchit pas la barre. C'est le levier
   le plus efficace contre les fausses alertes.

   ATTENTION a l'argument facile : si les erreurs etaient INDEPENDANTES
   d'une image a l'autre, exiger N images consecutives diviserait le taux
   par p^N. Elles ne le sont pas — deux images successives se ressemblent,
   et une erreur tend a persister. Le gain reel est plus faible, et il doit
   etre MESURE, pas calcule.

3. AGREGATION PAR EPISODE — une alerte par personne et par episode

   L'ouvrier n 3 sans casque pendant cinq minutes doit produire UNE alerte,
   pas 7 500. L'alerte se re-arme seulement apres un retour durable a la
   conformite.

Ce que ce module ne fait pas
-----------------------------
Il ne juge pas la CONFORMITE REGLEMENTAIRE. Il constate la presence ou
l'absence d'un objet ressemblant a un equipement de protection. La classe
`helmet` de SH17 ne distingue pas un casque de chantier d'un casque de velo
(D-032) : le vocabulaire employe est donc « casque non detecte », jamais
« ouvrier non conforme ».
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """L'etat d'une personne vis-a-vis d'un equipement."""
    EQUIPE = "equipe"              # l'equipement est detecte
    NON_EQUIPE = "non_equipe"      # il ne l'est pas
    INDETERMINE = "indetermine"    # pas encore assez d'information


@dataclass
class Reglages:
    """
    Les parametres de decision, tous justifies et aucun choisi au hasard.

    `seuil_haut` et `seuil_bas` encadrent le seuil optimal mesure au J10.
    Leur ECART vient du bruit observe sur la video : il doit valoir environ
    quatre ecarts-types de la confiance d'une image a l'autre, de sorte
    qu'une fluctuation ordinaire ne puisse pas traverser la zone morte.

    LES VALEURS CI-DESSOUS SONT MESUREES, PAS CHOISIES — 23 aout.
        seuil optimal J10 (F1 maximal, classe helmet) ...... 0,4653
        ecart-type du fremissement, mesure sur 72 s de video  0,0490
        zone morte = 4 x 0,0490 ............................ 0,196
        -> BAS  = 0,4653 - 0,098 = 0,367
        -> HAUT = 0,4653 + 0,098 = 0,563

    Elles etaient auparavant fixees a 0,35 / 0,55, valeurs provisoires
    posees AVANT la mesure. `tests/test_decision.py` verrouille desormais
    l'accord entre ces defauts et `resultats/video/clignotement.json` :
    si l'un bouge sans l'autre, le test echoue.
    """
    seuil_haut: float = 0.563
    seuil_bas: float = 0.367

    # FENETRE DE CONFIRMATION — fixee le 23 aout, apres mesure.
    #
    # Nombre d'images consecutives sans equipement avant de declencher.
    # La valeur par defaut correspond a DEUX SECONDES, a convertir selon la
    # cadence reelle : a 20 images/seconde analysees, cela fait 40.
    #
    # Le balayage mesure sur video donne :
    #
    #     0,3 s -> 94 % de fausses alertes      2,0 s -> 67 %
    #     0,5 s -> 91 %                         3,0 s -> 60 %
    #     1,0 s -> 82 %                         5,0 s -> 33 %
    #
    # La tendance est nette : un trou de detection d'une demi-seconde est
    # banal, un trou de plusieurs secondes est rare.
    #
    # POURQUOI PAS 5 SECONDES, QUI DONNE LE MEILLEUR CHIFFRE
    # La video de reference ne contient que DEUX non-conformites reelles.
    # Choisir 5 s parce qu'il y produit 33 % reviendrait a ajuster un
    # parametre sur deux exemples : ce n'est pas un calibrage, c'est une
    # coincidence. Deux secondes se justifie independamment des donnees —
    # un ouvrier sans casque n'est pas un evenement fugace.
    #
    # LE PRIX, A ASSUMER
    # Un ouvrier qui traverse le champ en moins de deux secondes ne sera
    # jamais signale.
    images_avant_alerte: int = 40

    # Nombre d'images equipe consecutives avant de re-armer l'alerte pour
    # cette personne. Plus long que le declenchement : on prefere manquer
    # une seconde alerte que de re-alerter sur un retour momentane.
    images_avant_rearmement: int = 80

    # Au-dela, une personne non revue est oubliee : son identifiant peut
    # etre reattribue par le suivi.
    images_avant_oubli: int = 50

    # Longueur de l'historique conserve par personne, pour le lissage.
    memoire: int = 30

    def __post_init__(self):
        if not 0 <= self.seuil_bas < self.seuil_haut <= 1:
            raise ValueError(
                f"seuils incoherents : bas={self.seuil_bas}, haut={self.seuil_haut}")


@dataclass
class EtatPersonne:
    """Ce que le systeme retient d'une personne suivie."""
    identifiant: int
    verdict: Verdict = Verdict.INDETERMINE
    confiances: deque = field(default_factory=lambda: deque(maxlen=30))
    images_non_equipe: int = 0
    images_equipe: int = 0
    alerte_active: bool = False
    derniere_vue: int = 0
    basculements: int = 0          # pour mesurer le clignotement
    alertes_emises: int = 0


class MoteurDecision:
    """
    Applique hysteresis, confirmation temporelle et agregation par episode.

    Usage image par image :

        moteur = MoteurDecision(Reglages())
        for numero, detections in enumerate(video):
            alertes = moteur.traiter(numero, detections)

    `detections` est une liste de dictionnaires :
        {"identifiant": 3, "confiance_epi": 0.42}

    La confiance vaut 0 lorsque aucun equipement n'est associe a la
    personne — une absence de detection est une confiance nulle, pas une
    donnee manquante.
    """

    def __init__(self, reglages: Reglages | None = None):
        self.r = reglages or Reglages()
        self.personnes: dict[int, EtatPersonne] = {}
        self.journal: list[dict] = []

    # ---------------------------------------------------------------- API --

    def traiter(self, numero_image: int, detections: list[dict]) -> list[dict]:
        """Traite une image, renvoie les alertes DECLENCHEES par celle-ci."""
        alertes = []
        for d in detections:
            alerte = self._traiter_personne(numero_image, d["identifiant"],
                                            float(d.get("confiance_epi", 0.0)))
            if alerte:
                alertes.append(alerte)
        self._oublier(numero_image)
        return alertes

    def statistiques(self) -> dict:
        total_basculements = sum(p.basculements for p in self.personnes.values())
        return {
            "personnes_suivies": len(self.personnes),
            "basculements_de_verdict": total_basculements,
            "alertes_emises": len(self.journal),
            "alertes_par_personne": (len(self.journal) / len(self.personnes)
                                     if self.personnes else 0.0),
        }

    # ------------------------------------------------------------ interne --

    def _traiter_personne(self, numero: int, identifiant: int,
                          confiance: float) -> dict | None:
        etat = self.personnes.get(identifiant)
        if etat is None:
            etat = EtatPersonne(identifiant=identifiant)
            etat.confiances = deque(maxlen=self.r.memoire)
            self.personnes[identifiant] = etat

        etat.derniere_vue = numero
        etat.confiances.append(confiance)

        ancien = etat.verdict
        nouveau = self._verdict_avec_hysteresis(confiance, ancien)
        if nouveau != ancien and ancien != Verdict.INDETERMINE:
            etat.basculements += 1
        etat.verdict = nouveau

        # --- confirmation temporelle -------------------------------------
        if nouveau == Verdict.NON_EQUIPE:
            etat.images_non_equipe += 1
            etat.images_equipe = 0
        elif nouveau == Verdict.EQUIPE:
            etat.images_equipe += 1
            etat.images_non_equipe = 0
        else:
            etat.images_non_equipe = etat.images_equipe = 0

        # --- declenchement, une seule fois par episode --------------------
        if (not etat.alerte_active
                and etat.images_non_equipe >= self.r.images_avant_alerte):
            etat.alerte_active = True
            etat.alertes_emises += 1
            alerte = {
                "image": numero, "identifiant": identifiant,
                "images_consecutives": etat.images_non_equipe,
                "confiance_moyenne": round(
                    sum(etat.confiances) / len(etat.confiances), 4),
                # Vocabulaire prudent : le systeme constate une absence de
                # detection, il ne prononce pas une non-conformite (D-032).
                "message": f"casque non detecte — personne {identifiant}",
            }
            self.journal.append(alerte)
            return alerte

        # --- re-armement, plus exigeant que le declenchement ---------------
        if (etat.alerte_active
                and etat.images_equipe >= self.r.images_avant_rearmement):
            etat.alerte_active = False

        return None

    def _verdict_avec_hysteresis(self, confiance: float,
                                 precedent: Verdict) -> Verdict:
        """
        Le coeur du mecanisme : entre les deux seuils, on ne change rien.

        C'est cette zone morte qui rend un basculement rapide impossible.
        """
        if confiance >= self.r.seuil_haut:
            return Verdict.EQUIPE
        if confiance <= self.r.seuil_bas:
            return Verdict.NON_EQUIPE
        # Zone morte. Au tout debut, aucun verdict anterieur n'existe :
        # on reste indetermine plutot que de trancher au hasard.
        return precedent if precedent != Verdict.INDETERMINE else Verdict.INDETERMINE

    def _oublier(self, numero: int) -> None:
        perimes = [i for i, p in self.personnes.items()
                   if numero - p.derniere_vue > self.r.images_avant_oubli]
        for i in perimes:
            del self.personnes[i]


class MoteurSansMemoire:
    """
    Le systeme AVANT correction : un seuil unique, aucune memoire, une
    alerte par image.

    Conserve pour la comparaison. C'est lui qui produit le clignotement et
    le volume d'alertes intenable ; sans point de reference, le gain du
    moteur corrige ne serait qu'une affirmation.
    """

    def __init__(self, seuil: float = 0.465):
        self.seuil = seuil
        self.verdicts: dict[int, Verdict] = {}
        self.basculements = 0
        self.alertes = 0
        self.personnes: set[int] = set()

    def traiter(self, numero_image: int, detections: list[dict]) -> list[dict]:
        alertes = []
        for d in detections:
            identifiant = d["identifiant"]
            self.personnes.add(identifiant)
            confiance = float(d.get("confiance_epi", 0.0))
            nouveau = (Verdict.EQUIPE if confiance >= self.seuil
                       else Verdict.NON_EQUIPE)
            ancien = self.verdicts.get(identifiant)
            if ancien is not None and nouveau != ancien:
                self.basculements += 1
            self.verdicts[identifiant] = nouveau
            if nouveau == Verdict.NON_EQUIPE:
                self.alertes += 1                      # une alerte par image
                alertes.append({"image": numero_image, "identifiant": identifiant})
        return alertes

    def statistiques(self) -> dict:
        return {"personnes_suivies": len(self.personnes),
                "basculements_de_verdict": self.basculements,
                "alertes_emises": self.alertes,
                "alertes_par_personne": (self.alertes / len(self.personnes)
                                         if self.personnes else 0.0)}
