# -*- coding: utf-8 -*-
"""
Point d'entree de l'application deployee.

Streamlit Cloud lance le fichier designe dans la configuration du service.
On garde `src/app/interface.py` inchange — c'est le meme code qu'en local,
et c'est la condition pour que ce qui est demontre au jury soit ce qui a
ete teste.
"""
from src.app.interface import main

main()
