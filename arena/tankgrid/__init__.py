"""TankGrid: arena 3 × 3 de tanques num mapa 2D.

Os tanques destroem formas para ganhar XP, sobem de nível, evoluem de classe
(14 classes numa árvore) e disputam três torres no centro do mapa. O motor
puro (`game.py`) serve ao ambiente Gymnasium (`env.py`), aos bots
(`agents.py`), ao renderer, ao modo jogável e ao torneio.
"""
from .config import TankGridConfig
from .game import TankGridGame

__all__ = ["TankGridConfig", "TankGridGame"]
