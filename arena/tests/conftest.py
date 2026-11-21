"""Configuração dos testes da arena: `pytest` a partir de `arena/`.

Os testes rodam sem tela e sem torch (só o motor e o ambiente).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
