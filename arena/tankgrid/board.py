"""Mapa da arena: 48 × 30 casas, espelhado entre a esquerda e a direita.

Só a metade esquerda é descrita; a direita é o espelho (x → W-1-x). Assim
nenhum lado tem vantagem e uma política treinada de um lado funciona do
outro (as observações do vermelho também são espelhadas, ver
`observations.py`).
"""
from __future__ import annotations

from collections import deque

import numpy as np

WIDTH = 48
HEIGHT = 30
BASE_WIDTH = 3          # colunas 0..2 (azul) e 45..47 (vermelho)

# Muros da metade esquerda: (x0, y0, x1, y1) inclusivos.
LEFT_WALLS = (
    (13, 1, 15, 1),
    (16, 2, 16, 5),
    (8, 5, 8, 11),
    (13, 6, 13, 8),
    (12, 10, 18, 10),
    (12, 12, 12, 13),
    (15, 20, 15, 22),
    (19, 23, 19, 25),
)

# Torres 2 × 2 sobre a linha central: canto superior esquerdo de cada uma.
TOWER_CORNERS = ((23, 5), (23, 14), (23, 23))

# Ninho de pentágonos (metade esquerda; a direita é espelhada).
NEST = (20, 16, 22, 19)
# Faixa onde nascem formas comuns e invasores (metade esquerda).
FARM = (4, 0, 22, HEIGHT - 1)
INVADER_ZONE = (16, 0, 22, HEIGHT - 1)


def mirror_x(x: int) -> int:
    return WIDTH - 1 - x


def build_walls() -> np.ndarray:
    walls = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x0, y0, x1, y1 in LEFT_WALLS:
        walls[y0:y1 + 1, x0:x1 + 1] = True
        walls[y0:y1 + 1, mirror_x(x1):mirror_x(x0) + 1] = True
    return walls


def tower_cells(corner) -> tuple[tuple[int, int], ...]:
    x, y = corner
    return ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))


def base_mask(team: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    if team == 0:
        mask[:, :BASE_WIDTH] = True
    else:
        mask[:, WIDTH - BASE_WIDTH:] = True
    return mask


def spawn_cells(team: int) -> list[tuple[int, int]]:
    xs = range(BASE_WIDTH) if team == 0 else range(WIDTH - BASE_WIDTH, WIDTH)
    return [(x, y) for x in xs for y in range(1, HEIGHT - 1)]


def distance_map(solid: np.ndarray, targets) -> np.ndarray:
    """BFS de 8 vizinhos a partir das casas `targets` (ignora formas)."""
    dist = np.full(solid.shape, 10_000, dtype=np.int32)
    queue: deque = deque()
    for x, y in targets:
        dist[y, x] = 0
        queue.append((x, y))
    h, w = solid.shape
    while queue:
        x, y = queue.popleft()
        d = dist[y, x] + 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not solid[ny, nx] \
                        and dist[ny, nx] > d:
                    dist[ny, nx] = d
                    queue.append((nx, ny))
    return dist
