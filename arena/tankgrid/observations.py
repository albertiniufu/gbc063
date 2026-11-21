"""Observação local de um tanque e codificação das ações.

A observação é uma imagem `(C, V, V)` centrada no tanque (V = `view_size`,
15 por padrão), no **referencial canônico do time**: para o vermelho o mapa
é espelhado na horizontal, então todo tanque "nasce na esquerda". Assim uma
mesma política serve aos dois lados (útil para self-play e torneio).

Canais espaciais (0..13), recortados do mapa:

| canal | conteúdo |
| --- | --- |
| 0 | muro ou fora do mapa |
| 1 | formas: quadrado .33, triângulo .66, pentágono 1 |
| 2 | invasores |
| 3, 4 | tanques aliados / inimigos (vida relativa) |
| 5, 6 | classe dos aliados / inimigos ((índice+1)/14) |
| 7, 8, 9 | torres nossas / deles / neutras (vida relativa) |
| 10, 11 | tiros aliados / inimigos |
| 12, 13 | nossa base / base deles |
| 14, 15 | velocidade das formas (vx, vy ÷ velocidade máxima; vx espelhado) |

Canais escalares (16..), constantes no plano: vida, nível, recarga pronta,
pontos pendentes, evolução disponível, classe, tempo, placar, posição (x, y),
dono e direção (dx, dy) de cada torre, sequência de domínio e os 5
atributos. Total: `N_CHANNELS`.

Ação discreta do ambiente (referencial canônico):

- `a < 81`: `move = a // 9`, `shoot = a % 9` (sem evolução);
- `a ≥ 81` (só com `auto_upgrade=False`): evolução `a - 80` (1..10), sem
  mover nem atirar neste passo.
"""
from __future__ import annotations

import numpy as np

from .classes import MAX_ATTR_POINTS, MAX_LEVEL, MIRROR_DIR, N_CLASSES
from .game import (
    INVADER,
    PENTAGON,
    SHAPE_MAX_SPEED,
    SQUARE,
    TRIANGLE,
    TankGridGame,
)

N_SPATIAL = 16
N_TOWERS = 3
N_SCALAR = 10 + 3 * N_TOWERS + 1 + 5
N_CHANNELS = N_SPATIAL + N_SCALAR

N_MOVE_SHOOT = 81
N_UPGRADES = 10
SHAPE_VALUE = {SQUARE: 1 / 3, TRIANGLE: 2 / 3, PENTAGON: 1.0}


def n_actions(auto_upgrade: bool) -> int:
    return N_MOVE_SHOOT if auto_upgrade else N_MOVE_SHOOT + N_UPGRADES


def to_world_dir(d: int, team: int) -> int:
    return MIRROR_DIR[d] if team == 1 else d


def decode_action(action: int, team: int) -> tuple[int, int, int]:
    """Ação canônica do ambiente → `(move, shoot, upgrade)` do mundo."""
    a = int(action)
    if a >= N_MOVE_SHOOT:
        return 0, 0, a - N_MOVE_SHOOT + 1
    move, shoot = divmod(a, 9)
    return to_world_dir(move, team), to_world_dir(shoot, team), 0


def encode_action(move: int, shoot: int, team: int) -> int:
    """Inverso de `decode_action` para ações sem evolução."""
    return to_world_dir(move, team) * 9 + to_world_dir(shoot, team)


class ObservationBuilder:
    """Constrói observações; o mapa global é recalculado uma vez por passo."""

    def __init__(self, game: TankGridGame, view_size: int = 15):
        if view_size % 2 == 0:
            raise ValueError("view_size deve ser ímpar")
        self.game = game
        self.view = int(view_size)
        self.radius = self.view // 2
        self._cache: dict[int, np.ndarray] = {}
        self._cache_version = -1

    @property
    def shape(self) -> tuple[int, int, int]:
        return (N_CHANNELS, self.view, self.view)

    # ------------------------------------------------------------------ #
    def _global_layers(self, team: int) -> np.ndarray:
        """Mapa `(N_SPATIAL, H+2r, W+2r)` do ponto de vista de `team`."""
        game = self.game
        if self._cache_version != game.version:
            self._cache.clear()
            self._cache_version = game.version
        if team in self._cache:
            return self._cache[team]
        h, w, r = game.H, game.W, self.radius
        layers = np.zeros((N_SPATIAL, h + 2 * r, w + 2 * r), np.float32)
        layers[0] = 1.0
        inner = (slice(r, r + h), slice(r, r + w))
        layers[0][inner] = game.walls
        for shape in game.shapes:
            if not shape.alive:
                continue
            y, x = shape.y + r, shape.x + r
            if shape.kind == INVADER:
                layers[2, y, x] = 1.0
            else:
                layers[1, y, x] = SHAPE_VALUE[shape.kind]
            layers[14, y, x] = shape.vx / SHAPE_MAX_SPEED
            layers[15, y, x] = shape.vy / SHAPE_MAX_SPEED
        for tank in game.tanks:
            if not tank.alive:
                continue
            y, x = tank.y + r, tank.x + r
            enemy = tank.team != team
            layers[3 + enemy, y, x] = max(0.0, tank.hp) / tank.max_hp
            layers[5 + enemy, y, x] = (tank.cls + 1) / N_CLASSES
        for tower in game.towers:
            if tower.owner < 0:
                ch = 9
            else:
                ch = 7 if tower.owner == team else 8
            for x, y in tower.cells:
                layers[ch, y + r, x + r] = tower.hp / 15.0
        for b in game.bullets:
            layers[10 + (b.team != team), b.y + r, b.x + r] = 1.0
        layers[12][inner] = game.base[team]
        layers[13][inner] = game.base[1 - team]
        if team == 1:
            layers = np.ascontiguousarray(layers[:, :, ::-1])
            layers[14] *= -1.0          # vx também é espelhado
        self._cache[team] = layers
        return layers

    def observe(self, tank_id: int) -> np.ndarray:
        game = self.game
        tank = game.tanks[tank_id]
        team = tank.team
        layers = self._global_layers(team)
        x = tank.x if team == 0 else game.W - 1 - tank.x
        y = tank.y
        v = self.view
        obs = np.zeros((N_CHANNELS, v, v), np.float32)
        obs[:N_SPATIAL] = layers[:, y:y + v, x:x + v]

        scalars = [
            max(0.0, tank.hp) / tank.max_hp if tank.alive else 0.0,
            tank.level / MAX_LEVEL,
            1.0 if tank.reload == 0 else 0.0,
            min(1.0, tank.pending_points / 5.0),
            1.0 if tank.class_options else 0.0,
            (tank.cls + 1) / N_CLASSES,
            game.step_count / max(1, game.config.max_steps),
            float(np.tanh((game.score[team] - game.score[1 - team]) / 200)),
            x / (game.W - 1),
            y / (game.H - 1),
        ]
        for tower in game.towers:
            if tower.owner < 0:
                scalars.append(0.0)
            else:
                scalars.append(1.0 if tower.owner == team else -1.0)
        for tower in game.towers:
            cx, cy = tower.center
            if team == 1:
                cx = game.W - 1 - cx
            scalars.append((cx - x) / game.W)
            scalars.append((cy - y) / game.H)
        hold = game.hold[team] - game.hold[1 - team]
        scalars.append(hold / max(1, game.config.hold_to_win))
        scalars.extend(p / MAX_ATTR_POINTS for p in tank.points)
        values = np.asarray(scalars, dtype=np.float32)
        obs[N_SPATIAL:] = values[:, None, None]
        return obs
