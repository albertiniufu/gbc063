"""Renderer pygame do TankGrid no visual dos jogos .io de tanque.

Papel quadriculado cinza, tanques azul (#00b2e1) e vermelho (#f14e54),
formas amarelas/vermelhas/roxas, torres hexagonais e placar no topo.
`frame(game)` desenha numa superfície fora da tela e devolve um array
`(H, W, 3)`; não precisa de janela. Quem roda sem tela define
`SDL_VIDEODRIVER=dummy` antes de importar este módulo; o modo jogável
(`play.py`) abre uma janela de verdade.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from .classes import DIRS, perpendicular, rotate_dir
from .game import INVADER, PENTAGON, SQUARE, TRIANGLE, TOWER_HP

CELL = 20
HEADER = 58
BG = (205, 205, 205)
GRID = (196, 196, 196)
ARENA = (215, 215, 215)
WALL = (95, 97, 104)
OUTLINE = (85, 85, 85)
BARREL = (153, 153, 153)
TEAM = ((0, 178, 225), (241, 78, 84))
TEAM_DARK = ((0, 126, 161), (196, 54, 60))
BASE_TINT = ((150, 200, 220), (225, 170, 172))
NEUTRAL = (153, 153, 153)
HEADER_BG = (32, 36, 44)
TEXT = (238, 241, 246)
MUTED = (170, 178, 192)
SHAPE_STYLE = {
    SQUARE: ((255, 232, 105), (191, 174, 78), 4, math.pi / 4),
    TRIANGLE: ((252, 118, 119), (189, 88, 89), 3, -math.pi / 2),
    PENTAGON: ((118, 141, 252), (88, 105, 189), 5, -math.pi / 2),
    INVADER: ((241, 119, 221), (180, 89, 165), 3, 0.3),
}


def _poly(cx, cy, r, n, rot):
    return [(cx + r * math.cos(rot + i * 2 * math.pi / n),
             cy + r * math.sin(rot + i * 2 * math.pi / n)) for i in range(n)]


def _angle(d: int) -> float:
    """Ângulo de tela (y para baixo) da direção `d`."""
    dx, dy = DIRS[d]
    return math.atan2(dy, dx)


class TankGridRenderer:
    def __init__(self, cell: int = CELL):
        pygame.init()
        pygame.font.init()
        self.cell = cell
        self.width = 48 * cell
        self.height = 30 * cell + HEADER
        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont("dejavusans,ubuntu,arial", 13,
                                        bold=True)
        self.big = pygame.font.SysFont("dejavusans,ubuntu,arial", 24,
                                       bold=True)
        self._background = None

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    def _cell_center(self, x, y):
        c = self.cell
        return x * c + c / 2, HEADER + y * c + c / 2

    def _draw_background(self, game):
        surf = pygame.Surface((self.width, self.height))
        surf.fill(ARENA)
        c = self.cell
        for team in (0, 1):
            ys, xs = np.nonzero(game.base[team])
            x0, x1 = xs.min(), xs.max() + 1
            pygame.draw.rect(surf, BASE_TINT[team],
                             (x0 * c, HEADER, (x1 - x0) * c, 30 * c))
        for x in range(49):
            pygame.draw.line(surf, GRID, (x * c, HEADER),
                             (x * c, self.height))
        for y in range(31):
            pygame.draw.line(surf, GRID, (0, HEADER + y * c),
                             (self.width, HEADER + y * c))
        ys, xs = np.nonzero(game.walls)
        for x, y in zip(xs, ys):
            pygame.draw.rect(surf, WALL, (x * c, HEADER + y * c, c, c))
        return surf

    def _tank(self, surf, tank, cx, cy, r):
        klass = tank.klass
        color = TEAM[tank.team]
        if not klass.barrels:   # Esmagador: casco hexagonal
            pygame.draw.polygon(surf, OUTLINE, _poly(cx, cy, r * 1.45, 6,
                                                     0))
        for rot, lateral, mult in klass.barrels:
            d = rotate_dir(tank.facing or 1, rot)
            ang = _angle(d)
            length = r * 1.6 * (1.4 if klass.range_bonus >= 12 else
                                1.2 if klass.range_bonus else 1.0)
            width = r * (0.9 if mult >= 3 else 0.62 if lateral == 0
                         else 0.5)
            ox, oy = 0.0, 0.0
            if lateral:
                px, py = perpendicular(d)
                norm = math.hypot(px, py)
                ox, oy = lateral * px / norm * r * 0.45, \
                    lateral * py / norm * r * 0.45
            ca, sa = math.cos(ang), math.sin(ang)
            pts = []
            for along, side in ((0, -1), (length, -1), (length, 1), (0, 1)):
                sw = side * width / 2
                pts.append((cx + ox + ca * along - sa * sw,
                            cy + oy + sa * along + ca * sw))
            pygame.draw.polygon(surf, BARREL, pts)
            pygame.draw.polygon(surf, OUTLINE, pts, 2)
        pygame.draw.circle(surf, color, (cx, cy), r)
        pygame.draw.circle(surf, OUTLINE, (cx, cy), r, 2)

    def _hp_bar(self, surf, cx, cy, frac, width, color=(133, 227, 125)):
        if frac >= 0.999:
            return
        x = cx - width / 2
        pygame.draw.rect(surf, (85, 85, 85), (x, cy, width, 4),
                         border_radius=2)
        pygame.draw.rect(surf, color, (x, cy, max(1, width * frac), 4),
                         border_radius=2)

    # ------------------------------------------------------------------ #
    def frame(self, game) -> np.ndarray:
        self.draw(game)
        array = pygame.surfarray.array3d(self.surface)
        return np.ascontiguousarray(array.transpose(1, 0, 2))

    def draw(self, game):
        if self._background is None:
            self._background = self._draw_background(game)
        surf = self.surface
        surf.blit(self._background, (0, 0))
        c = self.cell

        for shape in game.shapes:
            if not shape.alive:
                continue
            fill, edge, n, rot = SHAPE_STYLE[shape.kind]
            # Posição contínua e giro: as formas flutuam entre as casas.
            cx, cy = self._cell_center(shape.fx, shape.fy)
            rot += shape.angle
            size = c * (0.55 if shape.kind == PENTAGON else 0.42)
            pts = _poly(cx, cy, size, n, rot)
            pygame.draw.polygon(surf, fill, pts)
            pygame.draw.polygon(surf, edge, pts, 2)

        for tower in game.towers:
            cx, cy = tower.corner[0] * c + c, HEADER + tower.corner[1] * c + c
            pygame.draw.polygon(surf, OUTLINE, _poly(cx, cy, c * 1.05, 6, 0))
            color = NEUTRAL if tower.owner < 0 else TEAM[tower.owner]
            pygame.draw.circle(surf, color, (cx, cy), c * 0.7)
            pygame.draw.circle(surf, OUTLINE, (cx, cy), c * 0.7, 2)
            self._hp_bar(surf, cx, cy + c * 1.15, tower.hp / TOWER_HP, c * 1.6,
                         color if tower.owner >= 0 else (90, 90, 90))

        for tank in game.tanks:
            if not tank.alive:
                continue
            cx, cy = self._cell_center(tank.x, tank.y)
            self._tank(surf, tank, cx, cy, c * 0.42)
            self._hp_bar(surf, cx, cy + c * 0.6, max(0, tank.hp) / tank.max_hp,
                         c * 0.9)

        for b in game.bullets:
            cx, cy = self._cell_center(b.x, b.y)
            r = c * (0.28 if b.damage >= 3 else 0.2)
            pygame.draw.circle(surf, TEAM[b.team], (cx, cy), r)
            pygame.draw.circle(surf, OUTLINE, (cx, cy), r, 1)

        self._header(game)

    def _header(self, game):
        surf = self.surface
        pygame.draw.rect(surf, HEADER_BG, (0, 0, self.width, HEADER))
        for team in (0, 1):
            score = self.big.render(str(game.score[team]), True, TEAM[team])
            x = 14 if team == 0 else self.width - 14 - score.get_width()
            surf.blit(score, (x, 4))
            labels = []
            for tank in game.team_tanks(team):
                state = "" if tank.alive else f" ({tank.respawn})"
                labels.append(f"Nv{tank.level} {tank.klass.label}{state}")
            text = self.font.render("   ".join(labels), True, MUTED)
            x = 14 if team == 0 else self.width - 14 - text.get_width()
            surf.blit(text, (x, 36))
        mid = self.width / 2
        for k, tower in enumerate(game.towers):
            color = NEUTRAL if tower.owner < 0 else TEAM[tower.owner]
            cx = mid + (k - 1) * 26
            pygame.draw.circle(surf, color, (cx, 17), 9)
            pygame.draw.circle(surf, (20, 20, 20), (cx, 17), 9, 2)
        status = f"passo {game.step_count}/{game.config.max_steps}"
        if game.done:
            if game.winner == -1:
                status = "empate"
            else:
                status = ("azul" if game.winner == 0 else "vermelho") + \
                    " vence (" + game.win_reason + ")"
        text = self.font.render(status, True, TEXT)
        surf.blit(text, (mid - text.get_width() / 2, 36))
