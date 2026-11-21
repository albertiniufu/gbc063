"""Motor puro do TankGrid: estado, regras e um `step` simultâneo.

O motor não sabe nada de Gymnasium: recebe uma ação por tanque, em
coordenadas do mundo, e avança um passo. O ambiente (`env.py`), os bots
(`agents.py`), o renderer e o torneio usam o mesmo motor.

Ação de um tanque: tupla `(move, shoot, upgrade)`:

- `move`  ∈ 0..8 — 0 parado, 1..8 direções (ver `classes.DIRS`);
- `shoot` ∈ 0..8 — 0 não atira; o Esmagador usa a direção para investir;
- `upgrade` ∈ 0..10 — 0 nada, 1..5 atributo `upgrade-1`, 6..10 a opção de
  classe `upgrade-6` de `class_options`.

Ordem de um passo: evoluções → movimento → tiros (tanques e torres) →
balas (2 casas por passo) → invasores → física das formas → contato do
Esmagador → bases e regeneração → mortes, XP e níveis → renascimento →
novas formas → placar.

Física "de espaço" das formas (`shape_physics="space"`, o padrão): cada
forma tem posição contínua e velocidade, e segue por inércia, sem atrito,
girando devagar. Bate e volta em muros, torres, tanques e na borda das
bases; entre formas a colisão é elástica (conserva momento, com massas
diferentes). Tiros e tanques que encostam empurram as formas (o empurrão
é dividido pela massa). Pentágonos orbitam o centro do próprio ninho, presos
por uma "gravidade" fraca, e invasores aceleram com propulsão em direção à
presa, como naves. A casa de uma forma (`x`, `y`) é a posição contínua
arredondada; regras de tiro e colisão continuam na grade.
`shape_physics="static"` mantém as formas paradas.
"""
from __future__ import annotations

import math

import numpy as np

from . import board
from .classes import (
    CLASS_INDEX,
    CLASSES,
    DEFAULT_BUILD,
    DIRS,
    MAX_ATTR_POINTS,
    N_ATTRIBUTES,
    bullet_damage,
    bullet_range,
    class_options,
    level_from_xp,
    max_hp,
    perpendicular,
    regen_interval,
    reload_interval,
    rotate_dir,
)
from .config import TankGridConfig

# Formas: (vida, XP)
SQUARE, TRIANGLE, PENTAGON, INVADER = 0, 1, 2, 3
SHAPE_NAMES = ("quadrado", "triangulo", "pentagono", "invasor")
SHAPE_HP = (1.0, 3.0, 10.0, 3.0)
SHAPE_XP = (2.0, 5.0, 20.0, 8.0)
INVADER_SIGHT = 6
INVADER_BLAST = 3.0

# Física de espaço (unidades: casas e passos)
SHAPE_MASS = (1.0, 2.0, 5.0, 1.5)
SHAPE_DRIFT = (0.02, 0.10)       # velocidade inicial de deriva (mín, máx)
SHAPE_MAX_SPEED = 0.6            # < 1: no máximo uma casa por eixo/passo
SHAPE_MAX_SPIN = 0.06            # rad/passo (só visual)
RESTITUTION = 1.0                # quique perfeito, como no vácuo
BULLET_KNOCKBACK = 0.08          # impulso por ponto de dano
TANK_PUSH = 0.15                 # impulso de um tanque que encosta
NEST_GRAVITY = 0.01              # aceleração ∝ distância ao centro do ninho
INVADER_THRUST = 0.06
INVADER_MAX_SPEED = 0.45

TOWER_HP = 15.0
TOWER_RANGE = 6
TOWER_RELOAD = 3
TOWER_DAMAGE = 1.0
TOWER_XP = 25.0

BULLET_SPEED = 2
KILL_BONUS_XP = 5.0
BASE_HEAL = 1.0
BASE_DAMAGE = 2.0

EVENT_KEYS = ("score", "xp", "kills", "deaths", "damage", "captures",
              "towers_lost")


def quantize_dir(dx: float, dy: float) -> int:
    """Direção 1..8 mais próxima do vetor (dx, dy) (y para baixo)."""
    if dx == 0 and dy == 0:
        return 0
    angle = math.atan2(-dy, dx)
    k = int(round(angle / (math.pi / 4))) % 8
    return k + 1


def chebyshev(ax, ay, bx, by) -> int:
    return max(abs(ax - bx), abs(ay - by))


class Tank:
    __slots__ = ("id", "team", "slot", "x", "y", "hp", "xp", "level", "cls",
                 "points", "history", "reload", "regen_timer", "alive",
                 "respawn", "facing", "burst_left", "burst_dir", "dash_timer",
                 "kills", "deaths", "damage_dealt", "last_hit_by", "auto",
                 "build_pos")

    def __init__(self, tank_id: int, team: int, slot: int):
        self.id = tank_id
        self.team = team
        self.slot = slot
        self.x = self.y = 0
        self.xp = 0.0
        self.level = 1
        self.cls = CLASS_INDEX["basico"]
        self.points = [0] * N_ATTRIBUTES
        self.history: list[int] = []
        self.reload = 0
        self.regen_timer = 0
        self.alive = True
        self.respawn = 0
        self.facing = 1 if team == 0 else 5
        self.burst_left = 0
        self.burst_dir = 0
        self.dash_timer = 0
        self.kills = 0
        self.deaths = 0
        self.damage_dealt = 0.0
        self.last_hit_by = None
        self.auto = False
        self.build_pos = 0
        self.hp = float(self.max_hp)

    @property
    def klass(self):
        return CLASSES[self.cls]

    @property
    def max_hp(self) -> int:
        return max_hp(self.klass, self.points)

    @property
    def pending_points(self) -> int:
        return (self.level - 1) - sum(self.points)

    @property
    def class_options(self) -> tuple[str, ...]:
        return class_options(self.klass.key, self.level)


class Bullet:
    __slots__ = ("team", "owner", "x", "y", "dir", "damage", "cells_left",
                 "pierce", "alive")

    def __init__(self, team, owner, x, y, d, damage, cells, pierce=False):
        self.team = team
        self.owner = owner      # id do tanque ou -1 (torre)
        self.x, self.y = x, y
        self.dir = d
        self.damage = float(damage)
        self.cells_left = int(cells)
        self.pierce = pierce
        self.alive = True


class Shape:
    """Forma-alvo. `x, y` é a casa; `fx, fy` a posição contínua (centro da
    casa em coordenadas inteiras) e `vx, vy` a velocidade em casas/passo."""

    __slots__ = ("kind", "x", "y", "hp", "alive", "idx", "fx", "fy", "vx",
                 "vy", "angle", "spin", "anchor")

    def __init__(self, kind, x, y, idx=-1, vx=0.0, vy=0.0, spin=0.0,
                 angle=0.0, anchor=None):
        self.kind = kind
        self.idx = idx
        self.x, self.y = x, y
        self.fx, self.fy = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.angle = float(angle)
        self.spin = float(spin)
        self.anchor = anchor        # centro de órbita (pentágonos)
        self.hp = SHAPE_HP[kind]
        self.alive = True

    @property
    def mass(self) -> float:
        return SHAPE_MASS[self.kind]

    def push(self, ix: float, iy: float) -> None:
        """Aplica um impulso (dividido pela massa)."""
        self.vx += ix / self.mass
        self.vy += iy / self.mass


class Tower:
    __slots__ = ("index", "corner", "cells", "owner", "hp", "cooldown")

    def __init__(self, index, corner):
        self.index = index
        self.corner = corner
        self.cells = board.tower_cells(corner)
        self.owner = -1
        self.hp = TOWER_HP
        self.cooldown = TOWER_RELOAD

    @property
    def center(self) -> tuple[float, float]:
        return self.corner[0] + 0.5, self.corner[1] + 0.5

    def distance(self, x, y) -> int:
        return min(chebyshev(cx, cy, x, y) for cx, cy in self.cells)


class TankGridGame:
    """Estado completo de uma partida 3 × 3 (ou `team_size` × `team_size`)."""

    W, H = board.WIDTH, board.HEIGHT

    def __init__(self, config: TankGridConfig | None = None, seed=None):
        self.config = config or TankGridConfig()
        self.walls = board.build_walls()
        self.tower_at = np.full((self.H, self.W), -1, dtype=np.int16)
        for k, corner in enumerate(board.TOWER_CORNERS):
            for x, y in board.tower_cells(corner):
                self.tower_at[y, x] = k
        self.solid = self.walls | (self.tower_at >= 0)
        self.base = (board.base_mask(0), board.base_mask(1))
        # Mapas de distância estáticos (ignoram formas e tanques), para bots.
        self.dist_to_tower = [
            board.distance_map(self.solid, board.tower_cells(c))
            for c in board.TOWER_CORNERS]
        self.dist_to_base = [
            board.distance_map(self.solid, [
                (x, y) for y in range(self.H) for x in range(self.W)
                if self.base[team][y, x]])
            for team in (0, 1)]
        self._free_cells = {}
        self.recorder = None        # gravador de replay (ver replay.py)
        self.reset(self.config.seed if seed is None else seed)

    # ------------------------------------------------------------------ #
    # Estado inicial
    # ------------------------------------------------------------------ #
    def reset(self, seed=None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        cfg = self.config
        self.step_count = 0
        self.version = 0
        self.score = [0, 0]
        self.hold = [0, 0]
        self.winner = None          # None em jogo; 0/1 vencedor; -1 empate
        self.win_reason = ""
        self.towers = [Tower(k, c) for k, c in
                       enumerate(board.TOWER_CORNERS)]
        self.tank_at = np.full((self.H, self.W), -1, dtype=np.int16)
        self.shape_at = np.full((self.H, self.W), -1, dtype=np.int32)
        self.shapes: list[Shape] = []
        self.bullets: list[Bullet] = []
        self.tanks: list[Tank] = []
        n = int(cfg.team_size)
        ys = np.linspace(8, self.H - 9, n).round().astype(int) if n > 1 \
            else np.array([self.H // 2])
        for team in (0, 1):
            for slot in range(n):
                tank = Tank(len(self.tanks), team, slot)
                x = 1 if team == 0 else board.mirror_x(1)
                tank.x, tank.y = x, int(ys[slot])
                self.tank_at[tank.y, tank.x] = tank.id
                self.tanks.append(tank)
        self._fill_shapes(initial=True)
        self.events = self._empty_events()
        if self.recorder is not None:
            self.recorder.on_reset(self)

    @staticmethod
    def _empty_events():
        return [{key: 0.0 for key in EVENT_KEYS} for _ in (0, 1)]

    def team_tanks(self, team: int) -> list[Tank]:
        return [t for t in self.tanks if t.team == team]

    @property
    def done(self) -> bool:
        return self.winner is not None

    # ------------------------------------------------------------------ #
    # Passo
    # ------------------------------------------------------------------ #
    def step(self, actions) -> list[dict]:
        """Avança um passo. `actions[id] = (move, shoot, upgrade)`.

        Tanques sem ação ficam parados. Devolve os eventos do passo por time
        (pontos, XP, abates, mortes, dano, capturas, torres perdidas).
        """
        if self.done:
            raise RuntimeError("partida terminada; chame reset()")
        self.events = self._empty_events()
        acts = {}
        for tank in self.tanks:
            a = actions.get(tank.id) if isinstance(actions, dict) else (
                actions[tank.id] if tank.id < len(actions) else None)
            if a is None:
                a = (0, 0, 0)
            move, shoot, upgrade = (int(v) for v in a)
            acts[tank.id] = (move, shoot, upgrade)
        if self.recorder is not None:
            self.recorder.on_step(self, acts)

        for tank in self.tanks:
            if tank.alive and acts[tank.id][2]:
                self.apply_upgrade(tank, acts[tank.id][2])
        self._move_phase(acts)
        self._shoot_phase(acts)
        self._tower_phase()
        self._bullet_phase()
        self._invader_phase()
        self._shape_physics_phase()
        self._contact_phase()
        self._base_and_regen_phase()
        self._resolve_deaths()
        self._respawn_phase()
        self._auto_upgrades()
        self._fill_shapes()
        self._score_phase()
        self.version += 1
        return self.events

    # ------------------------------------------------------------------ #
    # Evoluções
    # ------------------------------------------------------------------ #
    def legal_upgrades(self, tank: Tank) -> list[int]:
        """Códigos de upgrade (1..10) válidos agora para `tank`."""
        legal = []
        if tank.pending_points > 0:
            legal += [1 + i for i in range(N_ATTRIBUTES)
                      if tank.points[i] < MAX_ATTR_POINTS]
        legal += [6 + i for i in range(len(tank.class_options))]
        return legal

    def apply_upgrade(self, tank: Tank, code: int) -> bool:
        code = int(code)
        if 1 <= code <= N_ATTRIBUTES:
            attr = code - 1
            if tank.pending_points <= 0 or tank.points[attr] >= \
                    MAX_ATTR_POINTS:
                return False
            old_max = tank.max_hp
            tank.points[attr] += 1
            tank.history.append(attr)
            tank.hp += max(0, tank.max_hp - old_max)
            return True
        if 6 <= code <= 10:
            options = tank.class_options
            k = code - 6
            if k >= len(options):
                return False
            old_max = tank.max_hp
            tank.cls = CLASS_INDEX[options[k]]
            tank.hp = max(1.0, min(tank.max_hp,
                                   tank.hp + tank.max_hp - old_max))
            tank.reload = 0
            tank.burst_left = 0
            return True
        return False

    def _auto_upgrades(self) -> None:
        for tank in self.tanks:
            if tank.auto:
                self.auto_upgrade(tank)

    def auto_upgrade(self, tank: Tank) -> None:
        """Gasta pontos pela build padrão e evolui para uma classe sorteada."""
        guard = 0
        while tank.pending_points > 0 and guard < 32:
            guard += 1
            attr = DEFAULT_BUILD[tank.build_pos % len(DEFAULT_BUILD)]
            tank.build_pos += 1
            if tank.points[attr] < MAX_ATTR_POINTS:
                self.apply_upgrade(tank, 1 + attr)
            elif all(p >= MAX_ATTR_POINTS for p in tank.points):
                break
        options = tank.class_options
        if options:
            # Básico no nível 5 escolhe um ramo (não espera o Esmagador).
            k = int(self.rng.integers(len(options)))
            self.apply_upgrade(tank, 6 + k)

    # ------------------------------------------------------------------ #
    # Movimento
    # ------------------------------------------------------------------ #
    def is_free(self, x: int, y: int) -> bool:
        return (0 <= x < self.W and 0 <= y < self.H and not self.solid[y, x]
                and self.shape_at[y, x] < 0 and self.tank_at[y, x] < 0)

    @property
    def space(self) -> bool:
        return self.config.shape_physics == "space"

    def _try_move(self, tank: Tank, d: int) -> bool:
        dx, dy = DIRS[d]
        nx, ny = tank.x + dx, tank.y + dy
        if not self.is_free(nx, ny):
            if self.space and 0 <= nx < self.W and 0 <= ny < self.H:
                s = self.shape_at[ny, nx]
                if s >= 0:          # encostar numa forma a empurra
                    norm = math.hypot(dx, dy)
                    self.shapes[s].push(TANK_PUSH * dx / norm,
                                        TANK_PUSH * dy / norm)
            return False
        self.tank_at[tank.y, tank.x] = -1
        tank.x, tank.y = nx, ny
        self.tank_at[ny, nx] = tank.id
        return True

    def _move_phase(self, acts) -> None:
        alive = [t for t in self.tanks if t.alive]
        order = self.rng.permutation(len(alive))
        for i in order:
            tank = alive[int(i)]
            move, shoot, _ = acts[tank.id]
            klass = tank.klass
            if tank.dash_timer > 0:
                tank.dash_timer -= 1
            if klass.dash and shoot and tank.dash_timer == 0:
                for _ in range(klass.dash):
                    if not self._try_move(tank, shoot):
                        break
                tank.dash_timer = klass.dash_cooldown
                tank.facing = shoot
                continue
            if move:
                for _ in range(klass.speed):
                    if not self._try_move(tank, move):
                        break
            if shoot:
                tank.facing = shoot
            elif move:
                tank.facing = move

    # ------------------------------------------------------------------ #
    # Tiros
    # ------------------------------------------------------------------ #
    def _fire(self, tank: Tank, aim: int) -> None:
        klass = tank.klass
        damage = bullet_damage(klass, tank.points)
        reach = bullet_range(klass, tank.points)
        for rot, lateral, mult in klass.barrels:
            d = rotate_dir(aim, rot)
            if klass.spread and self.rng.random() < klass.spread:
                d = rotate_dir(d, 1 if self.rng.random() < 0.5 else -1)
            sx, sy = tank.x, tank.y
            if lateral:
                px, py = perpendicular(d)
                lx, ly = sx + lateral * px, sy + lateral * py
                if (0 <= lx < self.W and 0 <= ly < self.H
                        and not self.solid[ly, lx]):
                    sx, sy = lx, ly
            self.bullets.append(Bullet(tank.team, tank.id, sx, sy, d,
                                       damage * mult, reach, klass.pierce))

    def _shoot_phase(self, acts) -> None:
        for tank in self.tanks:
            if not tank.alive:
                continue
            if tank.reload > 0:
                tank.reload -= 1
            klass = tank.klass
            if not klass.barrels:
                continue
            if tank.burst_left > 0:
                self._fire(tank, tank.burst_dir)
                tank.burst_left -= 1
                continue
            shoot = acts[tank.id][1]
            if shoot and tank.reload == 0:
                self._fire(tank, shoot)
                tank.reload = reload_interval(klass, tank.points)
                if klass.burst > 1:
                    tank.burst_left = klass.burst - 1
                    tank.burst_dir = shoot

    def _tower_phase(self) -> None:
        for tower in self.towers:
            if tower.owner < 0:
                continue
            if tower.cooldown > 0:
                tower.cooldown -= 1
                continue
            best, best_d = None, TOWER_RANGE + 1
            for tank in self.tanks:
                if tank.alive and tank.team != tower.owner:
                    d = tower.distance(tank.x, tank.y)
                    if d < best_d:
                        best, best_d = tank, d
            if best is None:
                continue
            cx, cy = tower.center
            d = quantize_dir(best.x - cx, best.y - cy)
            sx, sy = min(tower.cells, key=lambda c: chebyshev(
                c[0], c[1], best.x, best.y))
            self.bullets.append(Bullet(tower.owner, -1, sx, sy, d,
                                       TOWER_DAMAGE, TOWER_RANGE + 1))
            tower.cooldown = TOWER_RELOAD

    # ------------------------------------------------------------------ #
    # Balas
    # ------------------------------------------------------------------ #
    def _bullet_phase(self) -> None:
        for b in self.bullets:
            dx, dy = DIRS[b.dir]
            for _ in range(BULLET_SPEED):
                if b.cells_left <= 0:
                    b.alive = False
                    break
                nx, ny = b.x + dx, b.y + dy
                b.cells_left -= 1
                if not (0 <= nx < self.W and 0 <= ny < self.H) \
                        or self.walls[ny, nx]:
                    b.alive = False
                    break
                b.x, b.y = nx, ny
                k = self.tower_at[ny, nx]
                if k >= 0:
                    if self.towers[k].owner != b.team:
                        self._damage_tower(self.towers[k], b.damage,
                                           b.owner, b.team)
                        b.alive = False
                        break
                    continue
                s = self.shape_at[ny, nx]
                if s >= 0:
                    shape = self.shapes[s]
                    if self.space:      # o tiro empurra a forma
                        k = BULLET_KNOCKBACK * b.damage / math.hypot(dx, dy)
                        shape.push(k * dx, k * dy)
                    self._damage_shape(shape, b.damage, b.owner)
                    if not b.pierce:
                        b.alive = False
                        break
                    continue
                t = self.tank_at[ny, nx]
                if t >= 0 and self.tanks[t].team != b.team:
                    self._damage_tank(self.tanks[t], b.damage, b.owner,
                                      b.team)
                    b.alive = False
                    break
            if b.cells_left <= 0:
                b.alive = False
        self.bullets = [b for b in self.bullets if b.alive]

    # ------------------------------------------------------------------ #
    # Dano
    # ------------------------------------------------------------------ #
    def _credit(self, owner) -> Tank | None:
        if owner is None or owner < 0:
            return None
        return self.tanks[owner]

    def _damage_tank(self, tank: Tank, amount, owner, team) -> None:
        if not tank.alive or tank.hp <= 0:
            return
        tank.hp -= amount
        src = self._credit(owner)
        if src is not None:
            tank.last_hit_by = src.id
            src.damage_dealt += amount
            self.events[src.team]["damage"] += amount

    def _damage_shape(self, shape: Shape, amount, owner) -> None:
        if not shape.alive:
            return
        shape.hp -= amount
        if shape.hp <= 0:
            self._remove_shape(shape)
            src = self._credit(owner)
            if src is not None and src.alive:
                self.give_xp(src, SHAPE_XP[shape.kind])

    def _remove_shape(self, shape: Shape) -> None:
        shape.alive = False
        self.shape_at[shape.y, shape.x] = -1

    def _damage_tower(self, tower: Tower, amount, owner, team) -> None:
        if team is None or team < 0 or tower.owner == team:
            return
        tower.hp -= amount
        if tower.hp > 0:
            return
        previous = tower.owner
        tower.owner = team
        tower.hp = TOWER_HP
        tower.cooldown = TOWER_RELOAD
        self.events[team]["captures"] += 1
        if previous >= 0:
            self.events[previous]["towers_lost"] += 1
        src = self._credit(owner)
        if src is not None and src.alive:
            self.give_xp(src, TOWER_XP)

    def give_xp(self, tank: Tank, amount: float) -> None:
        tank.xp += amount
        self.events[tank.team]["xp"] += amount
        level = level_from_xp(tank.xp)
        if level > tank.level:
            tank.level = level

    # ------------------------------------------------------------------ #
    # Invasores, contato, bases
    # ------------------------------------------------------------------ #
    def _invader_phase(self) -> None:
        for shape in self.shapes:
            if not shape.alive or shape.kind != INVADER:
                continue
            target, best = None, INVADER_SIGHT + 1
            for tank in self.tanks:
                if tank.alive:
                    d = chebyshev(shape.x, shape.y, tank.x, tank.y)
                    if d < best:
                        target, best = tank, d
            if target is None:
                if not self.space and self.rng.random() < 0.3:
                    self._step_shape(shape, int(self.rng.integers(1, 9)))
                continue
            if best <= 1:
                self._damage_tank(target, INVADER_BLAST, None, None)
                self._remove_shape(shape)
                continue
            if self.space:
                # Propulsão: acelera rumo à presa, com inércia de nave.
                dx, dy = target.x - shape.fx, target.y - shape.fy
                norm = math.hypot(dx, dy) or 1.0
                shape.vx += INVADER_THRUST * dx / norm
                shape.vy += INVADER_THRUST * dy / norm
                continue
            d = quantize_dir(target.x - shape.x, target.y - shape.y)
            for alt in (d, rotate_dir(d, 1), rotate_dir(d, -1)):
                if self._step_shape(shape, alt):
                    break

    def _step_shape(self, shape: Shape, d: int) -> bool:
        """Movimento discreto de uma casa (física `static`)."""
        dx, dy = DIRS[d]
        nx, ny = shape.x + dx, shape.y + dy
        if self._shape_blocker(nx, ny, shape) is not None:
            return False
        self._relocate(shape, nx, ny)
        shape.fx, shape.fy = float(nx), float(ny)
        return True

    # ------------------------------------------------------------------ #
    # Física de espaço
    # ------------------------------------------------------------------ #
    def _shape_blocker(self, x: int, y: int, shape: Shape):
        """O que impede `shape` de ocupar (x, y): None, "muro", Tank ou
        Shape. Formas não entram nas bases nem saem do mapa."""
        if not (0 <= x < self.W and 0 <= y < self.H) or self.solid[y, x] \
                or self.base[0][y, x] or self.base[1][y, x]:
            return "muro"
        t = self.tank_at[y, x]
        if t >= 0:
            return self.tanks[t]
        s = self.shape_at[y, x]
        if s >= 0 and s != shape.idx:
            return self.shapes[s]
        return None

    def _relocate(self, shape: Shape, x: int, y: int) -> None:
        self.shape_at[shape.y, shape.x] = -1
        shape.x, shape.y = x, y
        self.shape_at[y, x] = shape.idx

    @staticmethod
    def _cell(v: float) -> int:
        return int(math.floor(v + 0.5))

    def _shape_physics_phase(self) -> None:
        if not self.space:
            return
        alive = [s for s in self.shapes if s.alive]
        for i in self.rng.permutation(len(alive)):
            self._integrate(alive[int(i)])

    def _integrate(self, s: Shape) -> None:
        if s.anchor is not None:            # órbita no ninho
            s.vx += NEST_GRAVITY * (s.anchor[0] - s.fx)
            s.vy += NEST_GRAVITY * (s.anchor[1] - s.fy)
        top = INVADER_MAX_SPEED if s.kind == INVADER else SHAPE_MAX_SPEED
        speed = math.hypot(s.vx, s.vy)
        if speed > top:
            s.vx, s.vy = s.vx * top / speed, s.vy * top / speed
        s.angle = (s.angle + s.spin) % (2 * math.pi)

        px, py = s.fx + s.vx, s.fy + s.vy
        nx, ny = self._cell(px), self._cell(py)
        if (nx, ny) == (s.x, s.y):
            s.fx, s.fy = px, py
            return
        if self._shape_blocker(nx, ny, s) is None:
            self._relocate(s, nx, ny)
            s.fx, s.fy = px, py
            return

        # Colisão: descobre o(s) eixo(s) bloqueado(s) e quica neles.
        hit_x = nx != s.x and self._shape_blocker(nx, s.y, s) is not None
        hit_y = ny != s.y and self._shape_blocker(s.x, ny, s) is not None
        if not hit_x and not hit_y:        # bateu só na quina
            hit_x, hit_y = nx != s.x, ny != s.y
        for axis, hit, cell in (("x", hit_x, (nx, s.y)),
                                ("y", hit_y, (s.x, ny))):
            if not hit:
                continue
            other = self._shape_blocker(*cell, s)
            if other is None:               # quina: usa a casa diagonal
                other = self._shape_blocker(nx, ny, s)
            if isinstance(other, Shape):
                self._elastic(s, other, axis)
            elif axis == "x":
                s.vx = -s.vx * RESTITUTION
            else:
                s.vy = -s.vy * RESTITUTION
        # Fica na própria casa, encostado na borda.
        s.fx = min(max(s.fx, s.x - 0.49), s.x + 0.49)
        s.fy = min(max(s.fy, s.y - 0.49), s.y + 0.49)

    @staticmethod
    def _elastic(a: Shape, b: Shape, axis: str) -> None:
        """Choque elástico 1D no eixo: conserva momento e energia."""
        ma, mb = a.mass, b.mass
        ua = a.vx if axis == "x" else a.vy
        ub = b.vx if axis == "x" else b.vy
        va = ((ma - mb) * ua + 2 * mb * ub) / (ma + mb)
        vb = ((mb - ma) * ub + 2 * ma * ua) / (ma + mb)
        if axis == "x":
            a.vx, b.vx = va, vb
        else:
            a.vy, b.vy = va, vb

    def _contact_phase(self) -> None:
        for tank in self.tanks:
            klass = tank.klass
            if not tank.alive or klass.body_damage <= 0:
                continue
            dmg = klass.body_damage + 0.5 * tank.points[2]
            hit_towers = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x, y = tank.x + dx, tank.y + dy
                    if not (0 <= x < self.W and 0 <= y < self.H):
                        continue
                    t = self.tank_at[y, x]
                    if t >= 0 and self.tanks[t].team != tank.team:
                        self._damage_tank(self.tanks[t], dmg, tank.id,
                                          tank.team)
                    s = self.shape_at[y, x]
                    if s >= 0:
                        self._damage_shape(self.shapes[s], dmg, tank.id)
                    k = self.tower_at[y, x]
                    if k >= 0 and k not in hit_towers:
                        hit_towers.add(k)
                        self._damage_tower(self.towers[k], dmg, tank.id,
                                           tank.team)

    def _base_and_regen_phase(self) -> None:
        for tank in self.tanks:
            if not tank.alive:
                continue
            top = tank.max_hp
            if self.base[tank.team][tank.y, tank.x]:
                tank.hp = min(top, tank.hp + BASE_HEAL)
            elif self.base[1 - tank.team][tank.y, tank.x]:
                self._damage_tank(tank, BASE_DAMAGE, None, None)
            if tank.hp < top:
                tank.regen_timer += 1
                if tank.regen_timer >= regen_interval(tank.points):
                    tank.hp = min(top, tank.hp + 1)
                    tank.regen_timer = 0
            else:
                tank.regen_timer = 0

    # ------------------------------------------------------------------ #
    # Mortes e renascimento
    # ------------------------------------------------------------------ #
    def _resolve_deaths(self) -> None:
        for tank in self.tanks:
            if not tank.alive or tank.hp > 0:
                continue
            tank.alive = False
            tank.respawn = int(self.config.respawn_delay)
            tank.deaths += 1
            tank.burst_left = 0
            self.tank_at[tank.y, tank.x] = -1
            self.events[tank.team]["deaths"] += 1
            lost = tank.xp / 2.0
            tank.xp -= lost
            self._shrink_level(tank)
            killer = self._credit(tank.last_hit_by)
            tank.last_hit_by = None
            if killer is not None and killer.team != tank.team:
                killer.kills += 1
                self.events[killer.team]["kills"] += 1
                self.give_xp(killer, lost + KILL_BONUS_XP)

    def _shrink_level(self, tank: Tank) -> None:
        """Ao perder XP o nível cai e os pontos mais recentes voltam."""
        tank.level = level_from_xp(tank.xp)
        while sum(tank.points) > tank.level - 1 and tank.history:
            attr = tank.history.pop()
            tank.points[attr] -= 1
        tank.build_pos = min(tank.build_pos, sum(tank.points))

    def _respawn_phase(self) -> None:
        for tank in self.tanks:
            if tank.alive:
                continue
            tank.respawn -= 1
            if tank.respawn > 0:
                continue
            cells = [c for c in board.spawn_cells(tank.team)
                     if self.is_free(*c)]
            if not cells:
                continue
            x, y = cells[int(self.rng.integers(len(cells)))]
            tank.x, tank.y = x, y
            self.tank_at[y, x] = tank.id
            tank.alive = True
            tank.hp = float(tank.max_hp)
            tank.reload = 0
            tank.dash_timer = 0
            tank.regen_timer = 0

    # ------------------------------------------------------------------ #
    # Formas
    # ------------------------------------------------------------------ #
    def _zone_cells(self, zone, exclude_nest=False):
        key = (zone, exclude_nest)
        if key not in self._free_cells:
            x0, y0, x1, y1 = zone
            nx0, ny0, nx1, ny1 = board.NEST
            cells = []
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if self.solid[y, x]:
                        continue
                    if exclude_nest and nx0 <= x <= nx1 and ny0 <= y <= ny1:
                        continue
                    cells.append((x, y))
            self._free_cells[key] = cells
        return self._free_cells[key]

    def _spawn_pair(self, kind, zone, exclude_nest=False) -> int:
        cells = self._zone_cells(zone, exclude_nest)
        for _ in range(12):
            x, y = cells[int(self.rng.integers(len(cells)))]
            mx = board.mirror_x(x)
            if not (self.is_free(x, y) and self.is_free(mx, y)):
                continue
            if any(t.alive and chebyshev(t.x, t.y, cx, y) <= 1
                   for t in self.tanks for cx in (x, mx)):
                continue
            vx, vy, spin, angle = self._initial_motion(kind, x, y)
            for side, cx in ((1, x), (-1, mx)):
                # O par espelhado deriva espelhado: ninguém leva vantagem.
                anchor = None
                if kind == PENTAGON:
                    ax, ay = self._nest_center()
                    anchor = (ax if side == 1 else board.mirror_x(ax), ay)
                shape = Shape(kind, cx, y, len(self.shapes), vx=side * vx,
                              vy=vy, spin=side * spin,
                              angle=angle if side == 1 else math.pi - angle,
                              anchor=anchor)
                self.shape_at[y, cx] = shape.idx
                self.shapes.append(shape)
            return 2
        return 0

    @staticmethod
    def _nest_center() -> tuple[float, float]:
        x0, y0, x1, y1 = board.NEST
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _initial_motion(self, kind, x, y):
        """Velocidade, giro e ângulo iniciais (metade esquerda)."""
        angle = float(self.rng.uniform(0, 2 * math.pi))
        if not self.space:
            return 0.0, 0.0, 0.0, angle
        spin = float(self.rng.uniform(-SHAPE_MAX_SPIN, SHAPE_MAX_SPIN))
        if kind == PENTAGON:
            # Velocidade tangencial de órbita circular: v = √k · r.
            cx, cy = self._nest_center()
            rx, ry = x - cx, y - cy
            k = math.sqrt(NEST_GRAVITY)
            return -k * ry, k * rx, spin, angle
        heading = float(self.rng.uniform(0, 2 * math.pi))
        speed = float(self.rng.uniform(*SHAPE_DRIFT))
        return speed * math.cos(heading), speed * math.sin(heading), spin, \
            angle

    def _fill_shapes(self, initial=False) -> None:
        cfg = self.config
        # Compacta a lista de vez em quando para os índices não crescerem.
        if len(self.shapes) > 400:
            alive = [s for s in self.shapes if s.alive]
            self.shapes = alive
            self.shape_at[:] = -1
            for i, s in enumerate(alive):
                s.idx = i
                self.shape_at[s.y, s.x] = i
        targets = ((SQUARE, cfg.n_squares, board.FARM, True),
                   (TRIANGLE, cfg.n_triangles, board.FARM, True),
                   (PENTAGON, cfg.n_pentagons, board.NEST, False),
                   (INVADER, cfg.n_invaders, board.INVADER_ZONE, True))
        counts = [0, 0, 0, 0]
        for s in self.shapes:
            if s.alive:
                counts[s.kind] += 1
        for kind, target, zone, excl in targets:
            budget = 999 if initial else 2   # repõe aos poucos
            while counts[kind] + 1 < target and budget > 0:
                got = self._spawn_pair(kind, zone, excl)
                if not got:
                    break
                counts[kind] += got
                budget -= 1

    # ------------------------------------------------------------------ #
    # Placar e fim
    # ------------------------------------------------------------------ #
    def _score_phase(self) -> None:
        owners = [t.owner for t in self.towers]
        for owner in owners:
            if owner >= 0:
                self.score[owner] += 1
                self.events[owner]["score"] += 1
        for team in (0, 1):
            if all(o == team for o in owners):
                self.hold[team] += 1
            else:
                self.hold[team] = 0
        self.step_count += 1
        for team in (0, 1):
            if self.hold[team] >= self.config.hold_to_win:
                self.winner = team
                self.win_reason = "torres"
                return
        if self.step_count >= self.config.max_steps:
            if self.score[0] > self.score[1]:
                self.winner = 0
            elif self.score[1] > self.score[0]:
                self.winner = 1
            else:
                self.winner = -1
            self.win_reason = "tempo"

    # ------------------------------------------------------------------ #
    # Resumo
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        towers = [t.owner for t in self.towers]
        return {
            "winner": self.winner,
            "win_reason": self.win_reason,
            "step": self.step_count,
            "score": list(self.score),
            "towers": towers,
            "kills": [sum(t.kills for t in self.team_tanks(k))
                      for k in (0, 1)],
            "levels": [[t.level for t in self.team_tanks(k)]
                       for k in (0, 1)],
            "classes": [[t.klass.label for t in self.team_tanks(k)]
                        for k in (0, 1)],
        }
