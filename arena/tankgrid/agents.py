"""Bots de referência e adaptador de políticas treinadas.

Todo bot implementa `act(game, tank) -> (move, shoot, upgrade)` em
coordenadas do mundo. Bots usam upgrade automático (o ambiente liga
`tank.auto`), então devolvem `upgrade = 0`.

- `IdleBot`: fica parado (útil em testes).
- `RandomBot`: anda e atira ao acaso — o primeiro adversário a vencer.
- `HeuristicBot`: farma formas, alinha a mira (tiros só andam nas 8
  direções), foge para a base com pouca vida, caça inimigos ao alcance e,
  depois de crescer, toma as torres seguindo mapas de distância (BFS).
- `PolicyBot`: adaptador para uma política que recebe a observação
  canônica do time (`act(obs) -> int`), como a `Agent` do aluno.
"""
from __future__ import annotations

import numpy as np

from .classes import DIRS, bullet_range
from .game import INVADER, TankGridGame, Tank, chebyshev, quantize_dir
from .observations import ObservationBuilder, decode_action


class IdleBot:
    name = "idle"

    def reset(self, seed=None):
        pass

    def act(self, game: TankGridGame, tank: Tank):
        return (0, 0, 0)


class RandomBot:
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, game, tank):
        return (int(self.rng.integers(9)), int(self.rng.integers(9)), 0)


def _aligned_dir(dx: int, dy: int) -> int:
    """Direção 1..8 se (dx, dy) está numa das 8 linhas de tiro; senão 0."""
    if dx == 0 and dy == 0:
        return 0
    if dx == 0 or dy == 0 or abs(dx) == abs(dy):
        return quantize_dir(dx, dy)
    return 0


def _misalignment(dx: int, dy: int) -> int:
    ax, ay = abs(dx), abs(dy)
    return min(ax, ay, abs(ax - ay))


class HeuristicBot:
    """Bot de regras com mira alinhada e navegação por BFS."""

    name = "heuristic"

    def __init__(self, seed=None, farm_until_level: int = 6,
                 retreat_hp: float = 0.35, noise: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.farm_until_level = int(farm_until_level)
        self.retreat_hp = float(retreat_hp)
        self.noise = float(noise)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def act(self, game: TankGridGame, tank: Tank):
        if not tank.alive:
            return (0, 0, 0)
        if self.rng.random() < self.noise:
            return (int(self.rng.integers(9)), int(self.rng.integers(9)), 0)
        reach = bullet_range(tank.klass, tank.points)
        smasher = not tank.klass.barrels

        # 1) Pouca vida: volta para a base atirando no inimigo mais perto.
        if tank.hp < self.retreat_hp * tank.max_hp:
            move = self._descend(game, tank, game.dist_to_base[tank.team])
            enemy = self._nearest_enemy(game, tank, reach)
            shoot = self._shot_at(tank, enemy.x, enemy.y) if enemy else 0
            return (move, shoot, 0)

        # 2) Inimigo perto: briga (o Esmagador investe).
        enemy = self._nearest_enemy(game, tank, min(reach, 9))
        if enemy is not None:
            if smasher:
                d = quantize_dir(enemy.x - tank.x, enemy.y - tank.y)
                return (d, d if chebyshev(tank.x, tank.y, enemy.x,
                                          enemy.y) <= 4 else 0, 0)
            move = self._approach_align(game, tank, enemy.x, enemy.y, 4)
            return (move, self._shot_at(tank, enemy.x, enemy.y, move), 0)

        # 3) Invasor chegando: atira antes que exploda.
        threat = self._nearest_shape(game, tank, 4, invader=True)
        if threat is not None and not smasher:
            move = self._approach_align(game, tank, threat.x, threat.y, 3)
            return (move, self._shot_at(tank, threat.x, threat.y, move), 0)

        # 4) Cresceu (ou o tempo passou): toma a torre que não é nossa.
        late = game.step_count > 0.4 * game.config.max_steps
        if tank.level >= self.farm_until_level or late:
            target = self._target_tower(game, tank)
            if target is not None:
                dist_map = game.dist_to_tower[target.index]
                tx, ty = min(target.cells, key=lambda c: chebyshev(
                    c[0], c[1], tank.x, tank.y))
                near = chebyshev(tank.x, tank.y, tx, ty)
                if smasher:
                    move = self._descend(game, tank, dist_map)
                    d = quantize_dir(tx - tank.x, ty - tank.y)
                    return (move if near > 1 else d,
                            d if near <= 3 else 0, 0)
                if dist_map[tank.y, tank.x] > 6:
                    move = self._descend(game, tank, dist_map)
                else:
                    move = self._approach_align(game, tank, tx, ty, 3)
                shoot = self._shot_at(tank, tx, ty, move) \
                    if near <= reach else 0
                return (move, shoot, 0)

        # 5) Farm: forma mais próxima.
        shape = self._nearest_shape(game, tank, 12)
        if shape is not None:
            if smasher:
                return (quantize_dir(shape.x - tank.x, shape.y - tank.y),
                        0, 0)
            move = self._approach_align(game, tank, shape.x, shape.y, 2)
            return (move, self._shot_at(tank, shape.x, shape.y, move), 0)
        # Nada por perto: vai para o centro (torre do meio).
        return (self._descend(game, tank, game.dist_to_tower[1]), 0, 0)

    # ------------------------------------------------------------------ #
    def _nearest_enemy(self, game, tank, radius):
        best, best_d = None, radius + 1
        for other in game.tanks:
            if other.alive and other.team != tank.team:
                d = chebyshev(tank.x, tank.y, other.x, other.y)
                if d < best_d:
                    best, best_d = other, d
        return best

    def _nearest_shape(self, game, tank, radius, invader=False):
        best, best_d = None, radius + 1
        y0, y1 = max(0, tank.y - radius), min(game.H, tank.y + radius + 1)
        x0, x1 = max(0, tank.x - radius), min(game.W, tank.x + radius + 1)
        window = game.shape_at[y0:y1, x0:x1]
        for idx in window[window >= 0]:
            shape = game.shapes[int(idx)]
            if not shape.alive or (shape.kind == INVADER) != invader:
                continue
            # Desempate por distância euclidiana e sorteio, para não
            # favorecer um lado do mapa pela ordem de varredura.
            d = (chebyshev(tank.x, tank.y, shape.x, shape.y)
                 + 0.01 * ((tank.x - shape.x) ** 2
                           + (tank.y - shape.y) ** 2) ** 0.5
                 + 0.001 * self.rng.random())
            if d < best_d:
                best, best_d = shape, d
        return best

    def _target_tower(self, game, tank):
        candidates = [t for t in game.towers if t.owner != tank.team]
        if not candidates:
            # Todas nossas: protege a do meio.
            return None
        return min(candidates,
                   key=lambda t: game.dist_to_tower[t.index][tank.y, tank.x])

    def _descend(self, game, tank, dist_map) -> int:
        """Passo que mais diminui a distância no mapa BFS."""
        best, best_val = 0, dist_map[tank.y, tank.x]
        dirs = list(range(1, 9))
        self.rng.shuffle(dirs)
        for d in dirs:
            dx, dy = DIRS[d]
            nx, ny = tank.x + dx, tank.y + dy
            if game.is_free(nx, ny) and dist_map[ny, nx] < best_val:
                best, best_val = d, dist_map[ny, nx]
        return best

    def _approach_align(self, game, tank, tx, ty, desired) -> int:
        """Escolhe o passo que alinha a mira e mantém a distância `desired`."""
        best, best_score = 0, None
        for d in range(9):
            dx, dy = DIRS[d]
            nx, ny = tank.x + dx, tank.y + dy
            if d and not game.is_free(nx, ny):
                continue
            rx, ry = tx - nx, ty - ny
            dist = max(abs(rx), abs(ry))
            score = 3 * _misalignment(rx, ry) + abs(dist - desired)
            score += 0.1 * self.rng.random()
            if best_score is None or score < best_score:
                best, best_score = d, score
        return best

    def _shot_at(self, tank, tx, ty, move: int = 0) -> int:
        """Atira se o alvo está alinhado agora ou depois do movimento."""
        dx, dy = DIRS[move]
        aligned = _aligned_dir(tx - tank.x - dx, ty - tank.y - dy)
        if aligned:
            return aligned
        return _aligned_dir(tx - tank.x, ty - tank.y)


def default_greedy(agent) -> bool:
    """Políticas on-policy (PPO, A2C...) jogam amostrando; o argmax delas
    costuma travar em laços (tanque empurrando uma forma para sempre).
    Métodos de valor (DQN...) jogam gulosos."""
    return not bool(getattr(agent, "on_policy", False))


class SampledPolicy:
    """Envolve um agente para que `act(obs, greedy=True)` siga `greedy`.

    Serve para usar o avaliador comum (que sempre pede `greedy=True`) com
    políticas estocásticas.
    """

    def __init__(self, agent, greedy: bool):
        self.agent = agent
        self.greedy = bool(greedy)
        self.device = getattr(agent, "device", None)
        self.on_policy = getattr(agent, "on_policy", False)

    def act(self, obs, greedy=True, **kwargs):
        return self.agent.act(obs, greedy=self.greedy)


class PolicyBot:
    """Adaptador para uma política que recebe a observação canônica."""

    name = "policy"

    def __init__(self, agent, view_size: int = 15, greedy=None):
        self.agent = agent
        self.view_size = int(view_size)
        self.greedy = default_greedy(agent) if greedy is None else greedy
        self._builder = None

    def reset(self, seed=None):
        if hasattr(self.agent, "reset"):
            self.agent.reset(seed)
        self._builder = None

    def act(self, game, tank):
        if not tank.alive:
            return (0, 0, 0)
        if self._builder is None or self._builder.game is not game:
            self._builder = ObservationBuilder(game, self.view_size)
        obs = self._builder.observe(tank.id)
        try:
            action = self.agent.act(obs, greedy=self.greedy)
        except TypeError:
            action = self.agent.act(obs)
        return decode_action(int(action), tank.team)


BOT_NAMES = ("heuristic", "random", "idle")


def make_bot(spec: str, seed=None, view_size: int = 15):
    """`heuristic`, `random` ou `idle`."""
    spec = str(spec)
    if spec == "heuristic":
        return HeuristicBot(seed=seed)
    if spec == "random":
        return RandomBot(seed=seed)
    if spec == "idle":
        return IdleBot()
    raise ValueError(
        f"bot desconhecido: {spec!r} (use heuristic, random ou idle)")
