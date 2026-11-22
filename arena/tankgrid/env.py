"""Ambiente Gymnasium do TankGrid (ponto de vista do time azul).

Dois modos de controle (`TankGridConfig.control`):

- `"team"` (padrão): a política comanda os 3 tanques azuis com
  **parâmetros compartilhados**. Cada `step` decide a ação de um tanque
  vivo; quando todos os vivos decidiram, o mundo avança um passo. A
  observação é sempre a do próximo tanque a decidir. Recompensa do passo
  do mundo entra na sub-etapa que o fez avançar.
- `"single"`: a política comanda só o tanque 0; os aliados são bots
  (`ally_bot`).

O vermelho é controlado por `opponent` (`heuristic`, `random` ou `idle`).

Recompensa (pesos em `TankGridConfig`): diferença de pontos do passo,
capturas/perdas de torre, XP, abates, mortes, dano e ±`reward_win` no fim.
Os eventos brutos vão em `info["events"]` para quem quiser moldar a sua.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agents import make_bot
from .config import TankGridConfig
from .game import TankGridGame
from .observations import (
    N_MOVE_SHOOT,
    ObservationBuilder,
    decode_action,
    n_actions,
)

AGENT_TEAM = 0


class TankGridEnv(gym.Env):
    """3 × 3 de tanques: o agente comanda o azul contra um bot no vermelho."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(self, config: TankGridConfig | None = None,
                 opponent=None, render_mode: str | None = None, **overrides):
        super().__init__()
        cfg = config or TankGridConfig()
        if overrides:
            cfg = TankGridConfig(**{**cfg.to_dict(), **overrides})
        if cfg.control not in ("team", "single"):
            raise ValueError("control deve ser 'team' ou 'single'")
        self.config = cfg
        self.render_mode = render_mode
        self.game = TankGridGame(cfg)
        self.builder = ObservationBuilder(self.game, cfg.view_size)
        self.opponent = opponent if opponent is not None else make_bot(
            cfg.opponent, seed=cfg.seed + 1, view_size=cfg.view_size)
        self.ally_bot = make_bot(cfg.ally_bot, seed=cfg.seed + 2,
                                 view_size=cfg.view_size)
        self.action_space = spaces.Discrete(n_actions(cfg.auto_upgrade))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.builder.shape, dtype=np.float32)
        self._renderer = None
        self._pending: dict[int, tuple[int, int, int]] = {}
        self._queue: list[int] = []
        self._current = 0
        self._totals = None
        self._invalid = 0
        self._next_seed = int(cfg.seed)

    # ------------------------------------------------------------------ #
    @property
    def controlled(self) -> list[int]:
        blue = [t.id for t in self.game.team_tanks(AGENT_TEAM)]
        return blue if self.config.control == "team" else blue[:1]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Sem seed, cada reset joga a partida seguinte (seed + 1), como o
        # Gymnasium espera; antes todo reset() repetia a mesma partida.
        # Continua determinístico: a 1ª partida usa `config.seed`.
        game_seed = self._next_seed if seed is None else int(seed)
        self._next_seed = game_seed + 1
        self.game.reset(game_seed)
        for bot, offset in ((self.opponent, 1), (self.ally_bot, 2)):
            if hasattr(bot, "reset"):
                bot.reset(game_seed + offset)
        controlled = set(self.controlled)
        for tank in self.game.tanks:
            tank.auto = (tank.id not in controlled
                         or self.config.auto_upgrade)
        self._totals = {"reward": 0.0, "xp": 0.0, "kills": 0.0,
                        "deaths": 0.0, "captures": 0.0, "damage": 0.0}
        self._invalid = 0
        self._pending = {}
        self._queue = []
        self._phi = self.tower_potential()
        self._advance_until_decision()
        return self._obs(), self._info(None)

    def step(self, action):
        if self.game.done:
            raise RuntimeError("episódio terminado; chame reset()")
        tank = self.game.tanks[self._current]
        move, shoot, upgrade = decode_action(int(action), tank.team)
        if upgrade and upgrade not in self.game.legal_upgrades(tank):
            self._invalid += 1
            upgrade = 0
        self._pending[tank.id] = (move, shoot, upgrade)
        reward = 0.0
        events = None
        if self._queue:
            self._current = self._queue.pop(0)
        else:
            events, reward = self._world_step()
            self._advance_until_decision(collect=True)
            reward += self._extra_reward
        terminated = self.game.done
        info = self._info(events)
        if terminated:
            info["final"] = self.game.summary()
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ #
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[:N_MOVE_SHOOT] = True
        if self.action_space.n > N_MOVE_SHOOT and not self.game.done:
            tank = self.game.tanks[self._current]
            for code in self.game.legal_upgrades(tank):
                mask[N_MOVE_SHOOT + code - 1] = True
        return mask

    def render(self):
        if self._renderer is None:
            from .renderer import TankGridRenderer
            self._renderer = TankGridRenderer()
        return self._renderer.frame(self.game)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        return self.builder.observe(self._current)

    def _advance_until_decision(self, collect: bool = False) -> None:
        """Monta a fila de tanques a decidir; avança o mundo se vazia."""
        self._extra_reward = 0.0
        while not self.game.done:
            alive = [i for i in self.controlled
                     if self.game.tanks[i].alive]
            if alive:
                self._current = alive[0]
                self._queue = alive[1:]
                self._pending = {}
                return
            # Todos os nossos tanques estão mortos: o mundo segue sem nós.
            events, reward = self._world_step()
            if collect:
                self._extra_reward += reward
        self._queue = []

    def _world_step(self):
        game = self.game
        actions = dict(self._pending)
        controlled = set(self.controlled)
        for tank in game.tanks:
            if tank.id in actions or not tank.alive:
                continue
            if tank.team != AGENT_TEAM:
                actions[tank.id] = self.opponent.act(game, tank)
            elif tank.id not in controlled:
                actions[tank.id] = self.ally_bot.act(game, tank)
        events = game.step(actions)
        reward = self._reward(events)
        self._totals["reward"] += reward
        for key in ("xp", "kills", "deaths", "captures", "damage"):
            self._totals[key] += events[AGENT_TEAM][key]
        return events, reward

    def tower_potential(self) -> float:
        """Φ(s) = −distância média (BFS) dos nossos tanques até a torre mais
        próxima que não é nossa, em unidades de ~40 casas. Tanque morto
        conta como distância máxima; com as 3 torres nossas, Φ = 0."""
        game = self.game
        targets = [t.index for t in game.towers if t.owner != AGENT_TEAM]
        if not targets:
            return 0.0
        total = 0.0
        tanks = game.team_tanks(AGENT_TEAM)
        for tank in tanks:
            if tank.alive:
                d = min(int(game.dist_to_tower[k][tank.y, tank.x])
                        for k in targets)
                total += min(d, 40)
            else:
                total += 40
        return -total / (40.0 * len(tanks))

    def _reward(self, events) -> float:
        cfg = self.config
        own, opp = events[AGENT_TEAM], events[1 - AGENT_TEAM]
        r = cfg.reward_score * (own["score"] - opp["score"])
        r += cfg.reward_capture * (own["captures"] - own["towers_lost"])
        r += cfg.reward_xp * own["xp"]
        r += cfg.reward_kill * own["kills"]
        r += cfg.reward_death * own["deaths"]
        r += cfg.reward_damage * own["damage"]
        if cfg.reward_approach:
            # F = Φ(s') − Φ(s) (γ = 1, episódico): potencial telescópico.
            phi = self.tower_potential()
            r += cfg.reward_approach * (phi - self._phi)
            self._phi = phi
        if self.game.done:
            if self.game.winner == AGENT_TEAM:
                r += cfg.reward_win
            elif self.game.winner == 1 - AGENT_TEAM:
                r -= cfg.reward_win
        return float(r)

    def _info(self, events) -> dict:
        game = self.game
        return {
            "winner": game.winner,
            "step": game.step_count,
            "acting_tank": self._current,
            "score": tuple(game.score),
            "towers": tuple(t.owner for t in game.towers),
            "events": events,
            "totals": dict(self._totals or {}),
            "invalid_upgrades": self._invalid,
        }
