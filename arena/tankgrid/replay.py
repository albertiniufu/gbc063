"""Replays: gravar uma partida e assistir de novo depois.

O motor é determinístico: a mesma config, a mesma seed e as mesmas ações
geram a mesma partida. Por isso o replay guarda só isso (alguns KB por
partida) e re-simula tudo na hora de assistir.

Gravar (qualquer código que use `TankGridGame` ou `TankGridEnv`)::

    from tankgrid.replay import Recorder

    rec = Recorder(env.game)          # ou Recorder(game)
    obs, _ = env.reset(seed=3)        # cada reset começa uma gravação nova
    ...                               # joga a partida
    rec.save("partida.npz")

Assistir::

    python play.py --replay partida.npz

Ou percorrer em código, passo a passo::

    from tankgrid.replay import replay_game
    for game in replay_game("partida.npz"):
        ...                           # estado depois de cada passo

O gravador guarda a última partida desde o `reset` (ou desde que foi
criado, se a partida ainda estava no passo 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import TankGridConfig
from .game import TankGridGame

FORMAT = 1


class Recorder:
    """Grava as ações de cada passo de um `TankGridGame`."""

    def __init__(self, game: TankGridGame):
        if game.step_count != 0:
            raise ValueError(
                "crie o Recorder antes do primeiro passo (ou chame reset)")
        self.game = game
        game.recorder = self
        self.on_reset(game)

    def on_reset(self, game: TankGridGame) -> None:
        self.config = game.config.to_dict()
        self.seed = game.seed
        self.actions: list[list[tuple[int, int, int]]] = []
        self.auto: list[list[bool]] = []

    def on_step(self, game: TankGridGame, acts: dict) -> None:
        self.actions.append([acts[t.id] for t in game.tanks])
        self.auto.append([bool(t.auto) for t in game.tanks])

    def detach(self) -> None:
        if self.game.recorder is self:
            self.game.recorder = None

    def save(self, path) -> Path:
        """Salva a partida gravada em `path` (`.npz`)."""
        if self.seed is None:
            raise ValueError(
                "partida sem seed não pode ser reproduzida; use "
                "reset(seed=...)")
        n = len(self.game.tanks)
        meta = {
            "format": FORMAT,
            "config": self.config,
            "seed": int(self.seed),
            "summary": self.game.summary(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            actions=np.array(self.actions, dtype=np.int16).reshape(-1, n, 3),
            auto=np.array(self.auto, dtype=bool).reshape(-1, n))
        return path


def load_replay(path) -> dict:
    """Lê um replay: `meta` (config, seed, resumo) e as ações por passo."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"formato de replay desconhecido: {path}")
        return {"meta": meta, "actions": data["actions"],
                "auto": data["auto"]}


def replay_game(path):
    """Re-simula o replay e devolve o jogo no passo 0 e após cada passo.

    É sempre o mesmo objeto `TankGridGame`, atualizado a cada `yield`.
    """
    data = load_replay(path)
    meta = data["meta"]
    game = TankGridGame(TankGridConfig(**meta["config"]), seed=meta["seed"])
    yield game
    for acts, auto in zip(data["actions"], data["auto"]):
        for tank, flag in zip(game.tanks, auto):
            tank.auto = bool(flag)
        game.step({t.id: tuple(int(v) for v in acts[t.id])
                   for t in game.tanks})
        yield game


def check_replay(path) -> bool:
    """True se a re-simulação chega ao mesmo resumo gravado."""
    game = None
    for game in replay_game(path):
        pass
    return game is not None and \
        _json(game.summary()) == load_replay(path)["meta"]["summary"]


def _json(value):
    return json.loads(json.dumps(value))
