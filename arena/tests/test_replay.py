import numpy as np
import pytest

from tankgrid import TankGridConfig, TankGridGame
from tankgrid.agents import make_bot
from tankgrid.env import TankGridEnv
from tankgrid.replay import Recorder, check_replay, load_replay, replay_game


def _partida_de_bots(seed=11, passos=300):
    game = TankGridGame(TankGridConfig(max_steps=passos), seed=seed)
    rec = Recorder(game)
    bots = [make_bot("heuristic", seed=seed + 1), make_bot("random", seed=2)]
    for tank in game.tanks:
        tank.auto = True
    while not game.done:
        game.step({t.id: bots[t.team].act(game, t)
                   for t in game.tanks if t.alive})
    return game, rec


def test_replay_reproduz_a_partida(tmp_path):
    game, rec = _partida_de_bots()
    arquivo = rec.save(tmp_path / "p.npz")
    for final in replay_game(arquivo):
        pass
    assert final.summary() == game.summary()
    assert check_replay(arquivo)
    assert arquivo.stat().st_size < 50_000


def test_replay_do_ambiente_grava_a_ultima_partida(tmp_path):
    env = TankGridEnv(TankGridConfig(max_steps=150))
    rec = Recorder(env.game)
    rng = np.random.default_rng(0)
    for seed in (3, 4):                 # cada reset recomeça a gravação
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(rng.integers(81)))
    arquivo = rec.save(tmp_path / "env.npz")
    dados = load_replay(arquivo)
    assert dados["meta"]["seed"] == 4
    assert len(dados["actions"]) == env.game.step_count
    for final in replay_game(arquivo):
        pass
    assert final.summary() == env.game.summary()


def test_replay_parcial(tmp_path):
    """Uma partida interrompida também pode ser assistida."""
    game = TankGridGame(TankGridConfig(), seed=1)
    rec = Recorder(game)
    for _ in range(40):
        game.step({0: (1, 1, 0), 3: (5, 5, 0)})
    assert check_replay(rec.save(tmp_path / "parcial.npz"))


def test_recorder_exige_passo_zero():
    game = TankGridGame(TankGridConfig(), seed=1)
    game.step({})
    with pytest.raises(ValueError):
        Recorder(game)


def test_partida_sem_seed_nao_salva(tmp_path):
    game = TankGridGame(TankGridConfig(), seed=1)
    rec = Recorder(game)
    game.reset(None)
    with pytest.raises(ValueError):
        rec.save(tmp_path / "x.npz")


def test_play_assiste_replay(tmp_path):
    play = pytest.importorskip("play")
    _, rec = _partida_de_bots(passos=30)
    arquivo = rec.save(tmp_path / "p.npz")
    play.main(["--replay", str(arquivo), "--fps", "0"])
