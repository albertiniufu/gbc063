"""Contratos que treino, replays e torneio assumem."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tankgrid import TankGridConfig, TankGridGame
from tankgrid.agents import make_bot
from tankgrid.env import TankGridEnv
from tankgrid.observations import ObservationBuilder


@pytest.mark.parametrize("control", ["team", "single"])
def test_check_env(control):
    env = TankGridEnv(TankGridConfig(control=control, max_steps=50))
    check_env(env, skip_render_check=True)


def test_espelho_no_inicio():
    """O mapa é simétrico: no passo 0 o tanque k do vermelho vê o mesmo
    que o tanque k do azul (a observação do vermelho é espelhada)."""
    game = TankGridGame(TankGridConfig(), seed=4)
    builder = ObservationBuilder(game)
    for k in range(3):
        assert np.allclose(builder.observe(k), builder.observe(3 + k))


def _partida(seed):
    game = TankGridGame(TankGridConfig(max_steps=300), seed=seed)
    bots = [make_bot("heuristic", seed=seed + 1),
            make_bot("heuristic", seed=seed + 2)]
    for tank in game.tanks:
        tank.auto = True
    while not game.done:
        game.step({t.id: bots[t.team].act(game, t)
                   for t in game.tanks if t.alive})
    return game.summary()


def test_partida_deterministica():
    assert _partida(11) == _partida(11)


def test_observacao_dentro_do_espaco():
    env = TankGridEnv(TankGridConfig(max_steps=300))
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        assert env.observation_space.contains(obs)
        obs, _, done, _, _ = env.step(int(rng.integers(81)))
