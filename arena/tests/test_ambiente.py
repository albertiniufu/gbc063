import numpy as np

from tankgrid import TankGridConfig
from tankgrid.env import TankGridEnv


def test_reset_duplo_nao_reaproveita_observacao():
    """Dois resets seguidos: a observação é a da partida nova."""
    env = TankGridEnv(TankGridConfig(opponent="idle"))
    env.reset(seed=1)
    obs, _ = env.reset(seed=2)
    novo, _ = TankGridEnv(TankGridConfig(opponent="idle")).reset(seed=2)
    assert np.array_equal(obs, novo)


def _formas(env):
    return [(s.kind, s.x, s.y) for s in env.game.shapes if s.alive]


def test_reset_sem_seed_muda_a_partida():
    """VecEnvs (SB3) chamam reset() sem seed: cada episódio deve ser outro."""
    env = TankGridEnv(TankGridConfig(opponent="idle"))
    env.reset(seed=5)
    env.reset()
    primeira = _formas(env)
    env.reset()
    assert _formas(env) != primeira


def test_reset_sem_seed_e_deterministico():
    a = TankGridEnv(TankGridConfig(opponent="idle", seed=3))
    b = TankGridEnv(TankGridConfig(opponent="idle", seed=3))
    for _ in range(3):
        oa, _ = a.reset()
        ob, _ = b.reset()
        assert np.array_equal(oa, ob)
    a.reset(seed=10)
    b.reset(seed=10)
    a.reset()
    b.reset()
    assert _formas(a) == _formas(b)
