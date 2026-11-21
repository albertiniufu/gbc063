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
