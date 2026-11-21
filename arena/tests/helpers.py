"""Cenários pequenos e controlados para testar regras do motor."""
from tankgrid import TankGridConfig, TankGridGame

VAZIO = dict(n_squares=0, n_triangles=0, n_pentagons=0, n_invaders=0)


def duelo(**config):
    """Partida sem formas com só dois tanques vivos: o azul 0 em (10, 15)
    e o vermelho 3 em (16, 15), alinhados a leste. Os demais ficam mortos
    para sempre."""
    game = TankGridGame(TankGridConfig(**{**VAZIO, **config}))
    a, b = game.tanks[0], game.tanks[3]
    for tank in game.tanks:
        game.tank_at[tank.y, tank.x] = -1
        if tank not in (a, b):
            tank.alive = False
            tank.respawn = 10 ** 9
    a.x, a.y = 10, 15
    b.x, b.y = 16, 15
    game.tank_at[a.y, a.x] = a.id
    game.tank_at[b.y, b.x] = b.id
    return game, a, b
