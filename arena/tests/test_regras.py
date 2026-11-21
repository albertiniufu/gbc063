import pytest

from tankgrid.classes import CLASS_INDEX

from helpers import duelo


@pytest.mark.parametrize("classe", ["basico", "duplo", "artilheiro",
                                    "duplo_flanco", "triplo", "flanco"])
def test_classe_acerta_alvo_alinhado(classe):
    """Mirando direto num tanque a 6 casas, toda classe com canhão acerta."""
    game, a, b = duelo()
    a.cls = CLASS_INDEX[classe]
    b.hp = 1e6
    for _ in range(30):
        game.step({a.id: (0, 1, 0)})
    assert b.hp < 1e6, classe


def test_recarga_pronta_quando_o_tiro_sai():
    """O escalar "recarga pronta" vale 1 exatamente nos passos em que o
    tanque atiraria se pedisse."""
    from tankgrid.observations import N_SPATIAL, ObservationBuilder
    game, a, b = duelo()
    b.hp = 1e6
    builder = ObservationBuilder(game)
    for _ in range(12):
        pronta = builder.observe(a.id)[N_SPATIAL + 2, 0, 0]
        antes = {id(x) for x in game.bullets}
        game.step({a.id: (0, 1, 0)})
        atirou = any(x.owner == a.id and id(x) not in antes
                     for x in game.bullets)
        assert pronta == float(atirou)


def _morre_na_base_inimiga(espera):
    """O azul acerta o vermelho uma vez, espera `espera` passos e o
    vermelho morre na base azul (sem dono do dano). Devolve os abates."""
    from tankgrid.game import BASE_DAMAGE
    game, a, b = duelo()
    game.step({a.id: (0, 1, 0)})
    for _ in range(3):
        game.step({})
    assert b.last_hit_by == a.id
    for _ in range(espera):
        game.step({})
    game.tank_at[b.y, b.x] = -1
    b.x, b.y = 1, 15
    game.tank_at[b.y, b.x] = b.id
    b.hp = BASE_DAMAGE
    game.step({})
    assert not b.alive
    return a.kills


def test_abate_recente_tem_credito():
    assert _morre_na_base_inimiga(espera=5) == 1


def test_credito_de_abate_expira():
    assert _morre_na_base_inimiga(espera=100) == 0


def test_torre_atira_a_cada_3_passos():
    from tankgrid.game import TOWER_RELOAD
    game, a, b = duelo()
    torre = game.towers[1]                   # cantos (23..24, 14..15)
    torre.owner = 0
    game.tank_at[b.y, b.x] = -1
    b.x, b.y = 28, 15                        # 4 casas a leste da torre
    game.tank_at[b.y, b.x] = b.id
    b.hp = 1e6
    for _ in range(6):                       # passa o cooldown inicial
        game.step({})
    tiros = 0
    for _ in range(30):
        antes = {id(x) for x in game.bullets}
        game.step({})
        tiros += sum(x.owner == -1 and id(x) not in antes
                     for x in game.bullets)
    assert TOWER_RELOAD == 3 and tiros == 30 // TOWER_RELOAD


def test_renascimento_leva_respawn_delay_passos():
    """Morto no passo t, o tanque fica fora de `respawn_delay` decisões."""
    game, a, b = duelo()
    b.hp = 0.5
    game.step({a.id: (0, 1, 0)})            # mata no 3º passo (6 casas)
    while b.alive:
        game.step({})
    fora = 0
    while not b.alive:
        fora += 1
        game.step({})
    assert fora == game.config.respawn_delay == 8
