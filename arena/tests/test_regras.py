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
