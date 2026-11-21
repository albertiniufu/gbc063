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
