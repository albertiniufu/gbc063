"""Classes de tanque, árvore de evolução, atributos e tabela de níveis.

Tudo aqui é dado puro (sem estado): o motor (`game.py`) consulta estas
tabelas para calcular dano, recarga, alcance e as opções de evolução.

Direções são índices 1..8 (0 = nenhuma), no sentido anti-horário a partir
do leste: E, NE, N, NO, O, SO, S, SE. Um cano com `rot = k` atira na direção
girada de `k × 45°` em relação à mira do tanque.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------- #
# Direções (y cresce para baixo, como na tela)
# --------------------------------------------------------------------- #
DIRS: tuple[tuple[int, int], ...] = (
    (0, 0),    # 0 parado / não atira
    (1, 0),    # 1 E
    (1, -1),   # 2 NE
    (0, -1),   # 3 N
    (-1, -1),  # 4 NO
    (-1, 0),   # 5 O
    (-1, 1),   # 6 SO
    (0, 1),    # 7 S
    (1, 1),    # 8 SE
)
N_DIRS = 9
# Espelho horizontal (x → W-1-x): E↔O, NE↔NO, SE↔SO.
MIRROR_DIR = (0, 5, 4, 3, 2, 1, 8, 7, 6)


def rotate_dir(d: int, k: int) -> int:
    """Gira a direção `d` (1..8) em `k` passos de 45° anti-horários."""
    if d == 0:
        return 0
    return (d - 1 + k) % 8 + 1


def perpendicular(d: int) -> tuple[int, int]:
    """Deslocamento lateral (para a esquerda da mira) de um cano duplo."""
    dx, dy = DIRS[rotate_dir(d, 2)]
    return dx, dy


# --------------------------------------------------------------------- #
# Atributos
# --------------------------------------------------------------------- #
ATTRIBUTES = ("vida", "regeneracao", "dano", "recarga", "alcance")
ATTR_LABELS = ("Vida máxima", "Regeneração", "Dano", "Recarga", "Alcance")
N_ATTRIBUTES = len(ATTRIBUTES)
MAX_ATTR_POINTS = 5
# Intervalo de regeneração (passos por 1 de vida) por pontos investidos.
REGEN_INTERVAL = (10, 8, 6, 5, 4, 3)

# --------------------------------------------------------------------- #
# Níveis: XP acumulado para atingir cada nível (1..15)
# --------------------------------------------------------------------- #
LEVEL_XP = (0, 4, 10, 18, 28, 40, 54, 70, 88, 108, 130, 154, 180, 208, 238)
MAX_LEVEL = len(LEVEL_XP)
EVOLVE_LEVELS = (5, 10)


def level_from_xp(xp: float) -> int:
    level = 1
    for lvl, need in enumerate(LEVEL_XP, start=1):
        if xp >= need:
            level = lvl
    return level


# --------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class TankClass:
    key: str
    label: str
    tier: int                       # 0 = Básico, 1 = nível 5, 2 = nível 10
    # (rot em passos de 45°, deslocamento lateral -1/0/1, mult. de dano)
    barrels: tuple[tuple[int, int, float], ...] = ((0, 0, 1.0),)
    reload_mult: float = 1.0
    reload_fixed: int | None = None  # intervalo fixo (metralhadoras)
    range_bonus: int = 0
    hp_bonus: int = 0
    speed: int = 1
    spread: float = 0.0              # prob. de desviar o tiro em ±45°
    burst: int = 1                   # tiros seguidos por disparo (Caçador)
    pierce: bool = False             # atravessa formas (Destruidor)
    body_damage: float = 0.0         # dano por contato a cada passo
    dash: int = 0                    # casas da investida (Esmagador)
    dash_cooldown: int = 0
    description: str = ""


CLASSES: tuple[TankClass, ...] = (
    TankClass("basico", "Básico", 0,
              description="Um canhão, tudo equilibrado."),
    # ---- nível 5 ----
    TankClass("duplo", "Duplo", 1,
              barrels=((0, -1, 0.8), (0, 1, 0.8)),
              description="Dois canos lado a lado."),
    TankClass("atirador", "Atirador", 1,
              barrels=((0, 0, 2.0),), reload_mult=1.5, range_bonus=6,
              hp_bonus=-2, description="Canhão longo: dano 2, alcance 14."),
    TankClass("metralhadora", "Metralhadora", 1,
              barrels=((0, 0, 0.6),), reload_fixed=1, spread=0.3,
              description="Atira todo passo, tiros fracos que se espalham."),
    TankClass("flanco", "Flanco", 1,
              barrels=((0, 0, 1.0), (4, 0, 0.7)),
              description="Um canhão na frente e outro atrás."),
    # ---- nível 10 ----
    TankClass("triplo", "Triplo", 2,
              barrels=((0, 0, 1.0), (1, 0, 0.8), (-1, 0, 0.8)),
              description="Três tiros em leque."),
    TankClass("quadruplo", "Quádruplo", 2,
              barrels=((0, 0, 1.0), (2, 0, 1.0), (4, 0, 1.0), (6, 0, 1.0)),
              reload_mult=1.25, description="Quatro canhões em cruz."),
    TankClass("duplo_flanco", "Duplo-flanco", 2,
              barrels=((0, -1, 0.8), (0, 1, 0.8), (4, -1, 0.6),
                       (4, 1, 0.6)),
              description="Dois canos na frente e dois atrás."),
    TankClass("assassino", "Assassino", 2,
              barrels=((0, 0, 3.0),), reload_mult=2.0, range_bonus=12,
              hp_bonus=-2, description="Dano 3 a até 20 casas."),
    TankClass("cacador", "Caçador", 2,
              barrels=((0, 0, 1.5),), reload_mult=1.5, range_bonus=6,
              burst=2, description="Rajada de dois tiros, de longe."),
    TankClass("destruidor", "Destruidor", 2,
              barrels=((0, 0, 3.5),), reload_mult=1.75, pierce=True,
              description="Bala enorme que atravessa formas."),
    TankClass("artilheiro", "Artilheiro", 2,
              barrels=((0, -1, 0.6), (0, 1, 0.6)), reload_fixed=1,
              description="Dois canos que atiram todo passo."),
    TankClass("tri_angulo", "Tri-Ângulo", 2,
              barrels=((0, 0, 1.0), (3, 0, 0.6), (5, 0, 0.6)), speed=2,
              description="Anda 2 casas por passo; propulsores atiram "
                          "para trás."),
    TankClass("esmagador", "Esmagador", 2,
              barrels=(), hp_bonus=6, body_damage=1.5, dash=3,
              dash_cooldown=6,
              description="Sem canhão: casco que fere ao encostar e "
                          "investida de 3 casas."),
)
CLASS_INDEX = {c.key: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

TIER1 = ("duplo", "atirador", "metralhadora", "flanco")
CHILDREN = {
    "duplo": ("triplo", "quadruplo", "duplo_flanco"),
    "atirador": ("assassino", "cacador"),
    "metralhadora": ("destruidor", "artilheiro"),
    "flanco": ("tri_angulo", "quadruplo", "duplo_flanco"),
}
MAX_CLASS_OPTIONS = 5  # Básico no nível 10: 4 ramos + Esmagador


def class_options(cls_key: str, level: int) -> tuple[str, ...]:
    """Classes para as quais o tanque pode evoluir agora."""
    if cls_key == "basico":
        if level < EVOLVE_LEVELS[0]:
            return ()
        opts = TIER1
        if level >= EVOLVE_LEVELS[1]:
            opts = opts + ("esmagador",)
        return opts
    if cls_key in CHILDREN and level >= EVOLVE_LEVELS[1]:
        return CHILDREN[cls_key]
    return ()


# --------------------------------------------------------------------- #
# Estatísticas derivadas
# --------------------------------------------------------------------- #
BASE_HP = 10
BASE_RANGE = 8


def max_hp(cls: TankClass, points) -> int:
    return BASE_HP + 2 * int(points[0]) + cls.hp_bonus


def regen_interval(points) -> int:
    return REGEN_INTERVAL[int(points[1])]


def bullet_damage(cls: TankClass, points) -> float:
    return 1.0 + 0.5 * int(points[2])


def reload_interval(cls: TankClass, points) -> int:
    if cls.reload_fixed is not None:
        return int(cls.reload_fixed)
    base = 4.0 - 0.4 * int(points[3])          # 4 → 2
    return max(1, int(round(base * cls.reload_mult)))


def bullet_range(cls: TankClass, points) -> int:
    return BASE_RANGE + 2 * int(points[4]) + cls.range_bonus


# Ordem de investimento do upgrade automático (índices de ATTRIBUTES): os 14
# pontos dos níveis 2..15. Atributo já no máximo é pulado.
DEFAULT_BUILD: tuple[int, ...] = (2, 3, 0, 4, 2, 3, 0, 1, 2, 3, 4, 0, 1, 4)
