"""Configuração da arena tankgrid.

Os campos ficam no checkpoint (`config`), e `evaluate`/`play` reconstroem o
ambiente a partir deles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TankGridConfig:
    # Partida
    seed: int = 0
    team_size: int = 3
    max_steps: int = 800
    hold_to_win: int = 100          # passos seguidos com as 3 torres
    respawn_delay: int = 8
    n_squares: int = 40
    n_triangles: int = 16
    n_pentagons: int = 6
    n_invaders: int = 4
    shape_physics: str = "space"    # "space": formas à deriva com inércia,
                                    # quiques e órbitas; "static": paradas

    # Ambiente (ponto de vista do time azul, o do agente)
    control: str = "team"           # "team": a política comanda os 3
                                    # tanques; "single": só o tanque 0
    opponent: str = "heuristic"     # heuristic | random | idle
    ally_bot: str = "heuristic"     # bot dos aliados em control="single"
    auto_upgrade: bool = True       # True: atributos e classes automáticos
    view_size: int = 15             # janela local (ímpar)

    # Recompensa (pesos); a soma é a recompensa do time a cada passo
    reward_score: float = 0.01      # (pontos nossos − pontos deles) do passo
    reward_win: float = 1.0         # +1 vitória, −1 derrota, 0 empate
    reward_capture: float = 0.1     # por torre capturada (− se perdida)
    reward_xp: float = 0.0          # por ponto de XP ganho pelo time
    reward_kill: float = 0.0        # por abate de tanque inimigo
    reward_death: float = 0.0       # por tanque nosso abatido (use < 0)
    reward_damage: float = 0.0      # por ponto de dano causado em tanques
    reward_approach: float = 1.0    # PBRS: aproximação das torres que não
                                    # são nossas (não muda a política ótima;
                                    # use 0 para a recompensa "crua")

    def to_dict(self) -> dict:
        return asdict(self)
