"""Assista o seu agente jogar contra um bot de referência.

    python play.py --weights checkpoints/policy.pt
    python play.py --weights checkpoints/policy.pt --opponent random
    python play.py --weights checkpoints/policy.pt --seed 3 --fps 12

Feche a janela ou aperte Esc para sair.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent import Agent
from tankgrid import TankGridConfig, TankGridGame
from tankgrid.agents import PolicyBot, make_bot
from tankgrid.renderer import TankGridRenderer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="checkpoints/policy.pt")
    parser.add_argument("--opponent", choices=("heuristic", "random", "idle"),
                        default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12,
                        help="passos do mundo por segundo")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv=None) -> None:
    args = parse_args(argv)
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"pesos não encontrados em {weights}. Treine primeiro:\n"
            "  python train.py --episodes 300")

    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "pygame não instalado; rode `pip install -r requirements.txt`"
        ) from exc

    config = TankGridConfig(seed=args.seed, opponent=args.opponent,
                            max_steps=800, control="team",
                            auto_upgrade=True, view_size=15)
    agent = Agent(str(weights), device=resolve_device(args.device))
    game = TankGridGame(config, seed=args.seed)
    blue = PolicyBot(agent, view_size=15)
    red = make_bot(args.opponent, seed=args.seed + 1, view_size=15)
    blue.reset(args.seed + 1)
    red.reset(args.seed + 2)
    for tank in game.tanks:
        tank.auto = True

    renderer = TankGridRenderer()
    pygame.init()
    screen = pygame.display.set_mode((renderer.width, renderer.height))
    pygame.display.set_caption(
        f"GBC063 Arena — seu agente × {args.opponent}")
    clock = pygame.time.Clock()

    running = True
    try:
        while running and not game.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and \
                        event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            actions = {}
            for tank in game.tanks:
                if not tank.alive:
                    continue
                bot = blue if tank.team == 0 else red
                actions[tank.id] = bot.act(game, tank)
            game.step(actions)

            renderer.draw(game)
            screen.blit(renderer.surface, (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        pygame.quit()

    summary = game.summary()
    winner = summary["winner"]
    if winner is None:
        print("partida interrompida")
    elif winner == -1:
        print("empate")
    else:
        print(f"{'azul (seu agente)' if winner == 0 else 'vermelho'} vence "
              f"({summary['win_reason']})")
    print(f"placar: {summary['score'][0]}-{summary['score'][1]} | "
          f"passos: {summary['step']}")


if __name__ == "__main__":
    main()
