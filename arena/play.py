"""Assista o seu agente jogar contra um bot de referência.

    python play.py --weights checkpoints/policy.pt
    python play.py --weights checkpoints/policy.pt --opponent random
    python play.py --weights checkpoints/policy.pt --seed 3 --fps 12
    python play.py --weights checkpoints/policy.pt --save-replay p.npz
    python play.py --replay p.npz          # assiste de novo (sem pesos)

Feche a janela ou aperte Esc para sair; espaço pausa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from tankgrid import TankGridConfig, TankGridGame
from tankgrid.agents import PolicyBot, make_bot
from tankgrid.renderer import TankGridRenderer
from tankgrid.replay import Recorder, replay_game


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
    parser.add_argument("--save-replay", metavar="ARQUIVO",
                        help="grava a partida para assistir depois")
    parser.add_argument("--replay", metavar="ARQUIVO",
                        help="assiste a um replay gravado (ignora --weights)")
    return parser.parse_args(argv)


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Window:
    """Janela pygame com Esc para sair e espaço para pausar."""

    def __init__(self, title: str, fps: int):
        self.renderer = TankGridRenderer()
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.renderer.width, self.renderer.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.paused = False

    def show(self, game) -> bool:
        """Desenha `game`; segura enquanto pausado. False = fechar."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE):
                    return False
                if event.type == pygame.KEYDOWN and \
                        event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            self.renderer.draw(game)
            self.screen.blit(self.renderer.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.fps)
            if not self.paused:
                return True


def print_summary(game) -> None:
    summary = game.summary()
    winner = summary["winner"]
    if winner is None:
        print("partida interrompida")
    elif winner == -1:
        print("empate")
    else:
        print(f"{'azul' if winner == 0 else 'vermelho'} vence "
              f"({summary['win_reason']})")
    print(f"placar: {summary['score'][0]}-{summary['score'][1]} | "
          f"passos: {summary['step']}")


def watch_replay(args) -> None:
    window = Window(f"GBC063 Arena — replay {Path(args.replay).name}",
                    args.fps)
    game = None
    try:
        for game in replay_game(args.replay):
            if not window.show(game):
                break
    finally:
        pygame.quit()
    if game is not None:
        print_summary(game)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.replay:
        watch_replay(args)
        return
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"pesos não encontrados em {weights}. Treine primeiro:\n"
            "  python train.py --episodes 300")

    config = TankGridConfig(seed=args.seed, opponent=args.opponent,
                            max_steps=800, control="team",
                            auto_upgrade=True, view_size=15)
    from agent import Agent             # só aqui: o replay não precisa
    agent = Agent(str(weights), device=resolve_device(args.device))
    game = TankGridGame(config, seed=args.seed)
    blue = PolicyBot(agent, view_size=15)
    red = make_bot(args.opponent, seed=args.seed + 1, view_size=15)
    blue.reset(args.seed + 1)
    red.reset(args.seed + 2)
    for tank in game.tanks:
        tank.auto = True

    recorder = Recorder(game) if args.save_replay else None
    window = Window(f"GBC063 Arena — seu agente × {args.opponent}", args.fps)
    try:
        while window.show(game) and not game.done:
            actions = {}
            for tank in game.tanks:
                if not tank.alive:
                    continue
                bot = blue if tank.team == 0 else red
                actions[tank.id] = bot.act(game, tank)
            game.step(actions)
    finally:
        pygame.quit()
        if recorder is not None:
            print(f"replay salvo em {recorder.save(args.save_replay)}")

    print_summary(game)


if __name__ == "__main__":
    main()
