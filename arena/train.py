"""Treino do aluno: REINFORCE sem baseline (template simples).

Exemplo:
    python train.py --episodes 300
    python train.py --episodes 100 --opponent random --device cpu
    python train.py --max-minutes 30 --out checkpoints/policy.pt

O agente comanda os 3 tanques azuis (`control="team"`), com upgrade
automático, contra o bot escolhido no vermelho. A observação é
`(41, 15, 15)` e a ação é discreta em `0..80`.

Isto é só um ponto de partida: **modifique a rede (`agent.py`) e o
algoritmo**. O contrato de execução do torneio está no topo de `agent.py`
e no README.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agent import N_ACTIONS, PolicyNet
from tankgrid import TankGridConfig
from tankgrid.env import TankGridEnv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opponent", choices=("heuristic", "random", "idle"),
                        default="heuristic")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda ou cuda:0")
    parser.add_argument("--out", default="checkpoints/policy.pt")
    parser.add_argument("--save-every", type=int, default=25,
                        help="0 desliga os checkpoints intermediários")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="0 = sem limite de tempo")
    return parser.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Retorno descontado de cada passo, sem baseline."""
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return torch.tensor(out, dtype=torch.float32)


def save_checkpoint(path: str, net: nn.Module, episode: int,
                    config: TankGridConfig, obs_shape, n_actions: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": net.state_dict(),
        "obs_shape": tuple(int(v) for v in obs_shape),
        "n_actions": int(n_actions),
        "episode": int(episode),
        "algorithm": "reinforce",
        "config": config.to_dict(),
    }
    torch.save(payload, target)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = TankGridConfig(
        seed=args.seed,
        opponent=args.opponent,
        max_steps=args.max_steps,
        control="team",
        auto_upgrade=True,
        view_size=15,
    )
    env = TankGridEnv(config=config)
    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)
    if tuple(obs_shape) != (41, 15, 15) or n_actions != N_ACTIONS:
        raise SystemExit(
            f"espaço inesperado: obs={obs_shape}, ações={n_actions}; "
            "o torneio usa (41,15,15) e 81 ações")

    net = PolicyNet(n_actions).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"device={device} obs={tuple(obs_shape)} ações={n_actions} "
          f"adversário={args.opponent} passos={args.max_steps}")
    print(f"torch={torch.__version__} numpy={np.__version__} "
          f"episódios={args.episodes}")
    print("REINFORCE sem baseline — ajuste a rede e o algoritmo à vontade.")

    started = time.time()
    recent_returns: list[float] = []
    recent_wins: list[int] = []
    episode = 0
    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + episode)
            log_probs = []
            entropies = []
            rewards = []
            done = False
            while not done:
                x = torch.as_tensor(obs, dtype=torch.float32,
                                    device=device).unsqueeze(0)
                logits = net(x).squeeze(0)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                obs, reward, terminated, truncated, info = env.step(
                    int(action.item()))
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                rewards.append(float(reward))
                done = bool(terminated or truncated)

            returns = discounted_returns(rewards, args.gamma).to(device)
            log_probs_t = torch.stack(log_probs)
            loss = -(log_probs_t * returns).mean()
            if args.entropy_coef:
                loss = loss - args.entropy_coef * torch.stack(entropies).mean()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               args.grad_clip)
            optimizer.step()

            total = float(sum(rewards))
            won = int(info.get("winner") == 0)
            draw = int(info.get("winner") == -1)
            score = info.get("score", (0, 0))
            recent_returns.append(total)
            recent_wins.append(won)
            window = recent_returns[-20:]
            win_window = recent_wins[-20:]

            if args.log_every and episode % args.log_every == 0:
                mean_ret = sum(window) / len(window)
                win_rate = 100.0 * sum(win_window) / len(win_window)
                print(f"ep {episode:5d} | retorno {total:9.2f} "
                      f"(média20 {mean_ret:9.2f}) | vitória {won} "
                      f"empate {draw} | placar {score[0]}-{score[1]} "
                      f"| passos {info.get('step', 0):3d} "
                      f"| loss {float(loss.detach()):9.3f} "
                      f"| win20 {win_rate:5.1f}%")

            if args.save_every and episode % args.save_every == 0:
                save_checkpoint(args.out, net, episode, config, obs_shape,
                                n_actions)

            if args.max_minutes and \
                    (time.time() - started) / 60.0 >= args.max_minutes:
                print(f"limite de {args.max_minutes:g} min atingido")
                break
    finally:
        env.close()
        save_checkpoint(args.out, net, episode, config, obs_shape, n_actions)

    elapsed = time.time() - started
    print(f"checkpoint salvo em {args.out}")
    print(f"tempo: {elapsed / 60.0:.1f} min | episódios: {episode}")
    print(f"para assistir: python play.py --weights {args.out}")


if __name__ == "__main__":
    main()
