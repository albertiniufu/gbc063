"""Agente do aluno — **modifique este arquivo**.

Contrato obrigatório do torneio (violar = risco de eliminação):

    class Agent:
        def __init__(self, weights_path: str, device: str = "cpu"): ...
        def reset(self, seed: int | None = None) -> None: ...   # opcional
        def act(self, obs: numpy.ndarray) -> int: ...

- `obs` é `float32` com forma `(41, 15, 15)` e valores em `[-1, 1]`.
- `act` devolve um inteiro de `0` a `80` (movimento × tiro; upgrade é
  automático no torneio).
- O agente não pode ler/escrever arquivos fora da própria pasta, abrir
  rede, criar subprocessos, alterar o jogo/runner nem usar informação
  privilegiada. Cada ação deve ser rápida (orçamento de tempo do torneio).
- `reset(seed)` será chamado no início de cada partida; use-o para semear
  seu gerador aleatório.
- O aluno **deve** alterar a rede e o algoritmo de treino. Este arquivo é
  só o ponto de partida; `train.py` mostra um REINFORCE sem baseline.

O ambiente e a observação são congelados: não mude `tankgrid/`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_CHANNELS = 41
VIEW_SIZE = 15
N_ACTIONS = 81


class PolicyNet(nn.Module):
    """Modelo mínimo de propósito: média espacial + uma camada linear.

    A rede ignora a posição dentro da janela 15 × 15: faz a média de cada
    um dos 41 canais e mapeia direto para as 81 ações. É fácil de ler e
    joga mal de propósito. Troque por uma CNN ou MLP que use a grade
    espacial para melhorar.
    """

    def __init__(self, n_actions: int = N_ACTIONS):
        super().__init__()
        self.fc = nn.Linear(N_CHANNELS, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs.mean(dim=(2, 3)))


class Agent:
    """Carrega `policy.pt` e joga uma partida."""

    on_policy = True  # o adaptador do torneio amostra em vez de argmax

    def __init__(self, weights_path: str = "checkpoints/policy.pt",
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.weights_path = str(weights_path)
        self.net = PolicyNet().to(self.device)
        state = torch.load(self.weights_path, map_location=self.device,
                           weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.net.load_state_dict(state)
        self.net.eval()
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(0)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._generator.manual_seed(int(seed))

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits = self.net(x).squeeze(0)
        if greedy:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1,
                                   generator=self._generator)
        return int(action.item())
