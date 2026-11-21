# GBC063 Arena — TankGrid

## O que é o TankGrid

Dois times de 3 tanques numa arena 48 × 30. Os tanques destroem formas para
ganhar XP, sobem de nível, evoluem de classe e disputam 3 torres no centro.
Ganha quem tiver mais pontos ao fim de 800 passos ou quem segurar as 3
torres por 100 passos seguidos. As regras completas estão em
[RULES.md](RULES.md). Antes de entregar, confira o
[checklist de submissão](#checklist-de-submissão).

### Orientações 

Pacote independente para treinar e submeter um agente na arena **TankGrid**.
Ele contém o motor da arena, um template de **REINFORCE sem baseline** e os
scripts `train.py`, `play.py` e `submit.py`.

O objetivo é treinar um agente que jogue bem o TankGrid e submetê-lo ao
torneio. **Você deve modificar o agente (`agent.py`) e o algoritmo de
treino.** O template é só o ponto de partida.

> **Modelo inicial mínimo:** a rede é só a média espacial de cada canal
> + uma camada linear. Ela ignora a posição na janela 15 × 15 e joga mal
> de propósito; melhorar a arquitetura e o treino é parte do desafio.

> **Contrato de execução:** o torneio carrega o seu `agent.py` e os seus
> pesos. Seguir o contrato é obrigatório; violações podem eliminar o time.
> Leia a seção [Contrato de execução](#contrato-de-execução) antes de
> começar.


## Instalação

Use **Python 3.13.18** e as versões exatas de
[requirements.txt](requirements.txt):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, numpy, gymnasium, pygame; print(torch.__version__, numpy.__version__, gymnasium.__version__, pygame.version.ver)"
```

Versões obrigatórias:

| Pacote | Versão |
| --- | --- |
| Python | 3.13.18 |
| torch | 2.15.0 |
| numpy | 2.5.4 |
| gymnasium | 1.4.0 |
| pygame | 2.6.1 |

Se você usa GPU, instale o build CUDA do torch **na mesma versão 2.15.0**.
A inferência do torneio roda em CPU; o treino pode usar GPU.

## Começando

```bash
cp equipe.example.json equipe.json   # preencha nome do time e integrantes
python train.py --episodes 300       # adversário padrão: heuristic
python play.py --weights checkpoints/policy.pt
python submit.py                     # valida e gera gbc063_submission.zip
```

Para um primeiro teste mais fácil:

```bash
python train.py --episodes 200 --opponent random --device cpu
```

### Arquivos

| Arquivo | Papel |
| --- | --- |
| `agent.py` | **seu agente**: modelo mínimo (média + linear) + `Agent.act(obs)`. Modifique. |
| `train.py` | template REINFORCE sem baseline. Modifique à vontade. |
| `play.py` | assiste seu agente contra um bot de referência. |
| `submit.py` | valida o contrato e gera o zip da submissão. |
| `equipe.json` | nome do time e integrantes (criado por você). |
| `tankgrid/` | motor da arena. **Não modifique.** |
| `RULES.md` | regras resumidas da arena. |

## Ambiente e interface

O agente comanda os **3 tanques azuis** com parâmetros compartilhados
(`control="team"`) e upgrade automático.

- **Observação:** `numpy.ndarray` `float32` com forma `(41, 15, 15)`. É uma
  janela 15 × 15 centrada no tanque que vai decidir, com 16 canais
  espaciais e 25 escalares (vida, nível, recarga, placar, torres,
  atributos etc.). O vermelho é espelhado: a mesma política joga dos dois
  lados.
- **Ação:** inteiro `0..80`. `move = a // 9` e `shoot = a % 9`; `0` é
  parado / não atira; `1..8` são as direções. O upgrade é automático.
- **Partida:** até 800 passos do mundo. Cada `env.step` decide um tanque;
  o mundo avança quando os 3 vivos decidiram.
- **Recompensa:** padrão do ambiente, com diferença de pontos, captura de
  torres e *potential-based shaping* em direção às torres
  (`reward_approach=1.0`). O template usa essa recompensa; você pode
  mudá-la no seu `train.py` (o torneio avalia vitória/placar).

O ambiente é um `gymnasium.Env`; exemplos de uso estão em `train.py`.

## Contrato de execução

A validação oficial é feita **de novo** para toda submissão. O `submit.py`
só antecipa problemas. Uma violação na validação ou durante o evento pode
eliminar o time, com o motivo registrado.

### Entrega

- Um arquivo chamado **`gbc063_submission.zip`** na **raiz** do repositório
  GitHub do time.
- Dentro do zip, tudo no nível raiz (layout plano):
  - `agent.py` — obrigatório;
  - outros `.py` auxiliares do agente, se precisar;
  - `policy.pt` — pesos gerados por `train.py` (o `submit.py` renomeia);
  - `equipe.json` — nome do time e integrantes.

Não inclua `tankgrid/`, `.venv/`, `checkpoints/`, `train.py`, `play.py` ou
`submit.py` no zip; o `submit.py` já monta o pacote correto.

### `equipe.json`

```json
{
  "time": "Nome do Time",
  "integrantes": [
    {"nome": "Nome Completo", "matricula": "123456"},
    {"nome": "Outro Integrante", "matricula": "654321"}
  ],
  "alteracoes": "Descreva o que você mudou em agent.py e no algoritmo de treino."
}
```

- `time` e `integrantes` são obrigatórios.
- `alteracoes` é obrigatório: a disciplina exige que você modifique o
  agente e o algoritmo; entregar o template intacto não é aceito.

### Interface obrigatória

```python
class Agent:
    def __init__(self, weights_path: str, device: str = "cpu"): ...
    def reset(self, seed: int | None = None) -> None: ...   # opcional
    def act(self, obs: numpy.ndarray) -> int: ...
```

- `act` recebe apenas a observação e devolve um `int` de `0` a `80`.
- `reset(seed)` é chamado no início de cada partida; use para semear seu
  RNG. Se você devolver ações estocásticas sem semear, os resultados podem
  variar entre partidas.
- Carregue os pesos **do caminho recebido** (`weights_path`); não hardcode
  `checkpoints/policy.pt`, porque o torneio extrai o zip em outra pasta.
- O `act` deve ser rápido. Cada ação tem um orçamento de tempo e cada
  partida tem um timeout; estourar = W.O.

### Proibições

O agente não pode:

- ler ou escrever arquivos fora da própria pasta;
- abrir rede, criar subprocessos ou instalar pacotes;
- alterar o jogo, o ambiente, o processo de avaliação ou o sistema de
  arquivos;
- usar informação privilegiada (objeto do jogo, RNG do jogo ou monkeypatch
  do processo de avaliação);
- devolver ação fora de `0..80` ou um tipo que não seja inteiro.

O código do aluno roda em um subprocesso isolado, com timeout e limites de
CPU/memória. Nenhuma dependência do repositório do time é instalada; use
somente o que está em `requirements.txt`.

### Checklist de submissão

- [ ] Modifiquei `agent.py` e o algoritmo de treino.
- [ ] `equipe.json` tem o nome do time e **todos** os integrantes.
- [ ] `python submit.py` passa sem erros (versões, agente, zip).
- [ ] `gbc063_submission.zip` está na raiz do repositório e foi commitado.
- [ ] O repositório é **público** no GitHub e foi enviado ao professor.
- [ ] Não modifiquei `tankgrid/`.

## Como a submissão é avaliada

O time publica `gbc063_submission.zip` na raiz de um repositório público. A
avaliação baixa o repositório, encontra o zip, valida o agente e roda uma
**qualificação offline** (mini-Swiss de 3 rodadas) para semear o evento.

O evento principal é um **Swiss de 4 rodadas**:

- 3 pontos por vitória, 1 por empate, 0 por derrota.
- Desempate: saldo de placar, confronto direto, Buchholz e seed de
  qualificação.
- O número de times pode variar; o Swiss pareia por grupos de pontuação,
  evita revanches e dá **bye** quando o número é ímpar.
- Bots de referência (`heuristic` e `random`) também participam do
  ranking.

Falhas de contrato, exceções e timeouts viram W.O. e podem eliminar o
time.

## Dicas de treino

- O template é **REINFORCE puro**, sem baseline: alta variância. Teste
  `--opponent random` primeiro para confirmar que o pipeline aprende algo.
- O adversário padrão `heuristic` é forte; use `--max-minutes` para
  orçamentos de treino e `--save-every` para não perder progresso.
- O modelo inicial é intencionalmente fraco: ele ignora a grade espacial.
  Troque por uma CNN/MLP que use a janela 15 × 15 e ajuste `PolicyNet` em
  `agent.py`.
- O upgrade é automático no torneio; não tente controlar atributos/classe
  pela ação (as ações `81..90` não existem no torneio).
- Avalie com `play.py` contra `random` e `heuristic` em várias seeds.
- Não use bibliotecas fora de `requirements.txt`: apenas as dependências
  do projeto estão disponíveis.

## Testes do motor

`tests/` tem testes das regras e do ambiente (sem tela e sem torch):

```bash
pip install pytest
pytest tests
```

## Pacote auto-contido

Este pacote contém a arena (`tankgrid/`), o template do agente
(`agent.py`), o treino (`train.py`), o `play.py` e o `submit.py`. Não
depende de arquivos ou projetos externos. O motor em `tankgrid/` é a
referência da arena; não o modifique.
