# Regras do TankGrid

Resumo para jogar e treinar. A fonte de verdade é o motor em
`tankgrid/game.py`; se houver divergência, vale o código.

Dois times de 3 tanques numa arena de **48 × 30 casas**, espelhada entre a
esquerda (azul) e a direita (vermelho). Cada lado nasce na sua base, no
extremo do mapa. As torres ficam na coluna central.

## Partida

| Elemento | Regra |
| --- | --- |
| Duração | 800 passos do mundo |
| Pontuação | +1 ponto por torre controlada a cada passo; mais pontos vence |
| Vitória imediata | segurar as 3 torres por 100 passos seguidos |
| Empate | mesmo placar ao fim dos 800 passos |

## Tanques

| Elemento | Regra |
| --- | --- |
| Vida | 10 (pode subir com atributos) |
| Movimento | 1 casa por passo, em 8 direções |
| Tiro | 8 direções; a bala anda 2 casas por passo até o alcance |
| Morte | perde metade do XP; o abatedor (quem causou dano nos últimos 30 passos) ganha esse XP + 5; renasce na base após 8 passos |
| Classe | não se perde na morte; o nível cai junto com o XP |
| Upgrade | automático no torneio (atributos e classes escolhidos pelo jogo) |

## Formas e obstáculos

| Elemento | Vida | XP | Massa | Observação |
| --- | --- | --- | --- | --- |
| Quadrado | 1 | 2 | 1 | flutua |
| Triângulo | 3 | 5 | 2 | flutua |
| Pentágono | 10 | 20 | 5 | orbita o ninho central |
| Invasor | 3 | 8 | 1,5 | acelera rumo ao tanque a até 6 casas e explode (3 de dano) |
| Torre (2 × 2) | 15 | 25 ao capturar | — | atira (dano 1 a cada 3 passos) em inimigos a até 6 casas, na direção (das 8) mais próxima |
| Base | — | — | — | cura 1/passo aliados e tira 2/passo inimigos |
| Muro | — | — | — | bloqueia passagem e tiros |

Todo tiro sai da casa do tanque, inclusive o dos canos lado a lado
(Duplo, Duplo-flanco, Artilheiro): os dois tiros seguem a mesma linha.
Tiros atravessam aliados e torres do próprio time. A bala é bloqueada por
muros e formas; o tiro continua na grade, mesmo com a física contínua das
formas.

## Física das formas

Com `shape_physics="space"` (padrão do torneio):

- inércia sem atrito: nascem com deriva lenta e giram devagar;
- quiques perfeitos em muros, torres, tanques, bordas e bases;
- colisões elásticas entre formas (massas diferentes);
- tiros e tanques empurram as formas (o impulso é dividido pela massa);
- pentágonos orbitam o ninho; invasores têm propulsão;
- velocidade máxima 0,6 casa/passo.

Cada forma nasce espelhada com a do outro lado, então o mapa é simétrico.

## Níveis, atributos e classes

**XP por nível:** 0, 4, 10, 18, 28, 40, 54, 70, 88, 108, 130, 154, 180,
208, 238 (15 níveis). Cada nível dá 1 ponto de atributo; nos níveis 5 e 10
abre a evolução de classe.

| Atributo | Efeito por ponto |
| --- | --- |
| Vida máxima | +2 (10 → 20) |
| Regeneração | 1 de vida a cada 10 → 3 passos |
| Dano | +0,5 (1 → 3,5) |
| Recarga | intervalo de 4 → 2 passos: 4, 4, 3, 3, 2, 2 com 0 a 5 pontos (arredondado; o 1º e o 3º ponto não mudam o intervalo) |
| Alcance | +2 casas (8 → 18) |

No nível 5 o Básico escolhe um ramo; no nível 10, uma especialização.
Quem continua Básico até o 10 libera o Esmagador.

| Nível 5 | Nível 10 |
| --- | --- |
| Duplo | Triplo, Quádruplo, Duplo-flanco |
| Atirador | Assassino, Caçador |
| Metralhadora | Destruidor, Artilheiro |
| Flanco | Tri-Ângulo, Quádruplo, Duplo-flanco |
| Básico no 10 | Esmagador |

O torneio usa upgrade automático, então você não escolhe atributos nem
classes pela ação.

## Observação e ação do agente

- Observação: `(41, 15, 15)` — janela 15 × 15 centrada no tanque que vai
  decidir, 16 canais espaciais + 25 escalares. O vermelho é espelhado.
- Ação: `0..80`; `move = a // 9`, `shoot = a % 9`. `0` = parado/não atira;
  `1..8` = direções (E, NE, N, NO, O, SO, S, SE).
- `control="team"`: a política comanda os 3 tanques azuis com parâmetros
  compartilhados; cada `step` decide um tanque vivo e o mundo avança quando
  todos decidiram.

## Bots de referência

- `random`: anda e atira ao acaso.
- `heuristic`: farma formas, alinha a mira, recua com pouca vida, caça
  inimigos e, depois de crescer, toma as torres por BFS.
- `idle`: fica parado (útil em testes).
