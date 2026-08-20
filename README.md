# GBC063 — Inteligência Artificial

Repositório oficial de materiais da disciplina **GBC063 — Inteligência Artificial** da Universidade Federal de Uberlândia (FACOM), ministrada pelo Prof. Marcelo Keese Albertini. O curso cobre IA de busca até aprendizado por reforço profundo ao longo de 16 semanas, usando Python. Este repositório contém o módulo de Aprendizagem por Reforço (14 aulas) e os simuladores interativos que as acompanham.

**Pergunta do curso:** "Como construir agentes que tomam boas decisões?" — respondida progressivamente através de busca → jogos → probabilidade → aprendizado supervisionado → aprendizado por reforço.

## Simuladores

Simuladores interativos — rodam direto no navegador:

- [Q-Learning passo a passo — Sarsamax](https://albertiniufu.github.io/gbc063/simuladores/q-learning-simulador.html) — um agente aprende a atravessar uma grade 4×4 até o tesouro sem cair nos buracos; cada clique executa exatamente uma linha do pseudocódigo e você vê a tabela Q mudar por causa dela.
- [Deep Q-Learning, linha por linha](https://albertiniufu.github.io/gbc063/simuladores/dqn.html) — o pseudocódigo do DQN (Mnih et al., Nature 2015) executa de verdade: uma rede neural minúscula aprende a atravessar um labirinto 5×5 com paredes e poços, com mapa de calor e opção Double DQN.
- [Mars Rover — processo de recompensa de Markov](https://albertiniufu.github.io/gbc063/aula01/simulador-mars-rover.html) — a cadeia de estados do Mars Rover com função valor, equação de Bellman com os números de agora, velocidade de convergência e amostragem de episódios.
- [Laboratório interativo — MDPs, Iteração de Valor e de Política](https://albertiniufu.github.io/gbc063/aula02/interativo-aula02.html) — companheiro dos slides da Aula 2: manipule o MDP, acompanhe a anatomia do backup, a convergência e os retornos amostrais (ponte para a Aula 3).
- [Console do transecto — avaliação de política sem modelo](https://albertiniufu.github.io/gbc063/aula03/simulador-avaliacao.html) — três formas de avaliar a mesma política (Monte Carlo, diferença temporal e equivalência de certeza) sobre um transecto de 7 sítios de coleta, com RMSE por episódio.
- [Caderno de demonstrações da Aula 4](https://albertiniufu.github.io/gbc063/aula04/aula04-demos.html) — quatro coisas que só se veem rodando: a tríade mortal, o viés de maximização virando curva, o exemplo A–B editável e a pegada de uma única atualização.
- [Laboratório de gradiente de política](https://albertiniufu.github.io/gbc063/aula05/aula05-laboratorio-gradiente-de-politica.html) — a função escore empurrando a distribuição, onde mora a variância, REINFORCE aprendendo a ser aleatório e quanto viés vale a pena comprar em estimadores n-passos.
- [Bandits bayesianos e amostragem de Thompson](https://albertiniufu.github.io/gbc063/aula11/interativo-aula11.html) — três estações para ver a exploração acontecer: um sorteio por passo, quatro algoritmos em muitas execuções e o que acontece quando o prior mente.
- [MCTS ao vivo](https://albertiniufu.github.io/gbc063/aula13/lab13-mcts.html) — árvore sintética com ramificação e profundidade configuráveis, simulações UCT, árvore construída e estatísticas na raiz.
- [Busca em árvore Monte Carlo — ingredientes do AlphaZero](https://albertiniufu.github.io/gbc063/aula14/laboratorio-mcts.html) — um Lig-4 movido pelos mesmos ingredientes do AlphaZero (PUCT, rede política–valor), com todos eles expostos.

## Aulas

- [Aula 1 — Introdução à Aprendizagem por Reforço](aula01/aula01.pdf) — panorama da área, decisão sequencial sob incerteza e processos de Markov.
- [Aula 2 — Decisões com modelo do mundo](aula02/aula02.pdf) — processos de decisão de Markov, avaliação, iteração de política e de valor.
- [Aula 3 — Avaliação de política sem modelo](aula03/aula03.pdf) — Monte Carlo, diferença temporal e equivalência de certeza.
- [Aula 4 — Controle sem modelo e aproximação de função valor](aula04/aula04.pdf) — MC/TD em lote, ε-guloso e GLIE, SARSA e Q-learning, VFA e DQN.
- [Aula 5 — Gradientes de política I](aula05/aula05.pdf) — políticas parametrizadas, razão de verossimilhança, REINFORCE, linha de base e vantagem.
- [Aula 6 — Gradientes de política II: variância, confiança e PPO](aula06/aula06.pdf) — linha de base, ator–crítico e limites de desempenho.
- [Aula 7 — PPO, GAE e imitação](aula07/aula07.pdf) — amostragem por importância, melhoria monotônica, imitação e RL inverso.
- [Aula 8 — Imitação, aprendizado de recompensa e RLHF](aula08/aula08.pdf) — erros compostos, DAGGER, RL inverso, Bradley–Terry, RLHF e DPO.
- [Aula 9 — Bandidos e arrependimento](aula09/aula09.pdf) — bandidos multi-braços, arrependimento, ε-guloso, otimismo e UCB.
- [Aula 10 — Aprendizado rápido II](aula10/aula10.pdf) — prova da cota do UCB, PAC e bandits bayesianos, alinhamento de valores.
- [Aula 11 — Bandits bayesianos](aula11/aula11.pdf) — conjugação Beta–Bernoulli, casamento de probabilidade e arrependimento bayesiano.
- [Aula 12 — Aprendizagem rápida em MDPs](aula12/aula12.pdf) — garantias PAC, MBIE-EB, lema da simulação, PSRL e exploração com generalização.
- [Aula 13 — MCTS e a família AlphaZero](aula13/aula13.pdf) — busca baseada em simulação, UCT, AlphaGo, AlphaZero e MuZero.
- [Aula 14 — MCTS II: AlphaZero](aula14/aula14.pdf) — PUCT, auto-jogo, rede política–valor, AlphaTensor, AlphaDev e raciocínio metanível.
