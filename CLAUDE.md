# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Visão geral

Materiais do curso **GBC063 — Inteligência Artificial** (UFU/FACOM), com núcleo em aprendizagem por reforço. Todo o conteúdo é em pt-BR. O site é publicado pelo GitHub Pages direto da branch `main` do repositório `albertiniufu/gbc063`, em <https://albertiniufu.github.io/gbc063/>.

O `README.md` é o índice do curso: uma linha por aula, com links para slides, handout, simuladores e atividades (por assunto). Ao adicionar ou renomear material, atualize a tabela.

O conteúdo programático atual vai até a **aula 7**. As aulas 8–14 ficam em `conteudo_extra/aulaNN/`, fora do conteúdo atual, e continuam compiláveis pelos mesmos comandos.

## Comandos

```bash
make all                          # compila todas as aulas (slides + handouts)
make aula05                       # compila aula05/aula05.pdf e aula05/handout-aula05.pdf
make -B aula05                    # força recompilação
make aula05/aula05-sucinta.pdf    # a versão sucinta NÃO entra em `make aulaNN`
make clean                        # apaga temporários LaTeX (*.aux, *.log, *.nav, ...)
```

- A compilação é sempre **XeLaTeX**: `latexmk -cd -xelatex -interaction=nonstopmode`. O `-cd` faz o PDF e os temporários ficarem em `aulaNN/`.
- Arquivo avulso: `latexmk -cd -xelatex aula01/nota-bellman.tex` (ou `cd aula01 && latexmk -xelatex nota-bellman.tex`).
- O Makefile descobre as aulas por `wildcard aula*` (curso) e `wildcard conteudo_extra/aula*` (extras): criar `aula15/aula15.tex` e `aula15/handout-aula15.tex` — ou `conteudo_extra/aula15/...` — já cria o alvo `make aula15`.
- Não há testes, lint nem CI. A verificação é compilar o PDF e abrir o HTML no navegador.

## Estrutura dos materiais

Cada `aulaNN/` contém até quatro famílias de material:

| Arquivo | Saída | Papel |
|---|---|---|
| `aulaNN.tex` | `aulaNN.pdf` | slides Beamer 16:9; inclui também organização da disciplina (ementa, calendário, avaliação) |
| `aulaNN-sucinta.tex` | `aulaNN-sucinta.pdf` | roteiro do apresentador: mesmos frames/definições/figuras do `aulaNN.tex`, texto enxuto |
| `handout-aulaNN.tex` | `handout-aulaNN.pdf` | resumo A4 (`article`), apenas conteúdo didático |
| `*.html` | — | simulador/laboratório interativo, arquivo único |

**`aulaNN-sucinta.tex` é mantido em paralelo a `aulaNN.tex`** — não é gerado automaticamente. Ao mudar slides, verifique se a versão sucinta precisa do mesmo ajuste (e vice-versa). Todo `.tex` que usa os `.sty` da raiz inclui `\input@path{{./}{../}}` antes do preâmbulo do tema (nas aulas de `conteudo_extra/`, `\input@path{{./}{../}{../../}}`); mantenha esse boilerplate em qualquer `.tex` novo.

### Estilos compartilhados (raiz)

- `beamerthemeAprendizadoRL.sty` — tema Beamer: paleta `rlAzul`/`rlTeal`/`rlLaranja`/`rlVerde`/..., fontes Fira Sans/Fira Mono (fallback TeX Gyre Heros/Latin Modern), caixas `defbox`, `teobox`, `algbox`, `ideiabox`, `alertabox`, `exbox`, `quizbox`, `respbox`.
- `handoutAprendizadoRL.sty` — mesmo visual para `article`: `\setaula{}`, `\setprof{}`, `\capa{título}{subtítulo}{autoria}{crédito}` e as mesmas caixas.
- Exceções legadas: `aula01/handout-aula01.tex` e `aula01/nota-bellman.tex` definem cores e estilo inline; não usam o pacote compartilhado.

### Regra de conteúdo

- **Handouts**: apenas conteúdo didático (definições, exemplos, exercícios, leituras). Nunca ementa, calendário, avaliação ou logística.
- **Slides**: organização da disciplina pertence aqui, não ao handout.
- `docs/` e `professor/` são diretórios privados do professor, ignorados pelo git; não os versione.

### HTML interativo

- Cada simulador é um HTML único com CSS/JS inline — sem build e sem bundler. As únicas dependências externas são Google Fonts.
- Use caminhos relativos: o site é servido direto do repositório pelo GitHub Pages. Os links do `README.md` usam URLs absolutas de `https://albertiniufu.github.io/gbc063/...`.
- Siga a paleta do curso (azul `#16324F`, teal `#18808D`, laranja `#E8590C`, âmbar `#F2A93B`, vermelho `#C1121F`, verde `#2B9348`, cinza `#5C677D`) e Fira Sans/Fira Mono.
- `simuladores/` guarda simuladores transversais (Q-Learning, DQN); os laboratórios específicos ficam em `aulaNN/`.

### Notebooks (`atividades/`)

- Sequência numerada: `0_qlearning` → `1_deep_qlearning` → `2_reinforce` → `3_a2c` → `4_ppo`.
- Feitos para **Google Colab**: dependências instaladas na própria célula com `%pip install`; use a API do `gymnasium` ≥ 1.0 (`reset()` devolve `(obs, info)`, `step()` devolve 5 valores).
- `dqn.yml` é config do RL Baselines3 Zoo (SpaceInvaders).
- Artefatos de treino (`logs/`, `rl_trained/`, `replay*.mp4`) estão no `.gitignore`.

## Artefatos versionados e git

PDFs e HTMLs são **commitados** — são o material publicado e linkado no README. Depois de editar um `.tex`, recompile e commite o PDF junto. Mensagens de commit em português, voz presente ("adiciona ...", "corrige ...").

`.research-af3/` é um clone aninhado não versionado do AlphaFold 3 (repositório git próprio, sem relação com o curso) — não modifique nem commite.
