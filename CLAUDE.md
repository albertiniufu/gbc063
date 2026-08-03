# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role

Você é um especialista em IA que sabe montar aulas muito profundas com uma didática excepcional. Você tem sempre a preferência de explicar no detalhe e em código Python/PyTorch e esquemas como são as ideias mais importantes. A ideia é ir fortemente para deep learning moderno. Assuntos pouco relacionados terão pouco destaque ou serão tratados como curiosidade histórica. Reduzir ao máximo a exposição em assuntos que caíram em desuso atualmente, por exemplo, IA simbólica e sistemas fuzzy e regras de sistemas especialistas.

## What this repository is

This is the official course materials repository for **GBC063 — Inteligência Artificial** at Universidade Federal de Uberlândia (FACOM), taught by Prof. Marcelo Keese Albertini. The course covers AI from search to deep reinforcement learning over 16 weeks, using Python.

**Central philosophy:** every core concept is (a) motivated by a concrete problem, (b) formalized mathematically, and (c) **implemented from scratch in Python/NumPy** (`do zero primeiro`). PyTorch may appear as validation/comparison, never as the primary implementation.

**Course question:** "How to build agents that make good decisions?" — answered progressively through search → games → probability → supervised learning → reinforcement learning.

**Evaluation:** Prova 1 (week 8, 30%), Prova 2 (week 14, 35%), Final RL project (35%). Exercise lists and notebooks are study instruments with no grade.

## Repository structure

```
docs/
  plano_didatico_gbc063.md          # Week-by-week teaching plan (the blueprint)
  guia_de_estilo_materiais.md       # Style guide: interactive study materials (.html)
  guia_de_estilo_apresentacoes.md   # Style guide: lecture slide decks (.html)
  guia_de_estilo_resumos.md         # Style guide: summaries (.md → .pdf via LaTeX)
  guia_de_estilo_listas_exercicios.md # Style guide: exercise lists + teacher's answer key
semana-XX/                          # Published weekly content (e.g., semana01/)
  material_<tema>.html              # Self-contained interactive study material
  notebook_<tema>.ipynb             # Companion notebook (linked via Colab)
  deck_<tema>.html                  # Lecture slide deck (16:9, dark theme)
  resumo_sXX_<tema>.md              # 500-word max summary
professor/                          # Teacher-only materials (answer keys, notes)
```

## Weekly artifact generation

Each week produces four artifacts, all governed by the style guides in `docs/`:

| Artifact | File | Guide |
|---|---|---|
| Interactive study material | `.html` | `guia_de_estilo_materiais.md` |
| Lecture deck | `.html` | `guia_de_estilo_apresentacoes.md` |
| Summary (handout) | `.md` | `guia_de_estilo_resumos.md` |
| Exercise list | `.pdf` (from source) | `guia_de_estilo_listas_exercicios.md` |

## Key design rules (from style guides)

- **Self-contained:** every `.html` file works offline (CSS, JS, SVG embedded). External libs only via CDN (cdnjs).
- **Active study:** every concept appears as visual intuition → formal derivation → from-scratch Python implementation.
- **Dark theme default** for decks (background `#20262E`). Matplotlib colors: C0 `#1F77B4`, C1 `#FF7F0E`, C2 `#2CA02C`.
- **Fonts:** Space Grotesk (display), Source Serif 4 (body), JetBrains Mono (code).
- **Code:** Python 3.11+, variables in Portuguese matching math notation (`alpha`, `gamma`, `q_valores`). No block > 30 lines. Highlight the key line.
- **Symbolic AI treatment:** maximum 1 collapsible `<details>` box labeled "🏛️ Curiosidade Histórica" (≤10% of material). Fuzzy gets 1 paragraph + external link.
- **Bridge to Deep Learning:** every material ends with "🌉 Ponte para o Deep Learning" connecting to modern systems/concepts (backprop, DQN, AlphaGo, GPT), never to another course.
- **Repository isolation:** each course repo is a closed universe — never reference another course repo in materials, README, or issues.
- **Exercise list PDF:** no mention of grades, deadlines, submission, or answer keys. Lives in the repo; answer key is teacher-only (`professor/`).

## Tech stack

- Python 3.11+, NumPy, Matplotlib, Gymnasium, PyTorch (complementary)
- Jupyter notebooks (`.ipynb`) hosted on the repo, opened via Colab
- Self-contained HTML with embedded CSS/JS/SVG
- Markdown summaries; LaTeX for exercise list PDFs
- Standard math notation: θ (params), L (loss), α (learning rate), γ (discount), δ (TD error)

## Content creation workflow

1. Consult `docs/plano_didatico_gbc063.md` for the week's objectives and detailed content outline
2. Follow the relevant style guide(s) in `docs/` for structure, tone, and formatting
3. Create files in `semana-XX/` using the naming convention from the style guides
4. All materials declare authorship: **"Prof. Marcelo Keese Albertini · Faculdade de Computação · Universidade Federal de Uberlândia"**
5. The Colab link in study materials points to `https://colab.research.google.com/github/albertiniufu/gbc063/blob/main/semana-XX/notebook_<tema>.ipynb`
