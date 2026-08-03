# GBC063 · S1 · O que é Inteligência Artificial?

**Prof. Marcelo Keese Albertini · FACOM · UFU**

[Material de estudo →](../semana01/material_introducao_ia.html) · [Deck →](../semana01/deck_introducao_ia.html)

---

**A IA funciona agora porque, depois de 70 anos tentando programar inteligência à mão, o campo aprendeu que métodos gerais que aprendem de dados e escalam com computação vencem — sempre.**

---

## Conceitos-chave

- **Agente racional:** sistema que percebe o ambiente e seleciona ações que maximizam uma métrica de desempenho esperada. É a definição operacional do curso inteiro.
- **IA simbólica:** paradigma baseado em regras e lógica escritas à mão. Funciona em domínios fechados; não escala.
- **Aprendizado de máquina:** paradigma baseado em extrair padrões de dados. Escala com computação e volume de dados.
- **The Bitter Lesson (Sutton, 2019):** métodos gerais que alavancam computação batem conhecimento artesanal — repetidamente e por larga margem. É o fato empírico mais robusto da história da IA.
- **Loop percepção → decisão → ação:** `obs → agente.decidir(obs) → env.step(acao) → obs'`. Esta interface de 4 linhas aparece em toda aula da Semana 2 à 16.

## A equação

```
obs = env.reset()
while not done:
    acao = agente.decidir(obs)
    obs, recompensa, done = env.step(acao)
```
*obs: observação · acao: ação selecionada · recompensa: sinal de reforço (aparece na Semana 10) · done: fim do episódio* · [derivação completa →](../semana01/material_introducao_ia.html#3)

## Código-síntese

```python
def agente_aspirador(percepcao):
    posicao, sujo = percepcao
    if sujo:        return "ASPIRAR"    # a decisão
    if posicao == "A": return "DIREITA"
    return "ESQUERDA"
```

## ⚠️ Armadilhas

- Confundir "imitar comportamento humano" com "agir racionalmente" — o curso adota a segunda. Um agente racional pode agir de forma não-humana e ainda assim ser ótimo (ex.: AlphaGo).
- Achar que ELIZA "entendia" — era casamento de padrões sem semântica. O mesmo efeito persiste hoje com LLMs: fluência não implica compreensão.
- Subestimar a Bitter Lesson — todo método que depende de conhecimento artesanal atinge um platô. Isso não é opinião; é o registro histórico de 70 anos.

## Você deve ser capaz de

- Explicar a diferença entre IA simbólica (programar regras) e aprendizado de máquina (extrair padrões de dados) em uma frase.
- Classificar qualquer sistema de IA em uma das quatro visões (pensar/agir × humana/racional) e justificar.
- Enunciar a Bitter Lesson de Sutton e dar dois exemplos históricos que a corroboram.
- Implementar o agente aspirador e prever seu comportamento com 3 cômodos ou sensor ruidoso.
- 🌉 Reconhecer o loop `percepção → decisão → ação` como a interface que o AlphaGo, o ChatGPT e o agente aspirador compartilham — o que muda é apenas a complexidade de `decidir()`.
