# GBC063 · Semana 2 · Resumo
## Agentes, ambientes e a caixa de ferramentas numérica

**Prof. Marcelo Keese Albertini · FACOM · Universidade Federal de Uberlândia**

---

### PEAS — anatomia do agente

- **P**erformance: medida de desempenho (recompensa acumulada)
- **E**nvironment: o mundo onde o agente opera (GridWorld 4×4)
- **A**ctuators: como age (mover ↑↓←→)
- **S**ensors: o que percebe (posição atual)

### O loop agente–ambiente (idêntico da S2 à S16)

```python
obs = env.reset()
while not done:
    acao = agente.decidir(obs)
    obs, recompensa, done = env.step(acao)
```

### 6 propriedades do ambiente (cada par → técnica do curso)

| Propriedade | Se for… | Técnica | Semana |
|---|---|---|---|
| Observável? | Parcialmente obs. | Estado interno | S2 |
| Determinístico? | Estocástico | Probabilidade | S7 |
| Episódico? | Sequencial | MDPs / RL | S10–14 |
| Estático? | Dinâmico | Anytime algorithms | S2 |
| Discreto? | Contínuo | Aproximação de função | S14 |
| Agente único? | Multiagente | Teoria dos jogos | S5 |

### 5 tipos de agente

1. **Reativo simples**: percepção → ação, sem memória (aspirador)
2. **Com estado interno**: mantém crença sobre o mundo
3. **Baseado em objetivo**: busca/planejamento (BFS, A*)
4. **Baseado em utilidade**: maximiza preferências
5. **Que aprende**: melhora com experiência (RL) ← destino do curso

### NumPy essencial

- `np.array`: operações elemento a elemento sem laços
- Broadcasting: operações entre arrays de dimensões diferentes
- `np.argmax`: coração da decisão do agente racional
- Vetorização: 500× mais rápido que laço `for` em Python puro

### GridWorld

Ambiente 4×4 com obstáculos, objetivo (⭐) e recompensa. Implementado como classe Python com `reset()` e `step()`. Este mesmo GridWorld será o ambiente das Semanas 10–14 para reinforcement learning.

---

*Próxima semana: Busca cega — BFS, DFS e custo uniforme.*
