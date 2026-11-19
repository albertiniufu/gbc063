# Simulação da variância de cada estimador do gradiente (Aula 6, slide "A variância de cada estimador ao longo do treino").
# Uso (a partir de aula06/sim):  python3 simulacao.py  &&  python3 figuras.py ..
# Gera out.json, lido por figuras.py para escrever ../fig-variancia-treino.tex e ../fig-variancia.tex.
import numpy as np, json
rng = np.random.default_rng(3)
N, T, gam, lam = 8, 40, 0.99, 0.95
def step(s, a):
    s2 = max(0, s + (1 if a == 1 else -1))
    if s2 == N - 1: return s2, 40.0 + rng.normal(0, 2), True
    return s2, 1.0 + rng.normal(0, 2), False
th = np.zeros((N, 2)); V = np.zeros(N)
names = ['R', 'G', 'GV', 'GAE', 'TD']; hist = {k: [] for k in names}; ret = []
for it in range(150):
    eps = []
    for _ in range(48):
        s, tr = 0, []
        for t in range(T):
            p = np.exp(th[s] - th[s].max()); p /= p.sum()
            a = rng.choice(2, p=p); s2, r, d = step(s, a)
            tr.append((s, a, r, s2, d, p)); s = s2
            if d: break
        eps.append(tr)
    grads = {k: [] for k in names}; Gs = []
    for tr in eps:
        rs = np.array([x[2] for x in tr]); L = len(rs)
        G = np.zeros(L); acc = 0
        for t in reversed(range(L)): acc = rs[t] + gam * acc; G[t] = acc
        Rtau = G[0]; Gs.append(Rtau)
        v = np.array([V[x[0]] for x in tr]); v2 = np.array([0.0 if x[4] else V[x[3]] for x in tr])
        delta = rs + gam * v2 - v
        gae = np.zeros(L); acc = 0
        for t in reversed(range(L)): acc = delta[t] + gam * lam * acc; gae[t] = acc
        W = {'R': np.full(L, Rtau), 'G': G, 'GV': G - v, 'GAE': gae, 'TD': delta}
        for k in names:
            g = np.zeros_like(th)
            for t, (s, a, r, s2, d, p) in enumerate(tr):
                sc = -p.copy(); sc[a] += 1; g[s] += sc * W[k][t]
            grads[k].append(g.ravel())
        for t, x in enumerate(tr): V[x[0]] += 0.002 * (G[t] - V[x[0]])
    for k in names:
        A = np.array(grads[k]); hist[k].append(float(((A - A.mean(0)) ** 2).sum(1).mean()))
    th += 0.0005 * np.array(grads['GAE']).mean(0).reshape(th.shape)
    ret.append(float(np.mean(Gs)))
json.dump({'hist': hist, 'ret': ret}, open('out.json', 'w'))
for k in names: print(k, [round(np.log10(np.mean(hist[k][i:i+10])), 2) for i in range(0, 150, 30)])
print('ret', [round(np.mean(ret[i:i+10]), 1) for i in range(0, 150, 30)])
