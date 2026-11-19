# Gera as figuras TikZ/pgfplots da Aula 6 a partir de out.json (ver simulacao.py).
# Uso (a partir de aula06/sim):  python3 figuras.py ..
import json, numpy as np, sys
out = sys.argv[1]
# ---- figura de variância (mesma semente/geração do slide)
seed = 7
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
def gauss():
    u = max(rnd(), 1e-9); v = rnd()
    return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
W, H = 800, 480; O = (170, 330); T = (430, 200)
def pts(sigma): return [(T[0] + sigma * gauss(), T[1] + sigma * gauss()) for _ in range(70)]
A = pts(290); B = pts(45)
s = 0.0078
def c(p): return f'({p[0]*s:.3f},{(H-p[1])*s:.3f})'
g = np.array([T[0]-O[0], T[1]-O[1]]); gn = np.linalg.norm(g); n = np.array([-g[1], g[0]])/gn
lines = ['% Gerado por figs.py (mesma semente do deck). Painéis de variância alta e baixa.']
def panel(P, col, title):
    L = []
    L.append(r'\begin{tikzpicture}')
    L.append(rf'\clip (0,0) rectangle ({W*s:.3f},{H*s:.3f});')
    L.append(rf'\fill[rlFundo] (0,0) rectangle ({W*s:.3f},{H*s:.3f});')
    q1 = np.array(O) + n*2000; q2 = np.array(O) - n*2000; b = -g/gn*2000
    poly = [q1, q2, q2+b, q1+b]
    L.append(r'\fill[rlLaranja!14] ' + ' -- '.join(c(p) for p in poly) + ' -- cycle;')
    L.append(rf'\draw[rlLaranja, dashed] {c(q1)} -- {c(q2)};')
    for p in P[:30]: L.append(rf'\draw[{col}, opacity=0.25] {c(O)} -- {c(p)};')
    for p in P: L.append(rf'\fill[{col}, opacity=0.75] {c(p)} circle (1.3pt);')
    L.append(rf'\draw[-{{Stealth[length=6pt]}}, line width=1.6pt, rlAzul] {c(O)} -- {c(T)};')
    L.append(rf'\fill[rlAzul] {c(O)} circle (1.8pt);')
    L.append(rf'\node[anchor=north west, font=\scriptsize\bfseries, text={col}] at (0.08,{H*s-0.06:.3f}) {{{title}}};')
    L.append(rf'\node[font=\scriptsize, text=rlAzul] at ({330*s:.3f},{(H-305)*s:.3f}) {{$\nabla J(\theta)$}};')
    L.append(rf'\node[anchor=south west, font=\tiny, text=rlLaranja] at (0.05,0.03) {{região onde o passo piora o objetivo}};')
    L.append(rf'\draw[rlCinza!40] (0,0) rectangle ({W*s:.3f},{H*s:.3f});')
    L.append(r'\end{tikzpicture}')
    return L
lines += panel(A, 'rlLaranja', 'Variância alta') + [r'\hfill'] + panel(B, 'rlAzulClaro', 'Variância baixa')
open(f'{out}/fig-variancia.tex', 'w').write('\n'.join(lines) + '\n')
# ---- variância por estimador ao longo do treino
d = json.load(open('out.json'))['hist']
names = [('R', 'rlLaranja', r'$R(\tau)$'), ('G', 'rlAmbar', r'$G_t$'), ('GV', 'rlAzulClaro!55', r'$G_t - V(s_t)$'),
         ('GAE', 'rlAzulClaro', r'GAE, $\lambda=0{,}95$'), ('TD', 'rlAzul', r'TD, $\delta_t$')]
L = ['% Gerado por figs.py a partir da simulação descrita no anexo (média móvel de 5 iterações).',
     r'\begin{tikzpicture}', r'\begin{axis}[width=0.66\textwidth, height=0.5\textheight, ymode=log,',
     r'  xmin=0, xmax=146, ymin=60, ymax=60000, xlabel={iteração de treino}, ylabel={variância do estimador},',
     r'  tick label style={font=\scriptsize}, label style={font=\scriptsize}, axis lines=left,',
     r'  ymajorgrids, grid style={rlCinza!20}, legend pos=outer north east, legend style={font=\scriptsize, draw=none, fill=none},',
     r'  legend cell align=left, every axis plot/.append style={line width=1.4pt, no marks}]']
for k, col, lab in names:
    v = np.convolve(np.array(d[k]), np.ones(5)/5, mode='valid')
    idx = list(range(0, len(v), 4)) + [len(v)-1]
    co = ' '.join(f'({i},{v[i]:.0f})' for i in idx)
    L.append(rf'\addplot[{col}] coordinates {{{co}}};')
    L.append(rf'\addlegendentry{{{lab}}}')
L += [r'\end{axis}', r'\end{tikzpicture}']
open(f'{out}/fig-variancia-treino.tex', 'w').write('\n'.join(L) + '\n')
print('ok')
