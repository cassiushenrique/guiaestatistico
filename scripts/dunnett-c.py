# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

alfa = 0.05

# Dados fictícios: resistência à compressão (MPa) de corpos de prova de concreto
# de quatro centrais dosadoras. É o conjunto das fichas da ANOVA de Welch e de
# Games-Howell, usado aqui para comparar os dois procedimentos diretamente.
A = np.array([57.0,54.0,51.5,31.1,55.9,32.9,60.2,57.5,59.5,59.4,54.7,47.2,49.2,65.6,46.8,49.9,45.5,47.3,49.2,54.3])
B = np.array([48.4,48.3,47.8,49.2,46.3,49.1,51.6,47.5,48.9,54.6])
C = np.array([50.5,48.9,51.9,57.5,58.1,50.0,54.4,53.0,51.5,48.1,55.6,50.8])
D = np.array([55.6,55.2,58.7,57.4,52.7,54.5,57.1,52.3,54.2,55.5])
gs = [A, B, C, D]; nomes = ["A", "B", "C", "D"]
k = len(gs); ns = np.array([len(g) for g in gs], float); N = int(ns.sum())
medias = np.array([g.mean() for g in gs]); vars_ = np.array([g.var(ddof=1) for g in gs])
QM = np.sum((ns - 1) * vars_) / (N - k); gl = N - k

print("central   n    média     DP   variância   s²/n")
for i in range(k):
    print(f"   {nomes[i]}     {int(ns[i]):2d}   {medias[i]:6.3f}  {np.sqrt(vars_[i]):5.3f}   "
          f"{vars_[i]:8.3f}  {vars_[i]/ns[i]:6.3f}")
print(f"Razão entre a maior e a menor variância = {vars_.max()/vars_.min():.2f}")
print(f"Brown-Forsythe: W = {stats.levene(*gs, center='median')[0]:.4f}, "
      f"p = {stats.levene(*gs, center='median')[1]:.5f}")
print(f"QM do resíduo agrupado = {QM:.3f} (DP = {np.sqrt(QM):.3f} MPa) | gl = {gl}")

# 1. O teste global compatível
w = ns / vars_; mu_w = np.sum(w*medias)/np.sum(w)
lam = np.sum((1 - w/w.sum())**2/(ns-1))
Fw = (np.sum(w*(medias-mu_w)**2)/(k-1)) / (1 + 2*(k-2)/(k**2-1)*lam)
glw = (k**2-1)/(3*lam)
print(f"\nANOVA de Welch: F({k-1}; {glw:.3f}) = {Fw:.4f}, p = {stats.f.sf(Fw, k-1, glw):.3e}")
print(f"ANOVA clássica: F({k-1}; {gl}) = {stats.f_oneway(*gs)[0]:.4f}, "
      f"p = {stats.f_oneway(*gs)[1]:.5f}")

# 2. O valor crítico de Dunnett C: média ponderada de dois valores de q
print("\nDunnett C: o valor crítico de cada par é uma média ponderada")
print("de q(k, n_i - 1) e q(k, n_j - 1), com pesos s_i²/n_i e s_j²/n_j.")
print(f"\nq(5%, k = {k}, gl) para cada central, com gl = n - 1:")
for i in range(k):
    print(f"  central {nomes[i]} (gl = {int(ns[i]-1)}): q = "
          f"{stats.studentized_range.ppf(1-alfa, k, ns[i]-1):.4f}")

def dunnett_c(i, j):
    si, sj = vars_[i]/ns[i], vars_[j]/ns[j]
    qi = stats.studentized_range.ppf(1-alfa, k, ns[i]-1)
    qj = stats.studentized_range.ppf(1-alfa, k, ns[j]-1)
    qc = (qi*si + qj*sj) / (si + sj)              # valor crítico ponderado
    ep = np.sqrt((si + sj)/2)                     # erro-padrão na escala de q
    return medias[j]-medias[i], ep, qc, abs(medias[j]-medias[i])/ep

def games_howell(i, j):
    si, sj = vars_[i]/ns[i], vars_[j]/ns[j]
    glw_ = (si+sj)**2 / (si**2/(ns[i]-1) + sj**2/(ns[j]-1))
    ep = np.sqrt((si+sj)/2)
    q = abs(medias[j]-medias[i])/ep
    return medias[j]-medias[i], ep, glw_, q, stats.studentized_range.sf(q, k, glw_)

print(f"\nDunnett C ({k*(k-1)//2} comparações)")
print(f"{'par':<8}{'diferença':>11}{'EP':>8}{'q obs.':>9}{'q crítico':>11}{'decisão':>12}{'IC 95%':>22}")
res_dc = []
for i, j in itertools.combinations(range(k), 2):
    dif, ep, qc, q = dunnett_c(i, j)
    marg = qc*ep
    sig = q > qc
    res_dc.append((f"{nomes[j]} - {nomes[i]}", dif, ep, q, qc, sig, dif-marg, dif+marg))
    print(f"{nomes[j]+' - '+nomes[i]:<8}{dif:11.3f}{ep:8.3f}{q:9.3f}{qc:11.3f}"
          f"{('  difere *' if sig else '  não difere'):>12}  [{dif-marg:7.3f}; {dif+marg:7.3f}]")

print(f"\nGames-Howell, para comparação")
print(f"{'par':<8}{'diferença':>11}{'EP':>8}{'gl':>9}{'q obs.':>9}{'p ajust.':>11}")
for i, j in itertools.combinations(range(k), 2):
    dif, ep, glw_, q, p = games_howell(i, j)
    print(f"{nomes[j]+' - '+nomes[i]:<8}{dif:11.3f}{ep:8.3f}{glw_:9.2f}{q:9.3f}{p:11.5f}"
          f"{'*' if p<alfa else ' '}")

print("\nTukey-Kramer, que supõe variâncias iguais")
q_t = stats.studentized_range.ppf(1-alfa, k, gl)
for i, j in itertools.combinations(range(k), 2):
    dif = medias[j]-medias[i]; ep = np.sqrt(QM/2*(1/ns[i]+1/ns[j]))
    p = stats.studentized_range.sf(abs(dif)/ep, k, gl)
    print(f"{nomes[j]+' - '+nomes[i]:<8}{dif:11.3f}{ep:8.3f}{gl:9d}{abs(dif)/ep:9.3f}{p:11.5f}"
          f"{'*' if p<alfa else ' '}")

# 3. Simulação
print("\nSimulação: taxa de erro do conjunto com 4 grupos de médias IGUAIS")
print("(10.000 experimentos em cada cenário)")
rng = np.random.default_rng(777)
M = 10000
cenarios = [("variâncias iguais (n = 10, 12, 10, 12; DP = 3)", [10,12,10,12], [3.,3.,3.,3.]),
            ("maior variância no maior grupo (n = 20, 10, 12, 10)", [20,10,12,10], [9.,2.,2.5,2.]),
            ("maior variância no menor grupo (n = 20, 12, 10, 6)", [20,12,10,6], [2.,2.5,2.,9.])]
# valores críticos de q por gl inteiro, tabelados uma vez (ptukey é lento)
gl_tab = np.unique(np.round(np.geomspace(3, 400, 90), 3))
q_tab = np.array([stats.studentized_range.ppf(1-alfa, 4, g_) for g_ in gl_tab])
qi = lambda g_: np.interp(np.clip(g_, gl_tab[0], gl_tab[-1]), gl_tab, q_tab)
print(f"(valores críticos interpolados; erro máximo = "
      f"{max(abs(qi(g_)-stats.studentized_range.ppf(1-alfa,4,g_)) for g_ in (4.7,9.3,17.6,33.1,88.4)):.6f})")
for titulo, ns_, sd_ in cenarios:
    nn = np.array(ns_, float); Nt = int(nn.sum()); glr = Nt - 4
    q_tk = stats.studentized_range.ppf(1-alfa, 4, glr)
    q_ind = np.array([qi(x-1) for x in nn])        # q(k, n_i - 1), fixo no cenário
    G = [rng.normal(50, s, (M, int(n_))) for n_, s in zip(ns_, sd_)]
    mu = np.column_stack([g.mean(axis=1) for g in G])
    vv = np.column_stack([g.var(axis=1, ddof=1) for g in G])
    qmr = (vv*(nn-1)).sum(axis=1)/glr
    a_tk = np.zeros(M, bool); a_gh = np.zeros(M, bool); a_dc = np.zeros(M, bool)
    for i, j in itertools.combinations(range(4), 2):
        d = np.abs(mu[:, j]-mu[:, i])
        a_tk |= d/np.sqrt(qmr/2*(1/nn[i]+1/nn[j])) > q_tk
        si, sj = vv[:, i]/nn[i], vv[:, j]/nn[j]
        ep = np.sqrt((si+sj)/2)
        glw_ = (si+sj)**2/(si**2/(nn[i]-1) + sj**2/(nn[j]-1))
        a_gh |= d/ep > qi(glw_)
        a_dc |= d/ep > (q_ind[i]*si + q_ind[j]*sj)/(si+sj)
    print(f"\n  {titulo}")
    print(f"    Tukey-Kramer {100*a_tk.mean():5.2f}%   Games-Howell {100*a_gh.mean():5.2f}%   "
          f"Dunnett C {100*a_dc.mean():5.2f}%   (alvo: 5%)")
