# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

alfa = 0.05

def games_howell(gs, nomes):
    """Comparações múltiplas com a variância de cada grupo e gl de Welch-Satterthwaite."""
    k = len(gs)
    n = np.array([len(g) for g in gs], float)
    m = np.array([g.mean() for g in gs])
    v = np.array([g.var(ddof=1) for g in gs])
    linhas = []
    for a, b in itertools.combinations(range(k), 2):
        dif = m[b] - m[a]
        ep = np.sqrt((v[a]/n[a] + v[b]/n[b]) / 2)          # erro-padrão na escala de q
        glw = (v[a]/n[a] + v[b]/n[b])**2 / ((v[a]/n[a])**2/(n[a]-1) + (v[b]/n[b])**2/(n[b]-1))
        q = abs(dif) / ep
        pa = stats.studentized_range.sf(q, k, glw)
        marg = stats.studentized_range.ppf(1-alfa, k, glw) * ep
        linhas.append((f"{nomes[b]} - {nomes[a]}", dif, ep*np.sqrt(2), glw, q, pa, dif-marg, dif+marg))
    return linhas

def tukey(gs, nomes):
    """Tukey-Kramer: variância agrupada e gl do resíduo."""
    k = len(gs)
    n = np.array([len(g) for g in gs], float)
    m = np.array([g.mean() for g in gs])
    v = np.array([g.var(ddof=1) for g in gs])
    N = int(n.sum()); QM = np.sum((n-1)*v)/(N-k); gl = N-k
    linhas = []
    for a, b in itertools.combinations(range(k), 2):
        dif = m[b] - m[a]
        ep = np.sqrt(QM/2 * (1/n[a] + 1/n[b]))
        q = abs(dif)/ep
        marg = stats.studentized_range.ppf(1-alfa, k, gl)*ep
        linhas.append((f"{nomes[b]} - {nomes[a]}", dif, ep*np.sqrt(2), float(gl),
                       q, stats.studentized_range.sf(q, k, gl), dif-marg, dif+marg))
    return linhas

# ===================================================================
# PARTE 1 - onde Games-Howell faz diferença: variâncias muito desiguais
# Dados fictícios: resistência à compressão (MPa) de corpos de prova de concreto
# de quatro centrais dosadoras. É o conjunto da ficha da ANOVA de Welch.
A = np.array([57.0,54.0,51.5,31.1,55.9,32.9,60.2,57.5,59.5,59.4,54.7,47.2,49.2,65.6,46.8,49.9,45.5,47.3,49.2,54.3])
B = np.array([48.4,48.3,47.8,49.2,46.3,49.1,51.6,47.5,48.9,54.6])
C = np.array([50.5,48.9,51.9,57.5,58.1,50.0,54.4,53.0,51.5,48.1,55.6,50.8])
D = np.array([55.6,55.2,58.7,57.4,52.7,54.5,57.1,52.3,54.2,55.5])
gs = [A, B, C, D]; nm = ["A", "B", "C", "D"]
n = np.array([len(g) for g in gs], float)
v = np.array([g.var(ddof=1) for g in gs])
print("central   n    média     DP   variância")
for i, s in enumerate(nm):
    print(f"   {s}     {int(n[i]):2d}   {gs[i].mean():6.3f}  {np.sqrt(v[i]):5.3f}   {v[i]:8.3f}")
print(f"Razão entre a maior e a menor variância = {v.max()/v.min():.2f}")
print(f"Brown-Forsythe: W = {stats.levene(*gs, center='median')[0]:.4f}, "
      f"p-valor = {stats.levene(*gs, center='median')[1]:.5f}")
N = int(n.sum()); k = 4
QM = np.sum((n-1)*v)/(N-k)
print(f"\nQM do resíduo agrupado = {QM:.3f} (DP = {np.sqrt(QM):.3f} MPa)")
print("Esse número é a única régua que o Tukey usa. Compare com os DP de cada central:")
print("  ele é mais que o dobro do DP de B, C e D, e bem menor que o de A.")

gh = games_howell(gs, nm); tk = tukey(gs, nm)
print("\nTukey-Kramer (variância agrupada, 48 gl para todos os pares)")
print(f"{'comparação':<12}{'dif.':>9}{'EP':>8}{'gl':>8}{'q':>8}{'p ajust.':>11}{'IC 95%':>24}")
for nome, dif, ep, g_, q, pa, lo, hi in tk:
    print(f"{nome:<12}{dif:9.3f}{ep:8.3f}{g_:8.2f}{q:8.3f}{pa:11.5f}{'*' if pa<alfa else ' '}"
          f" [{lo:8.3f}; {hi:8.3f}]")
print("\nGames-Howell (variância de cada par, gl de Welch-Satterthwaite)")
print(f"{'comparação':<12}{'dif.':>9}{'EP':>8}{'gl':>8}{'q':>8}{'p ajust.':>11}{'IC 95%':>24}")
for nome, dif, ep, g_, q, pa, lo, hi in gh:
    print(f"{nome:<12}{dif:9.3f}{ep:8.3f}{g_:8.2f}{q:8.3f}{pa:11.5f}{'*' if pa<alfa else ' '}"
          f" [{lo:8.3f}; {hi:8.3f}]")
i_bd = [l[0] for l in tk].index("D - B")
print(f"\nO par B-D: Tukey usa EP = {tk[i_bd][2]:.3f} com {tk[i_bd][3]:.0f} gl; "
      f"Games-Howell usa EP = {gh[i_bd][2]:.3f} com {gh[i_bd][3]:.2f} gl.")
print(f"O erro-padrão cai {100*(1 - gh[i_bd][2]/tk[i_bd][2]):.1f}%, porque B e D são justamente as")
print("duas centrais mais homogêneas: a régua agrupada, inflada pela central A, era")
print(f"larga demais para elas. A diferença de {tk[i_bd][1]:.2f} MPa, que o Tukey não detecta")
print(f"(p = {tk[i_bd][5]:.4f}), passa a p = {gh[i_bd][5]:.5f}.")

# ===================================================================
# PARTE 2 - onde Games-Howell quase não muda nada: variâncias homogêneas
print("\n" + "=" * 70)
print("chickwts: seis rações, variâncias homogêneas (o conjunto da ficha de Tukey)")
dados = {
 "horsebean": [179,160,136,227,217,168,108,124,143,140],
 "linseed":   [309,229,181,141,260,203,148,169,213,257,244,271],
 "soybean":   [243,230,248,327,329,250,193,271,316,267,199,171,158,248],
 "sunflower": [423,340,392,339,341,226,320,295,334,322,297,318],
 "meatmeal":  [325,257,303,315,380,153,263,242,206,344,258],
 "casein":    [368,390,379,260,404,318,352,359,216,222,283,332]}
nomes2 = list(dados); gs2 = [np.array(v_, float) for v_ in dados.values()]
v2 = np.array([g.var(ddof=1) for g in gs2])
print(f"Razão entre variâncias = {v2.max()/v2.min():.2f} | "
      f"Brown-Forsythe p = {stats.levene(*gs2, center='median')[1]:.4f}")
gh2 = games_howell(gs2, nomes2); tk2 = tukey(gs2, nomes2)
print(f"\n{'comparação':<26}{'dif.':>9}{'Tukey':>11}{'Games-Howell':>15}{'gl de G-H':>12}")
for (nome, dif, _, _, _, pt, _, _), (_, _, _, glw, _, pg, _, _) in zip(tk2, gh2):
    print(f"{nome:<26}{dif:9.2f}{pt:11.5f}{'*' if pt<alfa else ' '}"
          f"{pg:14.5f}{'*' if pg<alfa else ' '}{glw:12.2f}")
ct, cg = sum(1 for l in tk2 if l[5] < alfa), sum(1 for l in gh2 if l[5] < alfa)
print(f"\nSignificativas: Tukey {ct} | Games-Howell {cg}. As conclusões coincidem.")
print("Com variâncias homogêneas, Games-Howell custa poder (os gl de cada par são")
print("menores que os 65 do resíduo agrupado), mas o prejuízo é pequeno.")

# ===================================================================
# PARTE 3 - taxa de erro do conjunto nos dois cenários
print("\n" + "=" * 70)
print("Simulação: 10.000 experimentos com QUATRO grupos de médias IGUAIS")
rng = np.random.default_rng(2026)
M = 10000
# O valor crítico q(5%, k = 4, gl) é liso em gl: tabela-se uma vez e interpola-se,
# em vez de recalcular a integral em cada uma das 240.000 comparações da simulação.
gl_tab = np.unique(np.round(np.geomspace(3, 400, 90), 3))
q_tab = np.array([stats.studentized_range.ppf(1-alfa, 4, g_) for g_ in gl_tab])
q_interp = lambda g_: np.interp(np.clip(g_, gl_tab[0], gl_tab[-1]), gl_tab, q_tab)
erro = max(abs(q_interp(g_) - stats.studentized_range.ppf(1-alfa, 4, g_))
           for g_ in (4.7, 9.3, 17.6, 33.1, 88.4))
print(f"(erro máximo da interpolação do valor crítico: {erro:.6f})")
for titulo, ns_, sds_ in [("variâncias iguais (n = 10, 12, 10, 12; DP = 3 em todos)",
                           [10, 12, 10, 12], [3., 3., 3., 3.]),
                          ("variâncias desiguais (n = 20, 10, 12, 10; DP = 9, 2, 2.5, 2)",
                           [20, 10, 12, 10], [9., 2., 2.5, 2.]),
                          ("pior caso: a maior variância no menor grupo (n = 20, 12, 10, 6)",
                           [20, 12, 10, 6], [2., 2.5, 2., 9.])]:
    kk = 4
    N_ = sum(ns_); glr = N_ - kk
    q_t = stats.studentized_range.ppf(1-alfa, kk, glr)
    nn_ = np.array(ns_, float)
    G = [rng.normal(50, s, (M, nnn)) for nnn, s in zip(ns_, sds_)]
    mm = np.column_stack([g.mean(axis=1) for g in G])          # M x 4
    vv = np.column_stack([g.var(axis=1, ddof=1) for g in G])
    QMr = (vv * (nn_ - 1)).sum(axis=1) / glr
    achou_t = np.zeros(M, bool); achou_g = np.zeros(M, bool)
    for a, b in itertools.combinations(range(kk), 2):
        d = np.abs(mm[:, b] - mm[:, a])
        achou_t |= d / np.sqrt(QMr/2 * (1/nn_[a] + 1/nn_[b])) > q_t
        sa, sb = vv[:, a]/nn_[a], vv[:, b]/nn_[b]
        epg = np.sqrt((sa + sb)/2)
        glw = (sa + sb)**2 / (sa**2/(nn_[a]-1) + sb**2/(nn_[b]-1))
        achou_g |= d/epg > q_interp(glw)
    print(f"\n  {titulo}")
    print(f"    Tukey-Kramer: {100*achou_t.mean():5.2f}%   Games-Howell: "
          f"{100*achou_g.mean():5.2f}%   (alvo: 5%)")
