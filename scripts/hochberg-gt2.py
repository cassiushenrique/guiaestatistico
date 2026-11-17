# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, integrate
import itertools

alfa = 0.05

# A distribuição do módulo máximo estudentizado, que o procedimento usa.
# Não está no scipy: calcula-se por integração numérica.
def smm_cdf(h, m, nu):
    """P(máximo de |t_1|,...,|t_m| <= h), com t_i normais padronizadas por um mesmo s."""
    f = lambda s: (2*stats.norm.cdf(h*s) - 1)**m * stats.chi2.pdf(s*s*nu, nu) * 2*nu*s
    return integrate.quad(f, 0, np.inf, limit=200)[0]

def smm_sf(h, m, nu):
    return max(0.0, 1 - smm_cdf(h, m, nu))

def smm_ppf(p, m, nu):
    lo, hi = 0.5, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if smm_cdf(mid, m, nu) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

# Dados: chickwts (R). Peso (g) de 71 pintinhos aos 6 dias, segundo seis rações.
# É o mesmo conjunto das fichas de Bonferroni, Šidák, Tukey, REGW e Duncan.
dados = {
 "horsebean": [179,160,136,227,217,168,108,124,143,140],
 "linseed":   [309,229,181,141,260,203,148,169,213,257,244,271],
 "soybean":   [243,230,248,327,329,250,193,271,316,267,199,171,158,248],
 "sunflower": [423,340,392,339,341,226,320,295,334,322,297,318],
 "meatmeal":  [325,257,303,315,380,153,263,242,206,344,258],
 "casein":    [368,390,379,260,404,318,352,359,216,222,283,332]}
nomes = list(dados); gs = [np.array(v, float) for v in dados.values()]
k = len(gs); ns = np.array([len(g) for g in gs], float); N = int(ns.sum())
medias = np.array([g.mean() for g in gs]); vars_ = np.array([g.var(ddof=1) for g in gs])
QM = np.sum((ns - 1) * vars_) / (N - k); gl = N - k
m = k * (k - 1) // 2

print("ração        n    média     DP")
for i in np.argsort(-medias):
    print(f"{nomes[i]:<12}{int(ns[i]):2d}  {medias[i]:7.3f}  {np.sqrt(vars_[i]):6.2f}")
print(f"\nANOVA: F({k-1}; {gl}) = {stats.f_oneway(*gs)[0]:.4f}, p = {stats.f_oneway(*gs)[1]:.3e}")
print(f"QM do resíduo = {QM:.4f} | gl = {gl} | m = {m} comparações")
print(f"Tamanhos dos grupos: {ns.astype(int)} (desiguais)")

# 1. O valor crítico: o módulo máximo estudentizado
crit = smm_ppf(1 - alfa, m, gl)
print(f"\nMódulo máximo estudentizado: m({alfa}; {m}; {gl}) = {crit:.4f}")
print("Confira que a distribuição está correta: com m = 1 ele deve ser o t comum.")
print(f"  m(0,05; 1; {gl}) = {smm_ppf(1-alfa, 1, gl):.4f} | t(0,975; {gl}) = {stats.t.ppf(1-alfa/2, gl):.4f}")
print("\nComo o valor crítico cresce com o número de comparações")
print(" m    SMM     Šidák      Bonferroni   Tukey (q/raiz2, k correspondente)")
for mm, kk in [(1, 2), (3, 3), (6, 4), (10, 5), (15, 6), (28, 8), (45, 10)]:
    print(f" {mm:3d}  {smm_ppf(1-alfa, mm, gl):6.4f}  {stats.t.ppf(1-(1-(1-alfa)**(1/mm))/2, gl):9.4f}"
          f"  {stats.t.ppf(1-alfa/(2*mm), gl):12.4f}  {stats.studentized_range.ppf(1-alfa, kk, gl)/np.sqrt(2):12.4f}")
print("O SMM fica sempre entre Tukey e Šidák: é mais exato que Šidák, porque")
print("incorpora o fato de todas as comparações dividirem o mesmo QM do resíduo,")
print("mas ignora a dependência entre as médias, que é o que o Tukey aproveita.")

# 2. As comparações
print(f"\nHochberg GT2 ({m} comparações)")
print(f"{'comparação':<26}{'dif.':>9}{'EP':>8}{'t':>8}{'p ajust.':>11}{'IC 95%':>24}")
linhas = []
for a, b in itertools.combinations(range(k), 2):
    dif = medias[b] - medias[a]
    ep = np.sqrt(QM * (1/ns[a] + 1/ns[b]))
    t = abs(dif) / ep
    p = smm_sf(t, m, gl)
    linhas.append((f"{nomes[b]} - {nomes[a]}", dif, ep, t, p, dif - crit*ep, dif + crit*ep))
for nm_, dif, ep, t, p, lo, hi in sorted(linhas, key=lambda x: x[4]):
    print(f"{nm_:<26}{dif:9.2f}{ep:8.3f}{t:8.3f}{p:11.5f}{'*' if p<alfa else ' '}"
          f"  [{lo:8.2f}; {hi:8.2f}]")
print(f"\nSignificativas: {sum(1 for l in linhas if l[4] < alfa)} de {m}")

# 3. Comparação com os demais procedimentos
print("\nMultiplicadores críticos e largura do IC de sunflower - horsebean")
i = [l[0] for l in linhas].index("sunflower - horsebean")
dif, ep = linhas[i][1], linhas[i][2]
for nome, c in [("LSD (sem ajuste)", stats.t.ppf(1-alfa/2, gl)),
                ("Tukey-Kramer", stats.studentized_range.ppf(1-alfa, k, gl)/np.sqrt(2)),
                ("Hochberg GT2 (SMM)", crit),
                ("Šidák", stats.t.ppf(1-(1-(1-alfa)**(1/m))/2, gl)),
                ("Bonferroni", stats.t.ppf(1-alfa/(2*m), gl)),
                ("Scheffé", np.sqrt((k-1)*stats.f.ppf(1-alfa, k-1, gl)))]:
    print(f"  {nome:<20}{c:7.4f}   IC = [{dif-c*ep:7.2f}; {dif+c*ep:7.2f}]  (largura {2*c*ep:6.2f} g)")

tuk = [stats.studentized_range.sf(abs(medias[b]-medias[a])/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl)
       for a, b in itertools.combinations(range(k), 2)]
bon = [min(1, m*2*stats.t.sf(abs(medias[b]-medias[a])/np.sqrt(QM*(1/ns[a]+1/ns[b])), gl))
       for a, b in itertools.combinations(range(k), 2)]
print(f"\nSignificativas: Tukey {sum(p<alfa for p in tuk)} | GT2 {sum(1 for l in linhas if l[4]<alfa)}"
      f" | Bonferroni {sum(p<alfa for p in bon)}")

# 4. Onde o GT2 foi pensado para ajudar: n muito desiguais
print("\nSimulação: taxa de erro do conjunto com 4 grupos de médias IGUAIS")
print("(10.000 experimentos; variâncias iguais, tamanhos muito diferentes)")
rng = np.random.default_rng(404)
M = 10000
mm4 = 6
for titulo, ns4 in [("balanceado: 12, 12, 12, 12", [12, 12, 12, 12]),
                    ("moderado:   20, 12, 10, 8", [20, 12, 10, 8]),
                    ("extremo:    40, 12, 6, 5", [40, 12, 6, 5])]:
    nn = np.array(ns4, float); Nt = int(nn.sum()); glr = Nt - 4
    q_t = stats.studentized_range.ppf(1-alfa, 4, glr)
    c_g = smm_ppf(1-alfa, mm4, glr)
    c_b = stats.t.ppf(1-alfa/(2*mm4), glr)
    rej_t = rej_g = rej_b = 0
    for _ in range(M):
        g4 = [rng.normal(0, 1, int(x)) for x in ns4]
        mu = np.array([g.mean() for g in g4]); vv = np.array([g.var(ddof=1) for g in g4])
        qmr = np.sum((nn-1)*vv)/glr
        at = ag = ab = False
        for a, b in itertools.combinations(range(4), 2):
            d = abs(mu[b]-mu[a])
            if d/np.sqrt(qmr/2*(1/nn[a]+1/nn[b])) > q_t: at = True
            tt = d/np.sqrt(qmr*(1/nn[a]+1/nn[b]))
            if tt > c_g: ag = True
            if tt > c_b: ab = True
        rej_t += at; rej_g += ag; rej_b += ab
    print(f"  {titulo}:  Tukey-Kramer {100*rej_t/M:5.2f}%  GT2 {100*rej_g/M:5.2f}%  "
          f"Bonferroni {100*rej_b/M:5.2f}%")
print("\nOs três controlam a taxa de erro em todos os cenários, mas o resultado")
print("contraria a motivação histórica do GT2: Tukey-Kramer fica mais perto de 5%")
print("em TODOS eles, inclusive no desbalanceamento extremo, enquanto o GT2 gasta")
print("menos de 4%. O GT2 foi proposto em 1974, quando ainda não se sabia se o")
print("Tukey-Kramer controlava a taxa de erro com n desiguais; Hayter provou em")
print("1984 que sim. Desde então o GT2 deixou de ter vantagem nas comparações")
print("pareadas, e sua utilidade está em outro lugar: ele vale para QUALQUER")
print("conjunto de comparações que dividam o mesmo QM do resíduo, inclusive")
print("conjuntos em que o Tukey não se aplica.")
