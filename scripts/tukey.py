# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: chickwts (R). Peso (g) de 71 pintinhos aos 6 dias, segundo seis rações.
# O mesmo conjunto das fichas de Bonferroni e de Šidák, para comparação direta.
dados = {
 "horsebean": [179,160,136,227,217,168,108,124,143,140],
 "linseed":   [309,229,181,141,260,203,148,169,213,257,244,271],
 "soybean":   [243,230,248,327,329,250,193,271,316,267,199,171,158,248],
 "sunflower": [423,340,392,339,341,226,320,295,334,322,297,318],
 "meatmeal":  [325,257,303,315,380,153,263,242,206,344,258],
 "casein":    [368,390,379,260,404,318,352,359,216,222,283,332]}
nomes = list(dados)
gs = [np.array(v, float) for v in dados.values()]
k = len(gs); ns = np.array([len(g) for g in gs], float); N = int(ns.sum())
medias = np.array([g.mean() for g in gs])
QM = np.sum((ns - 1) * np.array([g.var(ddof=1) for g in gs])) / (N - k)
gl = N - k
alfa = 0.05
print("ração        n    média     DP      EP da média")
for i, nm in enumerate(nomes):
    print(f"{nm:<12}{int(ns[i]):2d}  {medias[i]:7.3f}  {gs[i].std(ddof=1):6.2f}    {np.sqrt(QM/ns[i]):.3f}")
print(f"\nANOVA: F({k-1}; {gl}) = {stats.f_oneway(*gs)[0]:.4f}, p = {stats.f_oneway(*gs)[1]:.4e}")
print(f"QM do resíduo = {QM:.3f} | DP residual = {np.sqrt(QM):.4f} g | n desiguais: {ns.astype(int)}")

# 1. A distribuição da amplitude estudentizada
print("\nDe onde vem o valor crítico: a AMPLITUDE entre k médias, não uma diferença qualquer")
print(f"q crítico (5%, k = {k}, gl = {gl}) = {stats.studentized_range.ppf(1-alfa, k, gl):.4f}")
print(" k médias   q crítico (5%)   equivalente na escala t (q/raiz(2))")
for kk in (2, 3, 4, 5, 6, 8, 10):
    q = stats.studentized_range.ppf(1-alfa, kk, gl)
    print(f"   {kk:3d}        {q:9.4f}          {q/np.sqrt(2):17.4f}")
print(f"Com k = 2 o valor na escala t é {stats.studentized_range.ppf(1-alfa,2,gl)/np.sqrt(2):.4f},")
print(f"exatamente o t bilateral de 5% com {gl} gl ({stats.t.ppf(1-alfa/2, gl):.4f}): sem múltiplas")
print("comparações, Tukey se reduz ao teste t comum.")

print("\nSimulação: distribuição da amplitude estudentizada com 6 médias e 65 gl")
rng = np.random.default_rng(5)
M = 200000
Z = rng.standard_normal((M, 6))
S = np.sqrt(rng.chisquare(gl, M) / gl)
amp = (Z.max(axis=1) - Z.min(axis=1)) / S
print(f"  percentil 95 simulado = {np.percentile(amp, 95):.4f} | teórico = "
      f"{stats.studentized_range.ppf(0.95, 6, gl):.4f}")

# 2. Tukey-Kramer: as 15 comparações
print(f"\nTukey-Kramer ({k*(k-1)//2} comparações; os n desiguais entram no erro-padrão)")
q95 = stats.studentized_range.ppf(1-alfa, k, gl)
linhas = []
for a, b in itertools.combinations(range(k), 2):
    dif = medias[b] - medias[a]
    ep = np.sqrt(QM/2 * (1/ns[a] + 1/ns[b]))      # erro-padrão na escala de q
    q = abs(dif) / ep
    pa = stats.studentized_range.sf(q, k, gl)
    marg = q95 * ep
    linhas.append((f"{nomes[b]} - {nomes[a]}", dif, q, pa, dif - marg, dif + marg))
print(f"{'comparação':<26}{'dif.':>9}{'q':>8}{'p ajust.':>11}{'IC 95% simultâneo':>26}")
for nm, dif, q, pa, lo, hi in sorted(linhas, key=lambda x: x[3]):
    print(f"{nm:<26}{dif:9.2f}{q:8.3f}{pa:11.5f}{'*' if pa<alfa else ' '}"
          f"  [{lo:8.2f}; {hi:8.2f}]")

# 3. Comparação com os demais ajustes
m = k*(k-1)//2
print("\nMultiplicadores críticos na escala t e largura do IC de sunflower - horsebean")
i = [l[0] for l in linhas].index("sunflower - horsebean")
ep_t = np.sqrt(QM*(1/ns[0] + 1/ns[3]))
dif = linhas[i][1]
for nome, cr in [("LSD (sem ajuste)", stats.t.ppf(1-alfa/2, gl)),
                 ("Tukey-Kramer", q95/np.sqrt(2)),
                 ("Šidák", stats.t.ppf(1 - (1-(1-alfa)**(1/m))/2, gl)),
                 ("Bonferroni", stats.t.ppf(1-alfa/(2*m), gl)),
                 ("Scheffé", np.sqrt((k-1)*stats.f.ppf(1-alfa, k-1, gl)))]:
    print(f"  {nome:<18}{cr:7.4f}   IC = [{dif-cr*ep_t:7.2f}; {dif+cr*ep_t:7.2f}]  "
          f"(largura {2*cr*ep_t:6.2f} g)")
print("Tukey é o mais curto dos ajustes para comparações pareadas; Scheffé, o mais longo,")
print("porque protege QUALQUER contraste, inclusive os decididos depois de ver os dados.")

# 4. Apresentação em letras
print("\nApresentação em letras (médias que compartilham letra não diferem a 5%)")
ordem = np.argsort(-medias)
sig = np.ones((k, k), bool)
for a, b in itertools.combinations(range(k), 2):
    dif = medias[b] - medias[a]
    pa = stats.studentized_range.sf(abs(dif)/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl)
    sig[a, b] = sig[b, a] = (pa >= alfa)         # True = NÃO diferem
# Grupos maximais de médias que NÃO diferem entre si (cliques do grafo de não significância)
cliques = []
for r in range(k, 0, -1):
    for comb in itertools.combinations(ordem, r):
        if all(sig[x, y] for x, y in itertools.combinations(comb, 2)):
            cj = set(comb)
            if not any(cj <= c for c in cliques):
                cliques.append(cj)
letras = [""] * k
for idx, c in enumerate(cliques):
    for j in c:
        letras[j] += chr(ord("a") + idx)
letras = ["".join(sorted(x)) for x in letras]
print("ração          média   letras")
for i in ordem:
    print(f"{nomes[i]:<12}{medias[i]:9.2f}   {letras[i]}")
print("Leitura: sunflower e casein (a) não diferem entre si; horsebean (c) é a única")
print("ração que difere de todas as demais.")

# 5. O erro de pular a ANOVA? Não: Tukey não precisa dela
print("\nTukey controla a taxa de erro do conjunto por conta própria:")
print("não é obrigatório que a ANOVA rejeite antes. O procedimento em dois passos")
print("(ANOVA e só depois Tukey) é ainda mais conservador que Tukey sozinho.")
print(f"Nestes dados a ANOVA rejeita com folga (p = {stats.f_oneway(*gs)[1]:.3e}), então a questão não se coloca.")

# 6. Simulação da taxa de erro do conjunto
print("\nSimulação: 10.000 experimentos com 6 grupos de médias IGUAIS (n = 12 cada)")
rng = np.random.default_rng(31)
M2 = 10000
glr = 6 * 11
q_crit = stats.studentized_range.ppf(1 - alfa, 6, glr)     # calculado UMA vez, fora do laço
t_crit = stats.t.ppf(1 - alfa/2, glr)
G = rng.normal(0, 1, (M2, 6, 12))
mm = G.mean(axis=2)                                        # médias: M2 x 6
qmr = G.var(axis=2, ddof=1).mean(axis=1)                   # QM do resíduo de cada experimento
ep = np.sqrt(qmr / 12)
difs = np.abs(mm[:, :, None] - mm[:, None, :])             # todas as diferenças
iu = np.triu_indices(6, 1)
t_obs = difs[:, iu[0], iu[1]] / (ep[:, None] * np.sqrt(2))
rej_lsd = int(np.sum((t_obs > t_crit).any(axis=1)))
rej_tuk = int(np.sum((mm.max(axis=1) - mm.min(axis=1)) / ep > q_crit))
print(f"  LSD (t sem ajuste): {100*rej_lsd/M2:.2f}% de experimentos com ao menos um falso positivo")
print(f"  Tukey-Kramer:       {100*rej_tuk/M2:.2f}%  (o alvo é 5%)")
