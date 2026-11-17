# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: chickwts (R). Peso (g) de 71 pintinhos aos 6 dias, segundo seis rações.
# Fonte original: Anonymous (1948), Biometrika 35, p. 214; ver McNeil (1977).
dados = {
 "horsebean": [179,160,136,227,217,168,108,124,143,140],
 "linseed":   [309,229,181,141,260,203,148,169,213,257,244,271],
 "soybean":   [243,230,248,327,329,250,193,271,316,267,199,171,158,248],
 "sunflower": [423,340,392,339,341,226,320,295,334,322,297,318],
 "meatmeal":  [325,257,303,315,380,153,263,242,206,344,258],
 "casein":    [368,390,379,260,404,318,352,359,216,222,283,332]}
nomes = list(dados)
gs = [np.array(v, float) for v in dados.values()]
k = len(gs)
ns = np.array([len(g) for g in gs], float)
N = int(ns.sum())
medias = np.array([g.mean() for g in gs])

print("ração        n    média     DP")
for i, nm in enumerate(nomes):
    print(f"{nm:<12}{int(ns[i]):2d}  {medias[i]:7.3f}  {gs[i].std(ddof=1):6.2f}")
F, pF = stats.f_oneway(*gs)
QM = np.sum((ns - 1) * np.array([g.var(ddof=1) for g in gs])) / (N - k)
gl = N - k
print(f"\nANOVA: F({k-1}; {gl}) = {F:.4f}, p-valor = {pF:.4e}")
print(f"QM do resíduo = {QM:.3f} | DP residual = {np.sqrt(QM):.3f} g")
print(f"Shapiro-Wilk dos resíduos: p = {stats.shapiro(np.concatenate([g-g.mean() for g in gs]))[1]:.4f}")
print(f"Brown-Forsythe: p = {stats.levene(*gs, center='median')[1]:.4f}")

# 1. O problema que o ajuste resolve
m = k * (k - 1) // 2
print(f"\nCom {k} grupos há m = {m} comparações duas a duas.")
print("Se cada uma fosse testada a 5%, a chance de ao menos um falso positivo seria")
print(f"  1 - 0,95^{m} = {1 - 0.95**m:.4f}, ou seja {100*(1-0.95**m):.1f}%, e não 5%.")
print("\nComo essa taxa cresce com o número de comparações:")
print(" k grupos   comparações   1 - 0,95^m")
for kk in (2, 3, 4, 5, 6, 8, 10, 20):
    mm = kk*(kk-1)//2
    print(f"   {kk:3d}        {mm:4d}          {1-0.95**mm:.4f}")

# 2. Simulação: a taxa de erro do conjunto, com e sem ajuste
print("\nSimulação: 10.000 experimentos com SEIS grupos de médias IGUAIS (n = 12 cada).")
rng = np.random.default_rng(2026)
M = 10000
falso_sem = falso_bonf = falso_holm = 0
pares = list(itertools.combinations(range(6), 2))
for _ in range(M):
    g6 = [rng.normal(0, 1, 12) for _ in range(6)]
    pv = np.array([stats.ttest_ind(g6[a], g6[b]).pvalue for a, b in pares])
    if (pv < 0.05).any(): falso_sem += 1
    if (np.minimum(1, len(pv)*pv) < 0.05).any(): falso_bonf += 1
    ordem = np.argsort(pv); holm = np.empty(len(pv)); ac = 0.0
    for pos, idx in enumerate(ordem):
        ac = max(ac, (len(pv) - pos) * pv[idx]); holm[idx] = min(1.0, ac)
    if (holm < 0.05).any(): falso_holm += 1
print(f"  sem ajuste:  {100*falso_sem/M:.2f}% dos experimentos tiveram ao menos um falso positivo")
print(f"  Bonferroni:  {100*falso_bonf/M:.2f}%")
print(f"  Holm:        {100*falso_holm/M:.2f}%")
print("\n  O alvo é 5%. Sem ajuste, 36,5% dos experimentos produzem ao menos um falso")
print("  positivo - bem acima de 5%, mas ABAIXO dos 53,7% da conta anterior. A conta")
print("  1 - 0,95^15 supõe testes independentes, e as 15 comparações não são: cada")
print("  grupo entra em cinco delas. Essa dependência é o que torna o Bonferroni")
print("  conservador aqui: ele fica em 3,82%, gastando menos que os 5% disponíveis.")

# 3. As 15 comparações dos dados reais
print(f"\nAs {m} comparações (teste t com variância agrupada, {gl} gl)")
brutos = []; difs = []; rotulos = []
for a, b in itertools.combinations(range(k), 2):
    dif = medias[b] - medias[a]
    ep = np.sqrt(QM * (1/ns[a] + 1/ns[b]))
    t = dif / ep
    brutos.append(2 * stats.t.sf(abs(t), gl)); difs.append(dif); rotulos.append(f"{nomes[b]} - {nomes[a]}")
brutos = np.array(brutos); difs = np.array(difs)
bonf = np.minimum(1, m * brutos)
ordem = np.argsort(brutos); holm = np.empty(m); ac = 0.0
for pos, idx in enumerate(ordem):
    ac = max(ac, (m - pos) * brutos[idx]); holm[idx] = min(1.0, ac)
tuk = np.array([stats.studentized_range.sf(abs(d)/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl)
                for d, (a, b) in zip(difs, itertools.combinations(range(k), 2))])
print(f"{'comparação':<26}{'dif.':>9}{'p bruto':>11}{'Bonferroni':>12}{'Holm':>11}{'Tukey':>11}")
for i in np.argsort(brutos):
    marca = lambda v: "*" if v < 0.05 else " "
    print(f"{rotulos[i]:<26}{difs[i]:9.2f}{brutos[i]:11.5f}{bonf[i]:11.5f}{marca(bonf[i])}"
          f"{holm[i]:10.5f}{marca(holm[i])}{tuk[i]:10.5f}{marca(tuk[i])}")
print(f"\nSignificativas: sem ajuste {np.sum(brutos<0.05)} | Bonferroni {np.sum(bonf<0.05)}"
      f" | Holm {np.sum(holm<0.05)} | Tukey {np.sum(tuk<0.05)}")

# 4. O nível individual e o intervalo de confiança correspondentes
alfa = 0.05
print(f"\nNível individual sob Bonferroni: {alfa}/{m} = {alfa/m:.6f}")
tcrit = stats.t.ppf(1 - alfa/(2*m), gl)
print(f"t crítico bilateral correspondente: {tcrit:.4f} (contra {stats.t.ppf(1-alfa/2, gl):.4f} sem ajuste)")
print(f"q crítico de Tukey: {stats.studentized_range.ppf(1-alfa, k, gl)/np.sqrt(2):.4f} na mesma escala")
print("\nIntervalos de confiança simultâneos de 95% (Bonferroni)")
for i in np.argsort(-np.abs(difs))[:5]:
    a, b = list(itertools.combinations(range(k), 2))[i]
    ep = np.sqrt(QM*(1/ns[a] + 1/ns[b]))
    print(f"  {rotulos[i]:<26}{difs[i]:8.2f}  IC = [{difs[i]-tcrit*ep:8.2f}; {difs[i]+tcrit*ep:8.2f}]")

# 5. Bonferroni fora das comparações pareadas: três contrastes planejados
print("\nBonferroni não é só para comparações pareadas. Três contrastes planejados:")
contr = {
 "vegetais (horsebean, linseed, soybean) vs animais (meatmeal, casein)":
     np.array([1/3, 1/3, 1/3, 0, -1/2, -1/2]),
 "sunflower vs a média de todas as outras":
     np.array([-1/5, -1/5, -1/5, 1, -1/5, -1/5]),
 "horsebean vs a média de todas as outras":
     np.array([1, -1/5, -1/5, -1/5, -1/5, -1/5])}
pc = []
for nome, c in contr.items():
    L = float(c @ medias); ep = float(np.sqrt(QM * np.sum(c**2 / ns)))
    t = L / ep; pv = 2 * stats.t.sf(abs(t), gl); pc.append(pv)
    print(f"  {nome}\n    L = {L:+8.3f} g | EP = {ep:6.3f} | t({gl}) = {t:+7.3f} | p = {pv:.3e} | "
          f"Bonferroni (3 testes) = {min(1, 3*pv):.3e}")
