# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: chickwts (R). Peso (g) de 71 pintinhos aos 6 dias, segundo seis rações.
# É o mesmo conjunto usado nas fichas de Bonferroni e de Tukey, para comparação direta.
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
m = k * (k - 1) // 2
print(f"ANOVA: F({k-1}; {gl}) = {stats.f_oneway(*gs)[0]:.4f} | QM do resíduo = {QM:.3f} | m = {m} comparações")

# 1. A conta exata contra a desigualdade de Bonferroni
alfa = 0.05
print("\nO nível individual que cada ajuste adota, para um alfa global de 5%")
print(" m testes   Bonferroni: alfa/m   Šidák: 1-(1-alfa)^(1/m)   razão Šidák/Bonferroni")
for mm in (1, 2, 3, 5, 10, 15, 20, 50, 100, 1000):
    b = alfa / mm
    s = 1 - (1 - alfa)**(1/mm)
    print(f"  {mm:5d}      {b:16.8f}   {s:22.8f}   {s/b:21.6f}")
print("\nŠidák é sempre um pouco menos exigente, e a vantagem encolhe quando m cresce:")
print("com m = 15, o nível individual passa de 0,00333 para 0,00342, apenas 2,5% maior.")

# 2. Taxa de erro do conjunto, sob independência
print("\nSe os m testes fossem independentes, a taxa de erro do conjunto seria:")
print(" m testes   sem ajuste   com Bonferroni   com Šidák")
for mm in (1, 3, 5, 10, 15, 50):
    print(f"  {mm:5d}     {1-(1-alfa)**mm:10.6f}   {1-(1-alfa/mm)**mm:14.6f}   "
          f"{1-(1-(1-(1-alfa)**(1/mm)))**mm:9.6f}")
print("Šidák atinge exatamente 0,05; Bonferroni fica abaixo. É essa folga não usada")
print("que torna o Bonferroni conservador.")

# 3. Simulação com testes de fato independentes, e depois com os dependentes
print("\nSimulação A: 20.000 famílias de 15 testes INDEPENDENTES, todas as nulas verdadeiras")
rng = np.random.default_rng(7)
M = 20000
sem = bon = sid = 0
for _ in range(M):
    pv = rng.random(m)
    if (pv < alfa).any(): sem += 1
    if (pv < alfa/m).any(): bon += 1
    if (pv < 1-(1-alfa)**(1/m)).any(): sid += 1
print(f"  sem ajuste: {100*sem/M:.2f}% | Bonferroni: {100*bon/M:.2f}% | Šidák: {100*sid/M:.2f}%")

print("\nSimulação B: 10.000 experimentos com 6 grupos de médias iguais (n = 12),")
print("em que as 15 comparações pareadas NÃO são independentes")
rng = np.random.default_rng(11)
M2 = 10000
pares = list(itertools.combinations(range(6), 2))
sem = bon = sid = 0
for _ in range(M2):
    g6 = [rng.normal(0, 1, 12) for _ in range(6)]
    pv = np.array([stats.ttest_ind(g6[a], g6[b]).pvalue for a, b in pares])
    if (pv < alfa).any(): sem += 1
    if (pv < alfa/m).any(): bon += 1
    if (pv < 1-(1-alfa)**(1/m)).any(): sid += 1
print(f"  sem ajuste: {100*sem/M2:.2f}% | Bonferroni: {100*bon/M2:.2f}% | Šidák: {100*sid/M2:.2f}%")
print("  Com dependência, os dois ficam abaixo de 5%: a independência suposta por Šidák")
print("  não vale, e a vantagem teórica dele sobre o Bonferroni praticamente some.")

# 4. As 15 comparações dos dados reais
print(f"\nAs {m} comparações ({gl} gl), ordenadas pelo valor-p bruto")
brutos = []; difs = []; rot = []
for a, b in itertools.combinations(range(k), 2):
    dif = medias[b] - medias[a]
    t = dif / np.sqrt(QM * (1/ns[a] + 1/ns[b]))
    brutos.append(2*stats.t.sf(abs(t), gl)); difs.append(dif); rot.append(f"{nomes[b]} - {nomes[a]}")
brutos = np.array(brutos); difs = np.array(difs)
bonf = np.minimum(1, m*brutos)
sida = 1 - (1 - brutos)**m
ordem = np.argsort(brutos)
holm = np.empty(m); ac = 0.0
hs = np.empty(m); ac2 = 0.0
for pos, idx in enumerate(ordem):
    ac = max(ac, (m - pos)*brutos[idx]); holm[idx] = min(1.0, ac)
    ac2 = max(ac2, 1 - (1 - brutos[idx])**(m - pos)); hs[idx] = min(1.0, ac2)
tuk = np.array([stats.studentized_range.sf(abs(d)/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl)
                for d, (a, b) in zip(difs, itertools.combinations(range(k), 2))])
mk = lambda v: "*" if v < 0.05 else " "
print(f"{'comparação':<26}{'p bruto':>10}{'Bonferroni':>12}{'Šidák':>11}{'Holm':>11}{'Holm-Šidák':>13}{'Tukey':>11}")
for i in ordem:
    print(f"{rot[i]:<26}{brutos[i]:10.5f}{bonf[i]:11.5f}{mk(bonf[i])}{sida[i]:10.5f}{mk(sida[i])}"
          f"{holm[i]:10.5f}{mk(holm[i])}{hs[i]:12.5f}{mk(hs[i])}{tuk[i]:10.5f}{mk(tuk[i])}")
print(f"\nSignificativas: bruto {np.sum(brutos<0.05)} | Bonferroni {np.sum(bonf<0.05)} | "
      f"Šidák {np.sum(sida<0.05)} | Holm {np.sum(holm<0.05)} | Holm-Šidák {np.sum(hs<0.05)} | "
      f"Tukey {np.sum(tuk<0.05)}")
print(f"\nMaior diferença entre Šidák e Bonferroni nestes dados: "
      f"{np.max(bonf - sida):.6f} (em valor-p ajustado)")
print("Nenhuma conclusão muda. Essa é a situação típica.")

# 5. Intervalos simultâneos
tb = stats.t.ppf(1 - alfa/(2*m), gl)
ts = stats.t.ppf(1 - (1 - (1-alfa)**(1/m))/2, gl)
tt = stats.studentized_range.ppf(1-alfa, k, gl)/np.sqrt(2)
print(f"\nMultiplicadores críticos na escala t ({gl} gl)")
print(f"  sem ajuste  : {stats.t.ppf(1-alfa/2, gl):.4f}")
print(f"  Šidák       : {ts:.4f}")
print(f"  Bonferroni  : {tb:.4f}")
print(f"  Tukey       : {tt:.4f}")
print("Tukey é o menor dos três ajustes porque foi construído especificamente para")
print("o conjunto de todas as comparações pareadas, aproveitando a dependência entre elas.")
i = int(np.argmax(np.abs(difs)))
a, b = list(itertools.combinations(range(k), 2))[i]
ep = np.sqrt(QM*(1/ns[a] + 1/ns[b]))
print(f"\nPara {rot[i]} (diferença de {difs[i]:.2f} g, EP = {ep:.3f}):")
for nome, cr in [("sem ajuste", stats.t.ppf(1-alfa/2, gl)), ("Šidák", ts), ("Bonferroni", tb), ("Tukey", tt)]:
    print(f"  {nome:<12} IC 95% = [{difs[i]-cr*ep:7.2f}; {difs[i]+cr*ep:7.2f}]  (largura {2*cr*ep:.2f} g)")
