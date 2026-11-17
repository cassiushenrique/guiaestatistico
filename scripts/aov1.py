# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: PlantGrowth (R). Peso seco de plantas sob controle e dois tratamentos.
ctrl = np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14])
trt1 = np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69])
trt2 = np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26])
grupos = [ctrl, trt1, trt2]
nomes = ["ctrl", "trt1", "trt2"]
k = len(grupos)
N = sum(len(g) for g in grupos)
todos = np.concatenate(grupos)

print("grupo   n   média    DP    variância")
for nm, g in zip(nomes, grupos):
    print(f" {nm}   {len(g):2d}  {g.mean():.3f}  {g.std(ddof=1):.3f}    {g.var(ddof=1):.3f}")
print(f"Média geral = {todos.mean():.4f} | N = {N}")

# 1. Pressupostos
residuos = np.concatenate([g - g.mean() for g in grupos])
W, pW = stats.shapiro(residuos)
print(f"\nShapiro-Wilk dos resíduos: W = {W:.4f}, p-valor = {pW:.4f}")
print(f"Levene (centro = média):   W = {stats.levene(*grupos, center='mean')[0]:.4f}, "
      f"p-valor = {stats.levene(*grupos, center='mean')[1]:.4f}")
print(f"Brown-Forsythe (mediana):  W = {stats.levene(*grupos, center='median')[0]:.4f}, "
      f"p-valor = {stats.levene(*grupos, center='median')[1]:.4f}")
print(f"Bartlett: qui-quadrado = {stats.bartlett(*grupos)[0]:.4f}, p-valor = {stats.bartlett(*grupos)[1]:.4f}")
vs = [g.var(ddof=1) for g in grupos]
print(f"Razão entre a maior e a menor variância = {max(vs)/min(vs):.3f}")

# 2. Decomposição da soma de quadrados, feita à mão
SQ_total = ((todos - todos.mean())**2).sum()
SQ_entre = sum(len(g) * (g.mean() - todos.mean())**2 for g in grupos)
SQ_dentro = sum(((g - g.mean())**2).sum() for g in grupos)
gl_entre, gl_dentro = k - 1, N - k
QM_entre, QM_dentro = SQ_entre / gl_entre, SQ_dentro / gl_dentro
F = QM_entre / QM_dentro
p = stats.f.sf(F, gl_entre, gl_dentro)
print(f"\nSQ total = {SQ_total:.4f} = SQ entre ({SQ_entre:.4f}) + SQ dentro ({SQ_dentro:.4f})")
print("\nFonte         GL      SQ      QM       F     p-valor")
print(f"Tratamento  {gl_entre:3d}  {SQ_entre:7.4f}  {QM_entre:.4f}  {F:.4f}  {p:.5f}")
print(f"Resíduo     {gl_dentro:3d}  {SQ_dentro:7.4f}  {QM_dentro:.4f}")
print(f"Total       {N-1:3d}  {SQ_total:7.4f}")
print(f"scipy: {stats.f_oneway(*grupos)}")

# 3. Tamanho de efeito
eta2 = SQ_entre / SQ_total
omega2 = (SQ_entre - gl_entre * QM_dentro) / (SQ_total + QM_dentro)
f_cohen = np.sqrt(eta2 / (1 - eta2))
print(f"\neta² = {eta2:.4f} | omega² = {omega2:.4f} | f de Cohen = {f_cohen:.4f}")

# 4. Tukey HSD: comparações múltiplas com a amplitude estudentizada
print("\nTukey HSD (todas as comparações):")
qcrit = stats.studentized_range.ppf(0.95, k, gl_dentro)
print(f"q crítico (5%, k = {k}, gl = {gl_dentro}) = {qcrit:.4f}")
for i, j in itertools.combinations(range(k), 2):
    dif = grupos[j].mean() - grupos[i].mean()
    ep = np.sqrt(QM_dentro / 2 * (1/len(grupos[i]) + 1/len(grupos[j])))
    q = abs(dif) / ep
    pa = stats.studentized_range.sf(q, k, gl_dentro)
    margem = qcrit * ep
    print(f"  {nomes[j]} - {nomes[i]}: diferença = {dif:+.4f} | IC 95% = "
          f"[{dif-margem:+.4f}; {dif+margem:+.4f}] | q = {q:.4f} | p ajustado = {pa:.4f}"
          + ("  *" if pa < 0.05 else ""))
print(f"(confere com TukeyHSD do R: a mesma estatística q e o mesmo IC)")

# 5. Contraste planejado: controle contra a média dos dois tratamentos
coef = np.array([1.0, -0.5, -0.5])
medias = np.array([g.mean() for g in grupos])
ns = np.array([len(g) for g in grupos], dtype=float)
L = coef @ medias
ep_L = np.sqrt(QM_dentro * np.sum(coef**2 / ns))
t_L = L / ep_L
p_L = 2 * stats.t.sf(abs(t_L), gl_dentro)
print(f"\nContraste ctrl - (trt1 + trt2)/2 = {L:+.4f} | EP = {ep_L:.4f} | "
      f"t = {t_L:+.4f} | p-valor = {p_L:.4f}")
print(f"SQ do contraste = {L**2 / np.sum(coef**2 / ns):.4f} (de um total de {SQ_entre:.4f} entre tratamentos)")

# 6. Alternativas para comparação
print("")
# Welch calculado à mão
w = ns / np.array(vs)
mu_w = np.sum(w * medias) / np.sum(w)
A = np.sum(w * (medias - mu_w)**2) / (k - 1)
B = 1 + 2*(k-2)/(k**2-1) * np.sum((1 - w/np.sum(w))**2 / (ns - 1))
Fw = A / B
glw = (k**2 - 1) / (3 * np.sum((1 - w/np.sum(w))**2 / (ns - 1)))
print(f"ANOVA de Welch:    F = {Fw:.4f}, gl = ({k-1}; {glw:.3f}), p-valor = {stats.f.sf(Fw, k-1, glw):.5f}")
H, pH = stats.kruskal(*grupos)
print(f"Kruskal-Wallis:    H = {H:.4f}, p-valor = {pH:.5f}")
