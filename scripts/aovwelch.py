# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados fictícios: resistência à compressão (MPa) de corpos de prova de concreto
# de quatro centrais dosadoras. A central A tem o processo menos controlado:
# muitos corpos de prova e dispersão muito maior que as demais.
A = np.array([57.0, 54.0, 51.5, 31.1, 55.9, 32.9, 60.2, 57.5, 59.5, 59.4,
              54.7, 47.2, 49.2, 65.6, 46.8, 49.9, 45.5, 47.3, 49.2, 54.3])
B = np.array([48.4, 48.3, 47.8, 49.2, 46.3, 49.1, 51.6, 47.5, 48.9, 54.6])
C = np.array([50.5, 48.9, 51.9, 57.5, 58.1, 50.0, 54.4, 53.0, 51.5, 48.1, 55.6, 50.8])
D = np.array([55.6, 55.2, 58.7, 57.4, 52.7, 54.5, 57.1, 52.3, 54.2, 55.5])
grupos = [A, B, C, D]
nomes = ["A", "B", "C", "D"]
k = len(grupos)
ns = np.array([len(g) for g in grupos], dtype=float)
N = int(ns.sum())
medias = np.array([g.mean() for g in grupos])
vars_ = np.array([g.var(ddof=1) for g in grupos])
todos = np.concatenate(grupos)

print("central   n    média     DP   variância   EP da média")
for i, nm in enumerate(nomes):
    print(f"   {nm}    {int(ns[i]):2d}   {medias[i]:6.3f}  {np.sqrt(vars_[i]):5.3f}    {vars_[i]:7.3f}      {np.sqrt(vars_[i]/ns[i]):.3f}")
print(f"Razão entre a maior e a menor variância = {vars_.max()/vars_.min():.2f}")

# 1. Pressupostos
res = np.concatenate([g - g.mean() for g in grupos])
print(f"\nShapiro-Wilk dos resíduos: W = {stats.shapiro(res)[0]:.4f}, p-valor = {stats.shapiro(res)[1]:.4f}")
for nm, g in zip(nomes, grupos):
    print(f"  Shapiro-Wilk (central {nm}): p-valor = {stats.shapiro(g)[1]:.4f}")
print(f"Levene (centro = média):   W = {stats.levene(*grupos, center='mean')[0]:.4f}, p-valor = {stats.levene(*grupos, center='mean')[1]:.5f}")
print(f"Brown-Forsythe (mediana):  W = {stats.levene(*grupos, center='median')[0]:.4f}, p-valor = {stats.levene(*grupos, center='median')[1]:.5f}")
print(f"Bartlett: qui-quadrado = {stats.bartlett(*grupos)[0]:.4f}, p-valor = {stats.bartlett(*grupos)[1]:.3e}")

# 2. ANOVA clássica (de variância agrupada)
SQ_entre = np.sum(ns * (medias - todos.mean())**2)
SQ_dentro = np.sum((ns - 1) * vars_)
gl1, gl2 = k - 1, N - k
QM_entre, QM_dentro = SQ_entre / gl1, SQ_dentro / gl2
Fc = QM_entre / QM_dentro
pc = stats.f.sf(Fc, gl1, gl2)
print("\nANOVA clássica")
print("Fonte         GL       SQ       QM       F     p-valor")
print(f"Central     {gl1:3d}  {SQ_entre:8.3f}  {QM_entre:7.3f}  {Fc:.4f}  {pc:.5f}")
print(f"Resíduo     {gl2:3d}  {SQ_dentro:8.3f}  {QM_dentro:7.3f}")
print(f"QM do resíduo = {QM_dentro:.3f} -> DP agrupado = {np.sqrt(QM_dentro):.3f} MPa")
print(f"scipy: {stats.f_oneway(*grupos)}")

# 3. ANOVA de Welch, passo a passo
w = ns / vars_                                  # peso de cada grupo: precisão da média
mu_w = np.sum(w * medias) / np.sum(w)           # média geral ponderada
print("\nANOVA de Welch, passo a passo")
print("central     w = n/s²   contribuição w(média - mu_w)²")
for i, nm in enumerate(nomes):
    print(f"   {nm}       {w[i]:7.4f}        {w[i]*(medias[i]-mu_w)**2:8.4f}")
print(f"Soma dos pesos = {w.sum():.4f} | média geral ponderada mu_w = {mu_w:.4f}")
lam = np.sum((1 - w / w.sum())**2 / (ns - 1))
Aw = np.sum(w * (medias - mu_w)**2) / (k - 1)
Bw = 1 + 2 * (k - 2) / (k**2 - 1) * lam
Fw = Aw / Bw
glw = (k**2 - 1) / (3 * lam)
pw = stats.f.sf(Fw, k - 1, glw)
print(f"Numerador A = {Aw:.4f} | lambda = {lam:.6f} | denominador B = {Bw:.6f}")
print(f"F de Welch = {Fw:.4f} | gl = ({k-1}; {glw:.3f}) | p-valor = {pw:.3e}")

# 4. Comparações múltiplas: Tukey (variância agrupada) x Games-Howell (variâncias próprias)
print("\nComparações múltiplas")
print("par      diferença   Tukey (p ajustado)   Games-Howell (p ajustado)")
for i, j in itertools.combinations(range(k), 2):
    dif = medias[j] - medias[i]
    ep_t = np.sqrt(QM_dentro / 2 * (1/ns[i] + 1/ns[j]))
    p_tukey = stats.studentized_range.sf(abs(dif) / ep_t, k, gl2)
    ep_gh = np.sqrt((vars_[i]/ns[i] + vars_[j]/ns[j]) / 2)
    gl_gh = (vars_[i]/ns[i] + vars_[j]/ns[j])**2 / (
        (vars_[i]/ns[i])**2/(ns[i]-1) + (vars_[j]/ns[j])**2/(ns[j]-1))
    p_gh = stats.studentized_range.sf(abs(dif) / ep_gh, k, gl_gh)
    marca = lambda v: "*" if v < 0.05 else " "
    print(f"{nomes[i]} vs {nomes[j]}   {dif:+7.3f}        {p_tukey:.4f} {marca(p_tukey)}"
          f"              {p_gh:.4f} {marca(p_gh)}  (gl = {gl_gh:.2f})")

# 5. Tamanho de efeito
eta2 = SQ_entre / (SQ_entre + SQ_dentro)
print(f"\neta² = {eta2:.4f} (ANOVA clássica)")
print(f"omega² = {(SQ_entre - gl1*QM_dentro)/(SQ_entre + SQ_dentro + QM_dentro):.4f}")

# 6. Simulação: taxa de erro tipo I quando as variâncias diferem e as médias são iguais
print("\nSimulação: 20.000 experimentos com as MESMAS médias populacionais (52 MPa)")
print("e os desvios-padrão e tamanhos deste exemplo (20/9; 10/2; 12/2,5; 10/2).")
rng = np.random.default_rng(2026)
ns_sim = [20, 10, 12, 10]
sd_sim = [9.0, 2.0, 2.5, 2.0]
rej_c = rej_w = 0
M = 20000
for _ in range(M):
    gs = [rng.normal(52.0, s, n) for n, s in zip(ns_sim, sd_sim)]
    if stats.f_oneway(*gs)[1] < 0.05:
        rej_c += 1
    m = np.array([g.mean() for g in gs]); v = np.array([g.var(ddof=1) for g in gs])
    nn = np.array(ns_sim, dtype=float); ww = nn / v
    muw = np.sum(ww*m)/ww.sum()
    lm = np.sum((1 - ww/ww.sum())**2/(nn-1))
    fw = (np.sum(ww*(m-muw)**2)/(k-1)) / (1 + 2*(k-2)/(k**2-1)*lm)
    if stats.f.sf(fw, k-1, (k**2-1)/(3*lm)) < 0.05:
        rej_w += 1
print(f"ANOVA clássica rejeita H0 em {100*rej_c/M:.2f}% dos casos (o nominal é 5%)")
print(f"ANOVA de Welch rejeita H0 em {100*rej_w/M:.2f}% dos casos")

# 7. Cenário inverso: variância grande no grupo PEQUENO
print("\nCenário inverso: a maior variância no grupo de menor tamanho (10/2; 12/2,5; 20/2; 6/9)")
rng = np.random.default_rng(7)
ns_inv = [10, 12, 20, 6]
sd_inv = [2.0, 2.5, 2.0, 9.0]
rej_c = rej_w = 0
for _ in range(M):
    gs = [rng.normal(52.0, s, n) for n, s in zip(ns_inv, sd_inv)]
    if stats.f_oneway(*gs)[1] < 0.05:
        rej_c += 1
    m = np.array([g.mean() for g in gs]); v = np.array([g.var(ddof=1) for g in gs])
    nn = np.array(ns_inv, dtype=float); ww = nn / v
    muw = np.sum(ww*m)/ww.sum()
    lm = np.sum((1 - ww/ww.sum())**2/(nn-1))
    fw = (np.sum(ww*(m-muw)**2)/(k-1)) / (1 + 2*(k-2)/(k**2-1)*lm)
    if stats.f.sf(fw, k-1, (k**2-1)/(3*lm)) < 0.05:
        rej_w += 1
print(f"ANOVA clássica rejeita H0 em {100*rej_c/M:.2f}% dos casos")
print(f"ANOVA de Welch rejeita H0 em {100*rej_w/M:.2f}% dos casos")
