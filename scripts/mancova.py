# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados fictícios: as MESMAS 36 parcelas da ficha da ANCOVA, com uma segunda resposta.
# Três cultivares de milho; em cada parcela mediu-se, antes da semeadura, o teor de
# matéria orgânica do solo (%), e na colheita a produtividade (kg/ha) e o teor de
# proteína do grão (%). Produtividade e proteína costumam andar em sentidos opostos.
mo = np.array([3.82,2.17,3.92,3.88,2.48,3.17,4.13,3.79,3.75,3.39,3.23,2.69,
               3.29,1.37,3.83,3.21,3.28,3.35,2.98,2.49,2.84,2.39,3.49,2.75,
               2.49,3.90,1.74,2.29,1.76,3.19,1.38,2.34,2.43,3.44,1.70,2.49])
prod = np.array([5309,4323,5577,5282,4288,4849,5513,5078,5120,4772,4946,4752,
                 5146,3691,5311,5377,5342,4838,5047,4578,5242,4517,5305,4803,
                 5092,5845,4461,4713,4182,5490,4199,4609,5057,5600,4502,4987], float)
prot = np.array([10.75,10.39,10.69,11.18,10.50,10.41,11.10,10.89,11.02,11.15,10.72,10.29,
                 10.16, 9.86,10.61,10.33,10.28,10.92,10.38,10.21, 9.73,10.07,10.40,10.03,
                  9.49, 9.99, 9.35, 9.59, 9.63,10.26, 9.14, 9.97, 9.20,10.02, 9.42, 9.62])
cult = np.repeat([0, 1, 2], 12)
rot = ["A", "B", "C"]
resp = ["produtividade (kg/ha)", "proteína (%)"]
Y = np.column_stack([prod, prot])
N, p = Y.shape
k = 3

def ols(X, Y):
    """Mínimos quadrados multivariado: devolve coeficientes e matriz E = resíduo'resíduo."""
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    R = Y - X @ B
    return B, R.T @ R, len(Y) - np.linalg.matrix_rank(X)

def testes(H, E, glh, gle, p):
    """Quatro estatísticas multivariadas, com as aproximações F usuais."""
    autov = np.sort(np.linalg.eigvals(np.linalg.inv(E) @ H).real)[::-1]
    s = min(glh, p)
    pillai = float(np.sum(autov[:s] / (1 + autov[:s])))
    wilks = float(np.prod(1 / (1 + autov[:s])))
    hl = float(np.sum(autov[:s])); roy = float(autov[0])
    m = (abs(glh - p) - 1) / 2; nn = (gle - p - 1) / 2
    out = {}
    d1 = s*(2*m + s + 1); d2 = s*(2*nn + s + 1)
    F = (pillai / (s - pillai)) * (d2 / d1)
    out["Pillai"] = (pillai, F, d1, d2, stats.f.sf(F, d1, d2))
    r_ = gle - (p - glh + 1)/2; u = (p*glh - 2)/4; den = p**2 + glh**2 - 5
    t = np.sqrt((p**2*glh**2 - 4)/den) if den > 0 else 1.0
    d1 = p*glh; d2 = r_*t - 2*u
    F = ((1 - wilks**(1/t)) / wilks**(1/t)) * (d2/d1)
    out["Wilks"] = (wilks, F, d1, d2, stats.f.sf(F, d1, d2))
    b = (gle + glh - p - 1)*(gle - 1)/((gle - p - 3)*(gle - p))
    d1 = s*(2*m + s + 1); d2 = 4 + (d1 + 2)/(b - 1)
    cte = d1*(d2 - 2)/(d2*(gle - p - 1)); F = hl/cte
    out["Hotelling-Lawley"] = (hl, F, d1, d2, stats.f.sf(F, d1, d2))
    rr = max(p, glh); d1 = rr; d2 = gle - rr + glh; F = roy*d2/d1
    out["Roy"] = (roy, F, d1, d2, stats.f.sf(F, d1, d2))
    return out, autov

def imprime(out):
    print("estatística           valor      F aprox.     gl1     gl2     p-valor")
    for nome, (v, F, d1, d2, pv) in out.items():
        print(f"{nome:<20}{v:8.5f}  {F:11.4f}  {d1:5.1f}  {d2:6.2f}   {pv:.4e}")

print("cultivar  n   produtividade média   proteína média   matéria orgânica média")
for i in range(k):
    m = cult == i
    print(f"    {rot[i]}     {m.sum():2d}        {prod[m].mean():8.2f}          {prot[m].mean():6.3f}            {mo[m].mean():.4f}")
print(f"  geral     {N}        {prod.mean():8.2f}          {prot.mean():6.3f}            {mo.mean():.4f}")

# 1. Como as duas respostas se relacionam
um = np.ones(N)
D = np.column_stack([(cult == 1).astype(float), (cult == 2).astype(float)])
X_comp = np.column_stack([um, mo, D])
B_comp, E_comp, gl_comp = ols(X_comp, Y)
R = Y - X_comp @ B_comp
print(f"\nCorrelação bruta entre produtividade e proteína: r = {np.corrcoef(prod, prot)[0,1]:+.4f}")
print(f"Correlação entre os RESÍDUOS do modelo:          r = {np.corrcoef(R.T)[0,1]:+.4f}")
print("A correlação bruta é fraca porque o solo empurra as duas para cima ao mesmo tempo.")
print("Descontados solo e cultivar, o que sobra é o compromisso clássico: parcelas que")
print("produziram mais que o esperado entregaram grão com menos proteína.")

# 2. Pressuposto: paralelismo multivariado (homogeneidade das inclinações)
X_int = np.column_stack([um, mo, D, D * mo[:, None]])
B_int, E_int, gl_int = ols(X_int, Y)
H_par = E_comp - E_int
out, _ = testes(H_par, E_int, 2, gl_int, p)
print("\nParalelismo multivariado (interação cultivar x matéria orgânica)")
imprime(out)
print("As inclinações não diferem: o modelo de planos paralelos está autorizado.")
print("Inclinações por cultivar e resposta (por ponto percentual de MO):")
for j in range(p):
    inc = [B_int[1, j], B_int[1, j] + B_int[4, j], B_int[1, j] + B_int[5, j]]
    print(f"  {resp[j]:<22}" + "  ".join(f"{rot[i]}: {inc[i]:9.3f}" for i in range(k)))

# 3. MANCOVA: efeito do cultivar, ajustado pela covariável
X_cov = np.column_stack([um, mo])            # só a covariável
_, E_cov, _ = ols(X_cov, Y)
H_cult = E_cov - E_comp
print("\nMANCOVA: efeito do CULTIVAR, ajustado pela matéria orgânica")
out_c, autov_c = testes(H_cult, E_comp, k - 1, gl_comp, p)
imprime(out_c)
print(f"Autovalores de E⁻¹H: {np.round(autov_c[:2], 4)}")

X_sem = np.column_stack([um, D])             # só o fator
_, E_sem, _ = ols(X_sem, Y)
H_mo = E_sem - E_comp
print("\nEfeito multivariado da COVARIÁVEL, ajustado pelo cultivar")
out_m, _ = testes(H_mo, E_comp, 1, gl_comp, p)
imprime(out_m)

# 4. Para comparar: MANOVA sem a covariável
_, E_nul, _ = ols(um[:, None], Y)
print("\nMANOVA sem a covariável (o que se veria ignorando o solo)")
out_s, _ = testes(E_nul - E_sem, E_sem, k - 1, N - k, p)
imprime(out_s)
print(f"det(E) com covariável = {np.linalg.det(E_comp):.4e}")
print(f"det(E) sem covariável = {np.linalg.det(E_sem):.4e}")
print(f"A covariável reduz o volume do erro multivariado em "
      f"{100*(1 - np.linalg.det(E_comp)/np.linalg.det(E_sem)):.2f}%.")

# 5. Médias ajustadas, resposta por resposta
mbar = mo.mean()
XtXi = np.linalg.inv(X_comp.T @ X_comp)
QM = np.diag(E_comp) / gl_comp
print(f"\nMédias ajustadas na matéria orgânica média ({mbar:.4f}%)")
print("cultivar   produtividade obs.  ajust.      proteína obs.  ajust.")
aj = np.zeros((k, p))
for i in range(k):
    x0 = np.array([1.0, mbar, float(i == 1), float(i == 2)])
    aj[i] = x0 @ B_comp
    print(f"    {rot[i]}        {prod[cult==i].mean():8.2f}  {aj[i,0]:8.2f}      "
          f"{prot[cult==i].mean():6.3f}  {aj[i,1]:6.3f}")

# 6. ANCOVAs univariadas de acompanhamento, com ajuste de Bonferroni
print("\nANCOVAs univariadas (ajuste de Bonferroni para 2 respostas)")
print("resposta                   F(2; 32)      p bruto    p ajustado")
for j in range(p):
    F = ((E_cov[j, j] - E_comp[j, j]) / (k - 1)) / (E_comp[j, j] / gl_comp)
    pv = stats.f.sf(F, k - 1, gl_comp)
    print(f"{resp[j]:<26}{F:9.4f}  {pv:11.3e}  {min(1, p*pv):11.3e}")

print("\nComparações entre médias ajustadas (Tukey, dentro de cada resposta)")
q95 = stats.studentized_range.ppf(0.95, k, gl_comp)
for j in range(p):
    print(f"  {resp[j]}")
    for i1, i2 in itertools.combinations(range(k), 2):
        c = np.array([0.0, 0.0, float(i2 == 1) - float(i1 == 1), float(i2 == 2) - float(i1 == 2)])
        dif = float(c @ B_comp[:, j]); ep = float(np.sqrt(c @ XtXi @ c * QM[j]))
        pa = stats.studentized_range.sf(abs(dif)/(ep/np.sqrt(2)), k, gl_comp)
        marg = q95 * ep / np.sqrt(2)
        print(f"    {rot[i2]} - {rot[i1]}: {dif:+9.3f} | IC 95% = [{dif-marg:+9.3f}; {dif+marg:+9.3f}]"
              f" | p ajustado = {pa:.5f}" + ("  *" if pa < 0.05 else ""))

# 7. Pressupostos restantes
print("\nPressupostos")
ns = np.array([np.sum(cult == i) for i in range(k)], float)
Ss = [np.cov(R[cult == i], rowvar=False) for i in range(k)]
Sp = sum((ns[j] - 1) * Ss[j] for j in range(k)) / (N - k)
M = (N - k) * np.log(np.linalg.det(Sp)) - sum((ns[j] - 1) * np.log(np.linalg.det(Ss[j])) for j in range(k))
cc = (sum(1/(ns - 1)) - 1/(N - k)) * (2*p**2 + 3*p - 1) / (6*(p + 1)*(k - 1))
glM = (k - 1) * p * (p + 1) / 2
print(f"  Teste M de Box nos resíduos: qui-quadrado = {M*(1-cc):.4f}, gl = {int(glM)}, "
      f"p-valor = {stats.chi2.sf(M*(1-cc), glM):.4f}")
for j in range(p):
    print(f"  Shapiro-Wilk dos resíduos de {resp[j]:<22} p-valor = {stats.shapiro(R[:, j])[1]:.4f}")
    print(f"  Levene dos resíduos de       {resp[j]:<22} p-valor = "
          f"{stats.levene(*[R[cult==i, j] for i in range(k)], center='median')[1]:.4f}")
d2 = np.sum((R @ np.linalg.inv(np.cov(R, rowvar=False))) * R, axis=1)
b2 = np.mean(d2**2)
Bk = (b2 - p*(p + 2)) / np.sqrt(8*p*(p + 2)/N)
print(f"  Curtose multivariada de Mardia nos resíduos: z = {Bk:+.4f}, "
      f"p-valor = {2*stats.norm.sf(abs(Bk)):.4f}")
