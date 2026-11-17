# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados fictícios: produtividade (kg/ha) de três cultivares de milho em 36 parcelas,
# 12 por cultivar, e o teor de matéria orgânica do solo (%) medido em cada parcela
# ANTES da semeadura. A matéria orgânica é a covariável: afeta a produtividade e
# variou entre as parcelas por razões alheias ao experimento.
mo = np.array([3.82, 2.17, 3.92, 3.88, 2.48, 3.17, 4.13, 3.79, 3.75, 3.39, 3.23, 2.69,
               3.29, 1.37, 3.83, 3.21, 3.28, 3.35, 2.98, 2.49, 2.84, 2.39, 3.49, 2.75,
               2.49, 3.90, 1.74, 2.29, 1.76, 3.19, 1.38, 2.34, 2.43, 3.44, 1.70, 2.49])
prod = np.array([5309, 4323, 5577, 5282, 4288, 4849, 5513, 5078, 5120, 4772, 4946, 4752,
                 5146, 3691, 5311, 5377, 5342, 4838, 5047, 4578, 5242, 4517, 5305, 4803,
                 5092, 5845, 4461, 4713, 4182, 5490, 4199, 4609, 5057, 5600, 4502, 4987],
                dtype=float)
cult = np.repeat([0, 1, 2], 12)
rot = ["A", "B", "C"]
k, N = 3, len(prod)

def ols(X, y):
    """Mínimos quadrados: devolve coeficientes, SQ do resíduo e graus de liberdade."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, float(r @ r), len(y) - np.linalg.matrix_rank(X)

print("cultivar   n   produtividade média   matéria orgânica média")
for i in range(k):
    m = cult == i
    print(f"    {rot[i]}     {m.sum():2d}        {prod[m].mean():8.2f}              {mo[m].mean():.4f}")
print(f"  geral     {N}        {prod.mean():8.2f}              {mo.mean():.4f}")
print(f"\nCorrelação entre matéria orgânica e produtividade: r = {np.corrcoef(mo, prod)[0,1]:.4f}")

# 1. O que a ANOVA de um fator responde, ignorando a covariável
F0, p0 = stats.f_oneway(*[prod[cult == i] for i in range(k)])
um = np.ones(N)
D = np.column_stack([(cult == 1).astype(float), (cult == 2).astype(float)])
b_nul, SQ_nul, gl_nul = ols(um[:, None], prod)
b_fat, SQ_fat, gl_fat = ols(np.column_stack([um, D]), prod)
print(f"\nANOVA de um fator (sem covariável): F({k-1}; {N-k}) = {F0:.4f}, p-valor = {p0:.4f}")
print(f"  QM do resíduo = {SQ_fat/gl_fat:.2f} | DP = {np.sqrt(SQ_fat/gl_fat):.2f} kg/ha")
print("  conclusão: não há evidência de diferença entre cultivares.")
print("  mas as parcelas da cultivar A caíram em solo mais rico e as da C, em solo mais pobre:")
Fmo, pmo = stats.f_oneway(*[mo[cult == i] for i in range(k)])
print(f"  a covariável difere entre cultivares: F = {Fmo:.4f}, p-valor = {pmo:.5f}")

# 2. Pressuposto central: paralelismo das retas
X_int = np.column_stack([um, mo, D, D * mo[:, None]])
X_par = np.column_stack([um, mo, D])
b_int, SQ_int, gl_int = ols(X_int, prod)
b_par, SQ_par, gl_par = ols(X_par, prod)
F_hom = ((SQ_par - SQ_int) / (k - 1)) / (SQ_int / gl_int)
p_hom = stats.f.sf(F_hom, k - 1, gl_int)
print("\nHomogeneidade das inclinações (interação cultivar x matéria orgânica)")
print(f"  modelo com interação:        SQ do resíduo = {SQ_int:9.2f} em {gl_int} gl")
print(f"  modelo com retas paralelas:  SQ do resíduo = {SQ_par:9.2f} em {gl_par} gl")
print(f"  F({k-1}; {gl_int}) = {F_hom:.4f} | p-valor = {p_hom:.4f}")
incl = [b_int[1], b_int[1] + b_int[4], b_int[1] + b_int[5]]
print("  inclinação estimada em cada cultivar (kg/ha por ponto percentual de MO):")
for i in range(k):
    print(f"    cultivar {rot[i]}: {incl[i]:8.2f}")
print("  as inclinações não diferem: o modelo de retas paralelas pode ser usado.")

# 3. A ANCOVA
QM_res = SQ_par / gl_par
b_cov, SQ_cov, gl_cov = ols(np.column_stack([um, mo]), prod)
SQ_mo_seq = SQ_nul - SQ_cov          # tipo I: covariável primeiro
SQ_cult_aj = SQ_cov - SQ_par         # efeito do cultivar DEPOIS do ajuste
print("\nANCOVA: produtividade = b0 + b1*MO + efeito do cultivar")
print("Fonte                    GL          SQ           QM        F      p-valor")
for nome, gl, sq in [("Matéria orgânica", 1, SQ_mo_seq), ("Cultivar", k-1, SQ_cult_aj)]:
    Fv = (sq / gl) / QM_res
    print(f"{nome:<22} {gl:3d}  {sq:11.2f}  {sq/gl:11.2f}  {Fv:7.4f}  {stats.f.sf(Fv, gl, gl_par):.3e}")
print(f"{'Resíduo':<22} {gl_par:3d}  {SQ_par:11.2f}  {QM_res:11.2f}")
print(f"{'Total':<22} {N-1:3d}  {SQ_nul:11.2f}")
print(f"\nDP do resíduo = {np.sqrt(QM_res):.2f} kg/ha, contra {np.sqrt(SQ_fat/gl_fat):.2f} kg/ha sem a covariável")
print(f"ou seja, o QM do resíduo caiu {100*(1 - QM_res/(SQ_fat/gl_fat)):.1f}%.")
XtXi = np.linalg.inv(X_par.T @ X_par)
ep = np.sqrt(np.diag(XtXi) * QM_res)
nomes_b = ["intercepto (cultivar A)", "matéria orgânica", "B - A", "C - A"]
print("\nCoeficientes")
for j, nm in enumerate(nomes_b):
    t = b_par[j] / ep[j]
    print(f"  {nm:<24} {b_par[j]:10.3f} | EP = {ep[j]:7.3f} | t({gl_par}) = {t:+8.4f} | "
          f"p-valor = {2*stats.t.sf(abs(t), gl_par):.3e}")
print(f"\nInclinação comum = {b_par[1]:.2f} kg/ha por ponto percentual de matéria orgânica")
print(f"IC 95% da inclinação = [{b_par[1]-stats.t.ppf(0.975,gl_par)*ep[1]:.2f}; "
      f"{b_par[1]+stats.t.ppf(0.975,gl_par)*ep[1]:.2f}]")

# 4. Médias ajustadas
mbar = mo.mean()
print(f"\nMédias ajustadas, avaliadas na média geral da covariável ({mbar:.4f}%)")
print("cultivar   média observada   MO média   média ajustada     EP      correção")
aj, ep_aj = [], []
for i in range(k):
    x0 = np.array([1.0, mbar, float(i == 1), float(i == 2)])
    v = float(x0 @ b_par)
    e = float(np.sqrt(x0 @ XtXi @ x0 * QM_res))
    aj.append(v); ep_aj.append(e)
    obs = prod[cult == i].mean()
    print(f"    {rot[i]}         {obs:8.2f}      {mo[cult==i].mean():.4f}      {v:8.2f}     {e:6.2f}   {v-obs:+8.2f}")
aj = np.array(aj)
print("A ordem das médias se INVERTE: a cultivar C, a pior em bruto, é a melhor ajustada.")

# 5. Comparações múltiplas entre médias ajustadas (Tukey)
print("\nComparações entre médias ajustadas (ajuste de Tukey)")
q95 = stats.studentized_range.ppf(0.95, k, gl_par)
for i, j in itertools.combinations(range(k), 2):
    ci = np.array([1.0, mbar, float(i == 1), float(i == 2)])
    cj = np.array([1.0, mbar, float(j == 1), float(j == 2)])
    c = cj - ci
    dif = float(c @ b_par)
    ep_d = float(np.sqrt(c @ XtXi @ c * QM_res))
    q = abs(dif) / (ep_d / np.sqrt(2))
    pa = stats.studentized_range.sf(q, k, gl_par)
    marg = q95 * ep_d / np.sqrt(2)
    print(f"  {rot[j]} - {rot[i]}: {dif:+8.2f} kg/ha | EP = {ep_d:6.2f} | "
          f"IC 95% = [{dif-marg:+8.2f}; {dif+marg:+8.2f}] | p ajustado = {pa:.5f}"
          + ("  *" if pa < 0.05 else ""))

# 6. Pressupostos restantes
res = prod - X_par @ b_par
print(f"\nShapiro-Wilk dos resíduos: W = {stats.shapiro(res)[0]:.4f}, p-valor = {stats.shapiro(res)[1]:.4f}")
print(f"Levene dos resíduos entre cultivares: p-valor = "
      f"{stats.levene(*[res[cult==i] for i in range(k)], center='median')[1]:.4f}")
ajust = X_par @ b_par
print(f"Correlação entre |resíduo| e valor ajustado: r = {np.corrcoef(np.abs(res), ajust)[0,1]:.4f}")
R2 = 1 - SQ_par / SQ_nul
print(f"R² do modelo = {R2:.4f} | R² ajustado = {1-(SQ_par/gl_par)/(SQ_nul/(N-1)):.4f}")
print("Faixa de matéria orgânica em cada cultivar (a sobreposição é o que torna o ajuste legítimo):")
for i in range(k):
    print(f"  cultivar {rot[i]}: [{mo[cult==i].min():.2f}; {mo[cult==i].max():.2f}]")

# 7. Quando o paralelismo FALHA: o caso de mtcars
print("\n" + "="*70)
print("Exemplo em que o pressuposto de paralelismo FALHA: mtcars")
print("Consumo (mpg) por transmissão (am), ajustado pelo peso (wt).")
mpg = np.array([21.0,21.0,22.8,21.4,18.7,18.1,14.3,24.4,22.8,19.2,17.8,16.4,17.3,15.2,
                10.4,10.4,14.7,32.4,30.4,33.9,21.5,15.5,15.2,13.3,19.2,27.3,26.0,30.4,
                15.8,19.7,15.0,21.4])
wt = np.array([2.620,2.875,2.320,3.215,3.440,3.460,3.570,3.190,3.150,3.440,3.440,4.070,
               3.730,3.780,5.250,5.424,5.345,2.200,1.615,1.835,2.465,3.520,3.435,3.840,
               3.845,1.935,2.140,1.513,3.170,2.770,3.570,2.780])
am = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1], dtype=float)
u2 = np.ones(len(mpg))
bi2, SQi2, gli2 = ols(np.column_stack([u2, wt, am, wt*am]), mpg)
bp2, SQp2, glp2 = ols(np.column_stack([u2, wt, am]), mpg)
Fh2 = ((SQp2 - SQi2) / 1) / (SQi2 / gli2)
print(f"\nDiferença bruta (manual - automática) = {mpg[am==1].mean()-mpg[am==0].mean():+.4f} mpg")
print(f"  (é o resultado da ficha do teste t de Welch: p = "
      f"{stats.ttest_ind(mpg[am==1], mpg[am==0], equal_var=False)[1]:.6f})")
print(f"Teste de paralelismo: F(1; {gli2}) = {Fh2:.4f}, p-valor = {stats.f.sf(Fh2, 1, gli2):.4f}")
print(f"  inclinação na automática = {bi2[1]:+.4f} | na manual = {bi2[1]+bi2[3]:+.4f} mpg por mil libras")
print("  as retas NÃO são paralelas: uma única 'diferença ajustada' não descreve os dados.")
print(f"\nSe a ANCOVA fosse aplicada de todo modo, a diferença ajustada seria {bp2[2]:+.4f} mpg,")
print("um número que esconde o que realmente acontece. O correto é descrever a diferença")
print("como função do peso, a partir do modelo com interação:")
for w0 in (1.6, 2.0, 2.4, 2.8, 3.2, 3.6):
    c = np.array([0.0, 0.0, 1.0, w0])
    d = float(c @ bi2)
    Xi2 = np.column_stack([u2, wt, am, wt*am])
    epd = float(np.sqrt(c @ np.linalg.inv(Xi2.T @ Xi2) @ c * SQi2/gli2))
    t = d/epd
    print(f"  peso = {w0:.1f}: diferença manual - automática = {d:+7.3f} mpg | "
          f"EP = {epd:5.3f} | t = {t:+7.3f} | p-valor = {2*stats.t.sf(abs(t), gli2):.4f}")
print("A vantagem do câmbio manual existe nos carros leves e desaparece nos pesados.")
