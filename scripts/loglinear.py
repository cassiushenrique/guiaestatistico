# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: UCBAdmissions (R). Admissões à pós-graduação em Berkeley, 1973.
# Três variáveis categóricas: Admissão (A), Sexo (S) e Departamento (D).
# É o mesmo conjunto da ficha do qui-quadrado de independência, aqui tratado
# com modelos, e não com um teste isolado.
dep = ["A", "B", "C", "D", "E", "F"]
tab = {"A": [[512, 313], [ 89,  19]], "B": [[353, 207], [ 17,   8]],
       "C": [[120, 205], [202, 391]], "D": [[138, 279], [131, 244]],
       "E": [[ 53, 138], [ 94, 299]], "F": [[ 22, 351], [ 24, 317]]}
# linhas = sexo (0 homem, 1 mulher); colunas = admissão (0 admitido, 1 rejeitado)

linhas = []
for d in dep:
    for s in (0, 1):
        for a in (0, 1):
            linhas.append((d, s, a, float(tab[d][s][a])))
Dv = np.array([dep.index(l[0]) for l in linhas])
Sv = np.array([l[1] for l in linhas])
Av = np.array([l[2] for l in linhas])
y  = np.array([l[3] for l in linhas])
print(f"{len(y)} células, total de {int(y.sum())} candidaturas")

def indic(v, niveis):
    """Codificação de tratamento: omite o primeiro nível."""
    return np.column_stack([(v == n).astype(float) for n in niveis[1:]])

bD = indic(Dv, list(range(6))); bS = indic(Sv, [0, 1]); bA = indic(Av, [0, 1])
um = np.ones(len(y))[:, None]

def inter(X1, X2):
    return np.column_stack([X1[:, i]*X2[:, j] for i in range(X1.shape[1]) for j in range(X2.shape[1])])

def poisson_glm(X, y, iters=200):
    """Regressão de Poisson com ligação logarítmica, por escore de Fisher (IRLS)."""
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean())
    for _ in range(iters):
        eta = X @ b; mu = np.exp(eta)
        W = mu
        z = eta + (y - mu)/mu
        A = X.T @ (W[:, None]*X); c = X.T @ (W*z)
        bn = np.linalg.solve(A, c)
        if np.max(np.abs(bn - b)) < 1e-12: b = bn; break
        b = bn
    mu = np.exp(X @ b)
    G2 = 2*np.sum(np.where(y > 0, y*np.log(y/mu), 0.0) - (y - mu))
    X2 = np.sum((y - mu)**2/mu)
    gl = len(y) - np.linalg.matrix_rank(X)
    return b, mu, G2, X2, gl

modelos = {
 "(A, S, D)  independência mútua":            np.column_stack([um, bA, bS, bD]),
 "(AD, S)  A e D associados, S independente": np.column_stack([um, bA, bS, bD, inter(bA, bD)]),
 "(SD, A)  S e D associados, A independente": np.column_stack([um, bA, bS, bD, inter(bS, bD)]),
 "(AD, SD)  A e S independentes DADO D":      np.column_stack([um, bA, bS, bD, inter(bA, bD), inter(bS, bD)]),
 "(AD, SD, AS)  associação homogênea":        np.column_stack([um, bA, bS, bD, inter(bA, bD), inter(bS, bD), inter(bA, bS)]),
}
print("\nModelos log-lineares ajustados (G² = desvio; valores pequenos = bom ajuste)")
print(f"{'modelo':<44}{'gl':>4}{'G²':>11}{'p':>12}{'X²':>11}")
res = {}
for nome, X in modelos.items():
    b, mu, G2, X2, gl = poisson_glm(X, y)
    res[nome] = (b, mu, G2, X2, gl)
    print(f"{nome:<44}{gl:4d}{G2:11.3f}{stats.chi2.sf(G2, gl):12.3e}{X2:11.3f}")
print(f"{'saturado':<44}{0:4d}{0.0:11.3f}{'-':>12}{0.0:11.3f}")

# 1. A comparação que responde à pergunta do estudo
G_cond = res["(AD, SD)  A e S independentes DADO D"][2]
gl_cond = res["(AD, SD)  A e S independentes DADO D"][4]
G_hom = res["(AD, SD, AS)  associação homogênea"][2]
gl_hom = res["(AD, SD, AS)  associação homogênea"][4]
print(f"\nTeste da associação sexo-admissão, controlando o departamento:")
print(f"  compara (AD, SD) com (AD, SD, AS)")
print(f"  diferença de desvios = {G_cond:.4f} - {G_hom:.4f} = {G_cond-G_hom:.4f}")
print(f"  gl = {gl_cond} - {gl_hom} = {gl_cond-gl_hom} | p-valor = "
      f"{stats.chi2.sf(G_cond-G_hom, gl_cond-gl_hom):.4f}")
print("  Não há evidência de associação entre sexo e admissão, dado o departamento.")
coef_AS = res["(AD, SD, AS)  associação homogênea"][0][-1]
print(f"  Coeficiente do termo A:S no modelo homogêneo = {coef_AS:+.4f}")
print(f"  exp(coeficiente) = {np.exp(coef_AS):.4f}: é a razão de chances comum,")
print("  muito próxima da de Mantel-Haenszel (0,9047) da ficha do qui-quadrado.")

# 2. Por que a tabela agregada engana: o modelo que a reproduz
print("\nA tabela AGREGADA corresponde a somar sobre D, isto é, a ignorar o")
print("departamento. O modelo que descreve a tabela agregada é (AS):")
agreg = np.zeros((2, 2))
for d in dep: agreg += np.array(tab[d], float)
qui_ag, p_ag, _, _ = stats.chi2_contingency(agreg, correction=False)
print(f"  qui-quadrado da tabela agregada = {qui_ag:.4f}, p = {p_ag:.3e}")
print("  Colapsar uma tabela de três vias só é legítimo quando a variável")
print("  colapsada é independente de ao menos uma das outras duas. Aqui D está")
print("  associado tanto a S quanto a A, e por isso o colapso distorce.")
b_ad, _, G_ad, _, gl_ad = res["(AD, S)  A e D associados, S independente"]
b_sd, _, G_sd, _, gl_sd = res["(SD, A)  S e D associados, A independente"]
print(f"  Evidência: o modelo (AD, S) tem G² = {G_ad:.1f} e o (SD, A), G² = {G_sd:.1f}.")
print("  Os dois rejeitam, confirmando que D está associado às duas variáveis.")

# 3. Resíduos do modelo (AD, SD): onde o ajuste falha
b, mu, G2, X2, gl = res["(AD, SD)  A e S independentes DADO D"]
res_pad = (y - mu)/np.sqrt(mu)
print(f"\nResíduos de Pearson do modelo (AD, SD), por célula")
print(f"{'dep':<5}{'sexo':<10}{'situação':<12}{'obs.':>7}{'ajustado':>11}{'resíduo':>10}")
for idx, (d, s, a, o) in enumerate(linhas):
    if abs(res_pad[idx]) > 1.0:
        print(f"{d:<5}{('homem' if s==0 else 'mulher'):<10}"
              f"{('admitido' if a==0 else 'rejeitado'):<12}{o:7.0f}{mu[idx]:11.2f}{res_pad[idx]:10.3f}")
print("Só o departamento A apresenta resíduos relevantes: é ali que o modelo de")
print("independência condicional não descreve bem os dados.")

# 4. O modelo que acomoda o departamento A
bAS_D = np.column_stack([um, bA, bS, bD, inter(bA, bD), inter(bS, bD),
                         inter(bA, bS), inter(inter(bA, bS), bD)])
b4, mu4, G4, X4, gl4 = poisson_glm(bAS_D, y)
print(f"\nModelo saturado (AD, SD, AS, ASD): G² = {G4:.6f}, gl = {gl4}")
dep_a = np.array(tab["A"], float)
print(f"Departamento A isolado: qui² = {stats.chi2_contingency(dep_a, correction=False)[0]:.4f}, "
      f"p = {stats.chi2_contingency(dep_a, correction=False)[1]:.3e}")
print(f"Razão de chances do departamento A = "
      f"{dep_a[0,0]*dep_a[1,1]/(dep_a[0,1]*dep_a[1,0]):.4f}  (favorece as mulheres)")

# 5. Seleção de modelos por AIC e BIC
print("\nSeleção de modelos")
print(f"{'modelo':<44}{'gl':>4}{'G²':>10}{'AIC':>11}{'BIC':>11}")
nobs = len(y)
for nome, X in modelos.items():
    b_, mu_, G2_, X2_, gl_ = res[nome]
    p_ = len(b_)
    logL = np.sum(y*np.log(mu_) - mu_ - np.array([np.sum(np.log(np.arange(1, int(v)+1))) for v in y]))
    print(f"{nome:<44}{gl_:4d}{G2_:10.2f}{-2*logL+2*p_:11.2f}{-2*logL+np.log(nobs)*p_:11.2f}")
logL4 = np.sum(y*np.log(mu4) - mu4 - np.array([np.sum(np.log(np.arange(1, int(v)+1))) for v in y]))
print(f"{'saturado (AD, SD, AS, ASD)':<44}{gl4:4d}{G4:10.2f}"
      f"{-2*logL4+2*len(b4):11.2f}{-2*logL4+np.log(nobs)*len(b4):11.2f}")
print("\nO AIC prefere o saturado, por causa do departamento A; o BIC, que penaliza")
print("mais a complexidade, prefere a independência condicional (AD, SD).")
