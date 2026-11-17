# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

def polr(X, y, K, nomes):
    """Modelo de chances proporcionais (logit cumulativo) por máxima verossimilhança."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    n, p = X.shape
    def corta(par):
        beta = par[:p]; z = np.r_[par[p], par[p] + np.cumsum(np.exp(par[p+1:]))]
        return beta, z
    def menos_logv(par):
        beta, z = corta(par)
        eta = z[None, :] - (X @ beta)[:, None]
        F = 1 / (1 + np.exp(-eta))
        P = np.diff(np.c_[np.zeros(n), F, np.ones(n)], axis=1)
        return -np.sum(np.log(np.clip(P[np.arange(n), y], 1e-12, None)))
    res = optimize.minimize(menos_logv, np.r_[np.zeros(p), np.linspace(0, 1, K - 1)], method="BFGS")
    H = hessiana(menos_logv, res.x); ep_full = np.sqrt(np.diag(np.linalg.inv(H)))
    beta, z = corta(res.x)
    print(f"{'Termo':<16}{'Estimativa':>11}{'Erro-padrão':>13}{'t':>8}{'OR':>9}")
    for j, nm in enumerate(nomes):
        print(f"{nm:<16}{beta[j]:>11.4f}{ep_full[j]:>13.4f}{beta[j]/ep_full[j]:>8.3f}{np.exp(beta[j]):>9.3f}")
    for j, zz in enumerate(z):
        print(f"{'Limiar ' + str(j+1) + '|' + str(j+2):<16}{zz:>11.4f}")
    return dict(beta=beta, z=z, ep=ep_full[:p], logv=-res.fun, npar=len(res.x))

def hessiana(f, x, h=1e-4):
    p = len(x); H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            e1 = np.zeros(p); e2 = np.zeros(p); e1[i] = h; e2[j] = h
            H[i, j] = (f(x+e1+e2) - f(x+e1-e2) - f(x-e1+e2) + f(x-e1-e2)) / (4 * h * h)
    return H

# 1. Dados: mtcars; resposta ordinal = número de cilindros (4 < 6 < 8), preditor = peso
wt  = np.array([2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77, 3.57, 2.78])
cyl = np.array([6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4])
y = np.searchsorted([5, 7], cyl)          # 0 = 4 cil., 1 = 6 cil., 2 = 8 cil.
n = len(y)
for j, nome in enumerate(["4 cilindros", "6 cilindros", "8 cilindros"]):
    print(f"{nome}: n = {np.sum(y == j)} | peso médio = {wt[y == j].mean():.3f}")

# 2. Ajuste do modelo de chances proporcionais
m = polr(wt, y, 3, ["peso (wt)"])
print(f"\nLog-verossimilhança = {m['logv']:.4f} | AIC = {-2 * m['logv'] + 2 * m['npar']:.3f}")

# 3. Teste da razão de verossimilhanças contra o modelo nulo
props = np.bincount(y) / n
logv0 = np.sum(np.bincount(y) * np.log(props))
G = 2 * (m["logv"] - logv0)
print(f"Modelo nulo: {logv0:.4f} | G = {G:.3f}, gl = 1, p-valor = {stats.chi2.sf(G, 1):.3e}")
print(f"Pseudo-R² de McFadden = {1 - m['logv'] / logv0:.4f}")

# 4. Razão de chances e intervalo de confiança
b, ep = m["beta"][0], m["ep"][0]; zc = stats.norm.ppf(0.975)
print(f"\nOR por +1000 lb = {np.exp(b):.3f}, IC 95% = [{np.exp(b - zc*ep):.3f}; {np.exp(b + zc*ep):.3f}]")
print(f"OR por +500 lb  = {np.exp(0.5*b):.3f}")

# 5. Probabilidades estimadas por categoria
print("\n Peso    P(4 cil.)  P(6 cil.)  P(8 cil.)")
for w in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    F = 1 / (1 + np.exp(-(m["z"] - m["beta"][0] * w)))
    P = np.diff(np.r_[0, F, 1])
    print(f"{w:5.1f}  {P[0]:>9.3f}  {P[1]:>9.3f}  {P[2]:>9.3f}")

# 6. Verificação da hipótese de chances proporcionais:
#    duas regressões logísticas binárias com cortes diferentes devem ter inclinações parecidas
def logistica(x, yb):
    Xd = np.column_stack([np.ones(len(x)), x]); b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xd @ b)); W = p * (1 - p)
        b = b + np.linalg.solve(Xd.T @ (Xd * W[:, None]), Xd.T @ (yb - p))
    return b
b1 = logistica(wt, (y >= 1).astype(float))     # 6 ou 8 cilindros contra 4
b2 = logistica(wt, (y >= 2).astype(float))     # 8 cilindros contra 4 ou 6
print(f"\nInclinação do corte 'y >= 6 cil.' = {b1[1]:.4f} | do corte 'y >= 8 cil.' = {b2[1]:.4f}")
print(f"Modelo ordinal (inclinação única)   = {m['beta'][0]:.4f}")

# 7. Teste da hipótese de chances proporcionais (razão de verossimilhanças contra o multinomial)
Xd = np.column_stack([np.ones(n), wt])
def menos_logv_mult(b):
    B = np.vstack([np.zeros(2), b.reshape(2, 2)]); eta = Xd @ B.T
    mx = eta.max(axis=1, keepdims=True)
    return -np.sum(eta[np.arange(n), y] - mx.ravel() - np.log(np.exp(eta - mx).sum(axis=1)))
rm = optimize.minimize(menos_logv_mult, np.zeros(4), method="BFGS")
G2 = 2 * (-rm.fun - m["logv"])
print(f"Multinomial (inclinações livres): log-verossimilhança = {-rm.fun:.4f}")
print(f"G = {G2:.3f}, gl = 1, p-valor = {stats.chi2.sf(G2, 1):.4f}")

# 8. Gráfico: probabilidades estimadas por categoria
grade = np.linspace(1.4, 5.6, 200)
F = 1 / (1 + np.exp(-(m["z"][None, :] - m["beta"][0] * grade[:, None])))
P = np.diff(np.c_[np.zeros(200), F, np.ones(200)], axis=1)
for j, nome in enumerate(["4 cilindros", "6 cilindros", "8 cilindros"]): plt.plot(grade, P[:, j], label=nome)
plt.xlabel("Peso (1000 lb)"); plt.ylabel("Probabilidade estimada"); plt.legend(); plt.show()
