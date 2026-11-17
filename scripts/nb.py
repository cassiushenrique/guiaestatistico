# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma

def glm_poisson(X, y, nomes, theta=None):
    """GLM com ligação log por mínimos quadrados ponderados iterativos.
    theta = None: família Poisson. theta numérico: binomial negativa com esse parâmetro."""
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean())
    for _ in range(100):
        mu = np.exp(X @ b)
        W = mu if theta is None else mu / (1 + mu / theta)
        z = X @ b + (y - mu) / mu
        b_novo = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * z))
        if np.max(np.abs(b_novo - b)) < 1e-10: b = b_novo; break
        b = b_novo
    mu = np.exp(X @ b)
    W = mu if theta is None else mu / (1 + mu / theta)
    cov = np.linalg.inv(X.T @ (X * W[:, None])); ep = np.sqrt(np.diag(cov))
    zv = b / ep; pv = 2 * stats.norm.sf(np.abs(zv)); zc = stats.norm.ppf(0.975)
    print(f"{'Termo':<14}{'Estimativa':>11}{'Erro-padrão':>13}{'z':>8}{'p-valor':>11}{'exp(b)':>9}{'IC 95% de exp(b)':>24}")
    for nm, bi, ei, zi, pi in zip(nomes, b, ep, zv, pv):
        print(f"{nm:<14}{bi:>11.4f}{ei:>13.4f}{zi:>8.3f}{pi:>11.2e}{np.exp(bi):>9.3f}   [{np.exp(bi-zc*ei):>7.3f}; {np.exp(bi+zc*ei):>7.3f}]")
    if theta is None:
        logv = np.sum(y * np.log(mu) - mu - gammaln(y + 1))
    else:
        logv = np.sum(gammaln(y + theta) - gammaln(theta) - gammaln(y + 1)
                      + theta * np.log(theta / (theta + mu)) + y * np.log(mu / (theta + mu)))
    return dict(b=b, ep=ep, mu=mu, cov=cov, logv=logv, p=X.shape[1])

def estima_theta(X, y, tol=1e-8):
    """Alterna entre estimar os coeficientes (dado theta) e theta (dados os coeficientes)."""
    theta = 1.0
    for _ in range(50):
        b = np.zeros(X.shape[1]); b[0] = np.log(y.mean())
        for _ in range(100):
            mu = np.exp(X @ b); W = mu / (1 + mu / theta)
            z = X @ b + (y - mu) / mu
            b = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * z))
        mu = np.exp(X @ b)
        f = lambda th: -np.sum(gammaln(y + th) - gammaln(th) - gammaln(y + 1)
                               + th * np.log(th / (th + mu)) + y * np.log(mu / (th + mu)))
        novo = optimize.minimize_scalar(f, bounds=(0.05, 500), method="bounded",
                                        options={"xatol": 1e-9}).x
        if abs(novo - theta) < tol: theta = novo; break
        theta = novo
    # Erro-padrão de theta pela segunda derivada numérica da log-verossimilhança
    h = 1e-4 * theta
    d2 = (f(theta + h) - 2 * f(theta) + f(theta - h)) / h**2
    return theta, 1 / np.sqrt(d2)

# Dados do conjunto warpbreaks: 54 teares; número de rupturas do fio
quebras = np.array([26, 30, 54, 25, 70, 52, 51, 26, 67, 18, 21, 29, 17, 12, 18, 35, 30, 36,
                    36, 21, 24, 18, 10, 43, 28, 15, 26, 27, 14, 29, 19, 29, 31, 41, 20, 44,
                    42, 26, 19, 16, 39, 28, 21, 39, 29, 20, 21, 24, 17, 13, 15, 15, 16, 28])
la      = np.repeat([0, 1], 27)                     # tipo de lã: 0 = A, 1 = B
tensao  = np.tile(np.repeat([0, 1, 2], 9), 2)       # tensão: 0 = baixa (L), 1 = média (M), 2 = alta (H)
# Matriz do modelo: intercepto, lã B, tensão média, tensão alta (níveis de referência: lã A e tensão baixa)
X = np.column_stack([np.ones(54), la, (tensao == 1) * 1.0, (tensao == 2) * 1.0])
nomes = ["(Intercepto)", "la B", "tensao M", "tensao H"]

n = len(quebras)

# 1. Superdispersão: na Poisson, variância = média
print(f"Média = {quebras.mean():.3f} | variância = {quebras.var(ddof=1):.3f} | razão = {quebras.var(ddof=1)/quebras.mean():.3f}")

# 2. Modelo de Poisson (referência)
print("\nModelo de Poisson:")
mp = glm_poisson(X, quebras, nomes)
X2 = np.sum((quebras - mp["mu"])**2 / mp["mu"])
print(f"Qui-quadrado de Pearson / gl = {X2 / (n - 4):.3f} | AIC = {-2*mp['logv'] + 2*4:.2f}")

# 3. Modelo binomial negativa
theta, ep_theta = estima_theta(X, quebras)
print(f"\nModelo binomial negativa (theta estimado = {theta:.4f}, erro-padrão = {ep_theta:.4f}):")
mn = glm_poisson(X, quebras, nomes, theta=theta)
print(f"AIC = {-2*mn['logv'] + 2*5:.2f} (um parâmetro a mais: theta)")

# 4. Comparação dos dois modelos pela razão de verossimilhanças
G = 2 * (mn["logv"] - mp["logv"])
print(f"\nlog-verossimilhança: Poisson = {mp['logv']:.3f} | binomial negativa = {mn['logv']:.3f}")
print(f"G = {G:.3f}; como theta = 0 está na fronteira do espaço de parâmetros,")
print(f"o valor-p é metade do da qui-quadrado com 1 gl: p = {0.5 * stats.chi2.sf(G, 1):.3e}")

# 5. Variância implícita pelos dois modelos
print("\nPrevisto    var (Poisson)   var (binomial negativa)")
for mu0 in [15, 25, 35, 45]:
    print(f"{mu0:>8}  {mu0:>13}   {mu0 + mu0**2 / theta:>22.1f}")

# 6. Efeito sobre os erros-padrão
print("\nTermo           EP Poisson   EP binomial negativa   razão")
for nm, e1, e2 in zip(nomes, mp["ep"], mn["ep"]):
    print(f"{nm:<14}{e1:>11.4f}{e2:>23.4f}{e2/e1:>8.2f}")

# 7. Exemplo numérico do texto: variância prevista para mu = 25
print(f"\nPara mu = 25: variância Poisson = 25 | binomial negativa = 25 + 25²/{theta:.3f} = {25 + 625/theta:.2f}")

# 8. Gráfico: contagens observadas e variância pelos dois modelos
mu_grid = np.linspace(5, 60, 100)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(quebras, bins=12, color="lightgray", edgecolor="white")
ax[0].set_xlabel("Rupturas"); ax[0].set_ylabel("Frequência")
ax[1].plot(mu_grid, mu_grid, label="Poisson")
ax[1].plot(mu_grid, mu_grid + mu_grid**2 / theta, label="Binomial negativa")
ax[1].set_xlabel("Média prevista"); ax[1].set_ylabel("Variância"); ax[1].legend()
plt.show()
