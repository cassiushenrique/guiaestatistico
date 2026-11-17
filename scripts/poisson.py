# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
from scipy.special import gammaln

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
print(f"n = {n} | média = {quebras.mean():.3f} | variância = {quebras.var(ddof=1):.3f} | razão variância/média = {quebras.var(ddof=1)/quebras.mean():.3f}")
print("\nMédia e variância por combinação:")
for w, nw in [(0, "A"), (1, "B")]:
    for t, nt in [(0, "baixa"), (1, "media"), (2, "alta")]:
        g = quebras[(la == w) & (tensao == t)]
        print(f"  la {nw}, tensao {nt:<6} n = {len(g)} | média = {g.mean():6.3f} | variância = {g.var(ddof=1):7.3f}")

# 1. Ajuste do modelo de Poisson com ligação log
m = glm_poisson(X, quebras, nomes)

# 2. Deviance, AIC e teste global
mu0 = quebras.mean()
dev  = 2 * np.sum(np.where(quebras > 0, quebras * np.log(quebras / m["mu"]), 0) - (quebras - m["mu"]))
dev0 = 2 * np.sum(np.where(quebras > 0, quebras * np.log(quebras / mu0), 0) - (quebras - mu0))
print(f"\nDeviance nula = {dev0:.3f} ({n-1} gl) | deviance residual = {dev:.3f} ({n-m['p']} gl) | AIC = {-2*m['logv'] + 2*m['p']:.2f}")
print(f"Teste global: G = {dev0 - dev:.3f}, gl = {m['p']-1}, p-valor = {stats.chi2.sf(dev0-dev, m['p']-1):.3e}")

# 3. Diagnóstico de superdispersão
X2 = np.sum((quebras - m["mu"])**2 / m["mu"])
phi = X2 / (n - m["p"])
print(f"\nQui-quadrado de Pearson = {X2:.3f} | phi = X2 / gl = {phi:.3f} (esperado proximo de 1 sob Poisson)")
print(f"Deviance / gl = {dev / (n - m['p']):.3f} | p-valor do ajuste = {stats.chi2.sf(dev, n - m['p']):.3e}")

# 4. Taxas estimadas (contagens médias previstas)
print("\nContagem média prevista por combinação:")
for w, nw in [(0, "A"), (1, "B")]:
    for t, nt in [(0, "baixa"), (1, "media"), (2, "alta")]:
        x0 = np.array([1, w, t == 1, t == 2], dtype=float)
        print(f"  la {nw}, tensao {nt:<6} {np.exp(x0 @ m['b']):6.3f}")

# 5. Erros-padrão robustos (quasi-Poisson: multiplicam o erro-padrão por raiz de phi)
print("\nCorreção quase-Poisson (erro-padrão multiplicado por raiz de phi):")
for nm, bi, ei in zip(nomes, m["b"], m["ep"]):
    eq = ei * np.sqrt(phi)
    print(f"  {nm:<14} erro-padrão {ei:.4f} -> {eq:.4f} | t = {bi/eq:6.3f} | p-valor = {2*stats.t.sf(abs(bi/eq), n-m['p']):.4f}")

# 6. Exemplo numérico do texto: taxa prevista para la A e tensao alta
x0 = np.array([1, 0, 0, 1.])
print(f"\nla A e tensao alta: preditor linear = {x0 @ m['b']:.4f} | taxa = exp(...) = {np.exp(x0 @ m['b']):.3f}")

# 7. Gráficos: contagens observadas e resíduos
res_p = (quebras - m["mu"]) / np.sqrt(m["mu"])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].boxplot([quebras[(la == w) & (tensao == t)] for w in (0, 1) for t in (0, 1, 2)])
ax[0].set_xticks(range(1, 7)); ax[0].set_xticklabels(["A-L", "A-M", "A-H", "B-L", "B-M", "B-H"])
ax[0].set_ylabel("Rupturas")
ax[1].scatter(m["mu"], res_p); ax[1].axhline(0, color="red")
ax[1].set_xlabel("Contagem prevista"); ax[1].set_ylabel("Resíduo de Pearson")
plt.show()
