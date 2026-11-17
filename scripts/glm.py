# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

# Dados: besouros expostos a dissulfeto de carbono (Bliss, 1935)
# ldose = log10 da dose (mg/l) | expostos | mortos
ldose    = np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839])
expostos = np.array([59, 60, 62, 56, 63, 59, 62, 60.])
mortos   = np.array([6, 13, 18, 28, 52, 53, 61, 60.])
X = np.column_stack([np.ones(8), ldose])

ligacoes = {
    "logit":   (lambda e: 1 / (1 + np.exp(-e)),       lambda p: np.log(p / (1 - p))),
    "probit":  (lambda e: stats.norm.cdf(e),          lambda p: stats.norm.ppf(p)),
    "cloglog": (lambda e: 1 - np.exp(-np.exp(e)),     lambda p: np.log(-np.log(1 - p))),
}

def hessiana(f, x, h=1e-5):
    p = len(x); H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            e1 = np.zeros(p); e2 = np.zeros(p); e1[i] = h; e2[j] = h
            H[i, j] = (f(x+e1+e2) - f(x+e1-e2) - f(x-e1+e2) + f(x-e1-e2)) / (4 * h * h)
    return H

def glm_binomial(X, n, y, ligacao, nomes):
    """GLM com resposta binomial agrupada (y sucessos em n ensaios)."""
    inversa = ligacoes[ligacao][0]
    def menos_logv(b):
        p = np.clip(inversa(X @ b), 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(p) + (n - y) * np.log(1 - p))
    res = optimize.minimize(menos_logv, np.array([-40.0, 22.0]), method="BFGS")
    b = res.x; ep = np.sqrt(np.diag(np.linalg.inv(hessiana(menos_logv, b))))
    z = b / ep; pv = 2 * stats.norm.sf(np.abs(z))
    mu = n * inversa(X @ b)
    termo1 = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    termo2 = np.where(n - y > 0, (n - y) * np.log(np.where(n - y > 0, n - y, 1) / (n - mu)), 0.0)
    dev = 2 * np.sum(termo1 + termo2)
    print(f"  {'Termo':<14}{'Estimativa':>11}{'Erro-padrão':>13}{'z':>9}{'p-valor':>11}")
    for nm, bi, ei, zi, pi in zip(nomes, b, ep, z, pv):
        print(f"  {nm:<14}{bi:>11.4f}{ei:>13.4f}{zi:>9.3f}{pi:>11.2e}")
    print(f"  Deviance = {dev:.4f} em {len(y) - X.shape[1]} gl | p-valor do ajuste = {stats.chi2.sf(dev, len(y)-X.shape[1]):.4f}"
          f" | AIC = {2*res.fun + 2*X.shape[1]:.2f}")
    return dict(b=b, ep=ep, dev=dev, logv=-res.fun, mu=mu)

# 1. Dados observados
print("ldose   expostos  mortos  proporção")
for a, b_, c in zip(ldose, expostos, mortos):
    print(f"{a:.4f}  {int(b_):>7}  {int(c):>6}  {c/b_:>9.3f}")

# 2. Três ligações para a mesma família (binomial)
modelos = {}
for lig in ["logit", "probit", "cloglog"]:
    print(f"\nLigação {lig}:")
    modelos[lig] = glm_binomial(X, expostos, mortos, lig, ["(Intercepto)", "ldose"])

# 3. Comparação dos ajustes
print("\nLigação     Deviance      AIC")
for lig, m in modelos.items():
    print(f"{lig:<10}{m['dev']:>10.4f}{2*(-m['logv']) + 4:>9.2f}")

# 4. Dose letal 50% pela melhor ligação
for lig in ["logit", "cloglog"]:
    b = modelos[lig]["b"]
    alvo = ligacoes[lig][1](0.5)
    ld50 = (alvo - b[0]) / b[1]
    print(f"\n{lig}: dose letal 50% em log10 = {ld50:.4f}  ->  {10**ld50:.3f} mg/l")

# 5. Proporções previstas por cada ligação
print("\nldose    observado    logit   probit  cloglog")
for i in range(8):
    linha = [modelos[l]["mu"][i] / expostos[i] for l in ["logit", "probit", "cloglog"]]
    print(f"{ldose[i]:.4f}  {mortos[i]/expostos[i]:>9.3f}  {linha[0]:>7.3f} {linha[1]:>8.3f} {linha[2]:>8.3f}")

# 6. A mesma estrutura em outras famílias: três exemplos de ligação canônica
print("\nFamília          Resposta                Ligação canônica   Variância")
print("Normal           contínua                identidade         constante")
print("Binomial         sucesso/fracasso        logit              mu(1-mu)/n")
print("Poisson          contagem                log                mu")
print("Gama             contínua positiva       inversa            mu²")

# 7. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
grade = np.linspace(1.65, 1.92, 200); Xg = np.column_stack([np.ones(200), grade])
ax[0].scatter(ldose, mortos / expostos, s=40, zorder=3, label="observado")
for lig in ["logit", "probit", "cloglog"]:
    ax[0].plot(grade, ligacoes[lig][0](Xg @ modelos[lig]["b"]), label=lig)
ax[0].set_xlabel("log10 da dose"); ax[0].set_ylabel("Proporção de mortos"); ax[0].legend()
for lig in ["logit", "probit", "cloglog"]:
    ax[1].scatter(ldose, mortos - modelos[lig]["mu"], label=lig)
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_xlabel("log10 da dose"); ax[1].set_ylabel("Observado - previsto (nº de mortos)"); ax[1].legend()
plt.show()
