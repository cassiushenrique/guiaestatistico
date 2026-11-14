# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def logistica(X, y, nomes):
    """Regressão logística por máxima verossimilhança (mínimos quadrados ponderados iterativos)."""
    X = np.column_stack([np.ones(len(y)), X]); beta = np.zeros(X.shape[1])
    for it in range(50):
        p = 1 / (1 + np.exp(-X @ beta)); W = p * (1 - p)
        passo = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
        beta = beta + passo
        if np.max(np.abs(passo)) < 1e-10: break
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None])); ep = np.sqrt(np.diag(cov))
    z = beta / ep; pv = 2 * stats.norm.sf(np.abs(z))
    print(f"{'Termo':<13}{'Estimativa':>11}{'Erro-padrão':>13}{'z':>9}{'p-valor':>10}")
    for nm, b, e, zi, pi in zip(["(Intercepto)"] + nomes, beta, ep, z, pv):
        print(f"{nm:<13}{b:>11.4f}{e:>13.4f}{zi:>9.3f}{pi:>10.4f}")
    logv = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return dict(X=X, beta=beta, ep=ep, cov=cov, p=p, logv=logv, iteracoes=it + 1)

# 1. Dados: 23 lançamentos do ônibus espacial antes do acidente da Challenger (Dalal et al., 1989)
temp = np.array([66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58.])   # temperatura (°F)
falha = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1.])                          # dano em anel de vedação
n = len(falha)
print(f"n = {n} | lançamentos com dano = {int(falha.sum())} | temperatura média: com dano = {temp[falha == 1].mean():.1f}, sem dano = {temp[falha == 0].mean():.1f}")

# 2. Ajuste do modelo logit(p) = b0 + b1 * temperatura
m = logistica(temp, falha, ["temperatura"])
b0, b1 = m["beta"]
print("Iterações até a convergência:", m["iteracoes"])

# 3. Razão de chances (odds ratio) com IC de 95% de Wald
zc = stats.norm.ppf(0.975)
print(f"\nOR por 1 °F = {np.exp(b1):.4f}, IC 95% = [{np.exp(b1 - zc*m['ep'][1]):.4f}; {np.exp(b1 + zc*m['ep'][1]):.4f}]")
print(f"OR por 10 °F = {np.exp(10 * b1):.4f}, IC 95% = [{np.exp(10*(b1 - zc*m['ep'][1])):.4f}; {np.exp(10*(b1 + zc*m['ep'][1])):.4f}]")

# 4. Deviance, teste da razão de verossimilhanças, AIC e pseudo-R² de McFadden
p_nulo = falha.mean(); logv0 = n * (p_nulo * np.log(p_nulo) + (1 - p_nulo) * np.log(1 - p_nulo))
dev_nula, dev_res = -2 * logv0, -2 * m["logv"]
G = dev_nula - dev_res
print(f"\nDeviance nula = {dev_nula:.3f} ({n - 1} gl) | deviance residual = {dev_res:.3f} ({n - 2} gl) | AIC = {dev_res + 4:.3f}")
print(f"Teste da razão de verossimilhanças: G = {G:.3f}, gl = 1, p-valor = {stats.chi2.sf(G, 1):.4f}")
print(f"Pseudo-R² de McFadden = {1 - m['logv'] / logv0:.4f}")

# 5. Probabilidades estimadas, com IC de 95% (calculado na escala logit)
print("\nTemperatura  Probabilidade   IC 95%")
for t0 in [81, 70, 65, 53, 31]:
    x0 = np.array([1, t0]); eta = x0 @ m["beta"]; ep_eta = np.sqrt(x0 @ m["cov"] @ x0)
    inv = lambda v: 1 / (1 + np.exp(-v))
    print(f"{t0:>8} °F {inv(eta):>12.4f}   [{inv(eta - zc*ep_eta):.4f}; {inv(eta + zc*ep_eta):.4f}]")
print(f"Temperatura em que a probabilidade é 50%: {-b0 / b1:.1f} °F")

# 6. Classificação com ponto de corte 0,5 e área sob a curva ROC
pred = (m["p"] >= 0.5).astype(int)
vp = int(np.sum((pred == 1) & (falha == 1))); fn = int(np.sum((pred == 0) & (falha == 1)))
fp = int(np.sum((pred == 1) & (falha == 0))); vn = int(np.sum((pred == 0) & (falha == 0)))
print(f"\nMatriz de confusão: VP = {vp}, FN = {fn}, FP = {fp}, VN = {vn}")
print(f"Acurácia = {(vp + vn) / n:.3f} | sensibilidade = {vp / (vp + fn):.3f} | especificidade = {vn / (vn + fp):.3f}")
U = stats.mannwhitneyu(m["p"][falha == 1], m["p"][falha == 0]).statistic
print(f"AUC = {U / (falha.sum() * (n - falha.sum())):.4f}")

# 7. Gráfico: dados e curva ajustada
ts = np.linspace(30, 85, 200)
plt.scatter(temp, falha); plt.plot(ts, 1 / (1 + np.exp(-(b0 + b1 * ts))), color="red")
plt.xlabel("Temperatura (°F)"); plt.ylabel("Probabilidade de dano"); plt.show()
