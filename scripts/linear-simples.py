# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def mqo(X, y, nomes):
    """Mínimos quadrados ordinários: X sem a coluna de 1 (o intercepto é acrescentado aqui)."""
    X = np.column_stack([np.ones(len(y)), X]); n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    ajustado = X @ beta; res = y - ajustado
    sqe = res @ res; sqt = np.sum((y - y.mean())**2)
    s2 = sqe / (n - p); ep = np.sqrt(s2 * np.diag(XtX_inv))
    t = beta / ep; pv = 2 * stats.t.sf(np.abs(t), n - p); tc = stats.t.ppf(0.975, n - p)
    print(f"{'Termo':<12}{'Estimativa':>11}{'Erro-padrão':>13}{'t':>9}{'p-valor':>11}{'IC 95%':>22}")
    for nm, b, e, ti, pi in zip(["(Intercepto)"] + nomes, beta, ep, t, pv):
        print(f"{nm:<12}{b:>11.4f}{e:>13.4f}{ti:>9.3f}{pi:>11.2e}   [{b - tc*e:>8.4f}; {b + tc*e:>8.4f}]")
    r2 = 1 - sqe / sqt; r2a = 1 - (1 - r2) * (n - 1) / (n - p)
    F = ((sqt - sqe) / (p - 1)) / s2
    print(f"Erro-padrão residual = {np.sqrt(s2):.4f} com {n - p} gl | R² = {r2:.4f} | R² ajustado = {r2a:.4f}")
    print(f"F({p - 1}, {n - p}) = {F:.2f}, p-valor = {stats.f.sf(F, p - 1, n - p):.2e}")
    return dict(X=X, beta=beta, ep=ep, res=res, ajustado=ajustado, s2=s2, XtX_inv=XtX_inv, sqe=sqe, sqt=sqt, r2=r2, n=n, p=p)

# Dados do conjunto mtcars: peso (wt, em 1000 libras) e consumo (mpg)
wt  = np.array([2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77, 3.57, 2.78])
mpg = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
n = len(mpg)

# 1. Ajuste do modelo mpg = b0 + b1 * wt
m = mqo(wt, mpg, ["wt"])
b0, b1 = m["beta"]

# 2. Tabela de análise de variância
sqr = m["sqt"] - m["sqe"]
print(f"\nSQ regressão = {sqr:.2f} (1 gl) | SQ resíduos = {m['sqe']:.2f} ({n - 2} gl) | SQ total = {m['sqt']:.2f} ({n - 1} gl)")

# 3. Previsão para um carro de 3000 libras (wt = 3)
x0 = 3.0; y0 = b0 + b1 * x0
sxx = np.sum((wt - wt.mean())**2); s = np.sqrt(m["s2"]); tc = stats.t.ppf(0.975, n - 2)
ep_media = s * np.sqrt(1/n + (x0 - wt.mean())**2 / sxx)
ep_novo  = s * np.sqrt(1 + 1/n + (x0 - wt.mean())**2 / sxx)
print(f"\nPrevisão em wt = 3: {y0:.3f} mpg")
print(f"IC 95% para a média:           [{y0 - tc*ep_media:.3f}; {y0 + tc*ep_media:.3f}]")
print(f"Intervalo de predição de 95%:  [{y0 - tc*ep_novo:.3f}; {y0 + tc*ep_novo:.3f}]")

# Diagnóstico dos resíduos ----
h = np.sum((m["X"] @ m["XtX_inv"]) * m["X"], axis=1)                 # alavancagem (leverage)
r_pad = m["res"] / np.sqrt(m["s2"] * (1 - h))                        # resíduos padronizados
cook = r_pad**2 * h / ((1 - h) * m["p"])                             # distância de Cook
W, pw = stats.shapiro(m["res"])
print(f"\nShapiro-Wilk dos resíduos: W = {W:.4f}, p-valor = {pw:.4f}")
aux = np.linalg.lstsq(m["X"], m["res"]**2, rcond=None)[0]             # Breusch-Pagan (versão de Koenker)
e2 = m["res"]**2; r2_aux = 1 - np.sum((e2 - m["X"] @ aux)**2) / np.sum((e2 - e2.mean())**2)
bp = m["n"] * r2_aux
print(f"Breusch-Pagan: BP = {bp:.4f}, gl = {m['p'] - 1}, p-valor = {stats.chi2.sf(bp, m['p'] - 1):.4f}")
maiores = np.argsort(-cook)[:3]
print("Maiores distâncias de Cook (observação: valor):", ", ".join(f"{i + 1}: {cook[i]:.3f}" for i in maiores))
print("Maiores resíduos padronizados (observação: valor):", ", ".join(f"{i + 1}: {r_pad[i]:.2f}" for i in np.argsort(-np.abs(r_pad))[:3]))

# 4. Exemplo do texto (n = 5)
print("\nExemplo com n = 5:")
m5 = mqo(np.array([1, 2, 3, 4, 5.]), np.array([2, 4, 5, 4, 5.]), ["x"])

# 5. Gráficos: reta ajustada e resíduos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(wt, mpg); xs = np.linspace(wt.min(), wt.max(), 50); ax[0].plot(xs, b0 + b1 * xs, color="red")
ax[0].set_xlabel("Peso (1000 lb)"); ax[0].set_ylabel("Consumo (mpg)")
ax[1].scatter(m["ajustado"], m["res"]); ax[1].axhline(0, color="red")
ax[1].set_xlabel("Valores ajustados"); ax[1].set_ylabel("Resíduos")
plt.show()
