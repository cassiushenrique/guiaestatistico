# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

def rq(X, y, tau):
    """Regressão quantílica por programação linear:
    minimiza a soma de tau*|resíduo positivo| + (1-tau)*|resíduo negativo|."""
    n, p = X.shape
    custo = np.r_[np.zeros(2 * p), tau * np.ones(n), (1 - tau) * np.ones(n)]
    A = np.c_[X, -X, np.eye(n), -np.eye(n)]
    res = optimize.linprog(custo, A_eq=A, b_eq=y, bounds=[(0, None)] * (2 * p + 2 * n), method="highs")
    return res.x[:p] - res.x[p:2 * p]

def rq_boot(X, y, tau, B=1000, semente=2026):
    """Erro-padrão e IC de 95% por bootstrap de pares."""
    rng = np.random.default_rng(semente); n = len(y)
    saida = np.array([rq(X[i := rng.integers(0, n, n)], y[i], tau) for _ in range(B)])
    return saida.std(axis=0, ddof=1), np.percentile(saida, [2.5, 97.5], axis=0)

# 1. Dados: 60 domicílios; renda mensal e gasto com alimentação, em R$ mil (dados fictícios)
renda = np.array([1.2, 1.4, 1.4, 1.7, 1.7, 1.8, 1.9, 1.9, 2.1, 2.8, 3.4, 3.7, 3.7, 3.7, 3.8,
                  3.8, 4.1, 4.3, 4.6, 4.7, 4.7, 5.2, 5.3, 5.4, 5.5, 5.6, 5.9, 5.9, 6.0, 6.5,
                  6.6, 6.7, 6.7, 6.8, 7.2, 7.2, 8.2, 8.5, 8.5, 8.5, 8.7, 8.8, 9.2, 9.8, 9.8,
                  9.9, 9.9, 9.9, 10.3, 10.3, 10.6, 10.6, 10.7, 10.9, 10.9, 11.3, 11.3, 11.5, 11.7, 12.0])
gasto = np.array([0.74, 0.73, 0.57, 0.56, 0.66, 0.87, 0.95, 0.89, 1.01, 1.27, 1.07, 0.92, 1.76, 0.95, 0.62,
                  1.16, 1.25, 1.36, 1.75, 0.94, 1.21, 1.63, 1.32, 1.66, 1.24, 1.58, 1.68, 2.35, 2.28, 2.24,
                  0.83, 1.46, 1.24, 2.19, 1.72, 0.65, 0.41, 2.29, 1.92, 1.56, 1.32, 1.69, 2.47, 1.80, 2.46,
                  1.19, 2.88, 1.91, 2.04, 2.30, 2.12, 2.22, 1.33, 3.18, 1.64, 1.52, 2.37, 1.76, 2.66, 2.76])
n = len(renda); X = np.column_stack([np.ones(n), renda])
print(f"n = {n} | renda de {renda.min()} a {renda.max()} mil | gasto de {gasto.min()} a {gasto.max()} mil")

# 2. Mínimos quadrados (referência): modela a média
b_mq = np.linalg.solve(X.T @ X, X.T @ gasto)
res = gasto - X @ b_mq; s2 = res @ res / (n - 2)
ep_mq = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
print(f"\nMínimos quadrados: intercepto = {b_mq[0]:.4f} ({ep_mq[0]:.4f}) | inclinação = {b_mq[1]:.4f} ({ep_mq[1]:.4f})")

# 3. Heterocedasticidade: variabilidade cresce com a renda
metade = renda <= np.median(renda)
print(f"DP dos resíduos: rendas baixas = {res[metade].std(ddof=1):.4f} | rendas altas = {res[~metade].std(ddof=1):.4f}")

# 4. Regressão quantílica em cinco quantis
print("\n  tau   intercepto   inclinação   EP (bootstrap)        IC 95% da inclinação")
resultados = {}
for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
    b = rq(X, gasto, tau); ep, ic = rq_boot(X, gasto, tau)
    resultados[tau] = (b, ep, ic)
    print(f"{tau:5.2f}   {b[0]:>9.4f}   {b[1]:>10.4f}   {ep[1]:>13.4f}        [{ic[0,1]:.4f}; {ic[1,1]:.4f}]")

# 5. Teste informal: as inclinações de tau = 0,10 e tau = 0,90 diferem?
b10, b90 = resultados[0.10][0][1], resultados[0.90][0][1]
rng = np.random.default_rng(7)
dif = np.array([(lambda i: rq(X[i], gasto[i], 0.90)[1] - rq(X[i], gasto[i], 0.10)[1])(rng.integers(0, n, n)) for _ in range(500)])
print(f"\nDiferença entre as inclinações (0,90 − 0,10) = {b90 - b10:.4f}")
print(f"IC 95% por bootstrap = [{np.percentile(dif, 2.5):.4f}; {np.percentile(dif, 97.5):.4f}]")

# 6. Mediana comparada à média: efeito de um valor extremo
gasto2 = gasto.copy(); gasto2[59] = 9.00        # erro de digitação na última observação
b_mq2 = np.linalg.solve(X.T @ X, X.T @ gasto2)
print(f"\nCom um valor atípico (última observação: 2,76 -> 9,00 mil):")
print(f"  mínimos quadrados: inclinação {b_mq[1]:.4f} -> {b_mq2[1]:.4f}")
print(f"  mediana (tau = 0,50): inclinação {resultados[0.50][0][1]:.4f} -> {rq(X, gasto2, 0.50)[1]:.4f}")

# 7. Previsão: gasto em cada quantil para uma renda de R$ 8 mil
print("\nRenda de R$ 8 mil: gasto previsto em cada quantil")
for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
    b = resultados[tau][0]
    print(f"  tau = {tau:.2f}: R$ {b[0] + b[1] * 8:.3f} mil")

# 8. Gráfico: retas quantílicas
plt.scatter(renda, gasto, color="gray")
grade = np.linspace(1, 12.5, 50)
for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
    b = resultados[tau][0]
    plt.plot(grade, b[0] + b[1] * grade, label=f"tau = {tau:.2f}")
plt.plot(grade, b_mq[0] + b_mq[1] * grade, "k--", label="mínimos quadrados")
plt.xlabel("Renda mensal (R$ mil)"); plt.ylabel("Gasto com alimentação (R$ mil)"); plt.legend()
plt.show()
