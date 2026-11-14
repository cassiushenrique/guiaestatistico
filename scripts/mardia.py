# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def mardia(X):
    """Teste de Mardia: assimetria e curtose multivariadas."""
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n                                   # covariância com divisor n
    D = Xc @ np.linalg.inv(S) @ Xc.T                    # produtos de Mahalanobis
    b1 = np.sum(D**3) / n**2                            # assimetria multivariada
    b2 = np.sum(np.diag(D)**2) / n                      # curtose multivariada
    est_a = n * b1 / 6; gl = p * (p + 1) * (p + 2) // 6
    p_a = stats.chi2.sf(est_a, gl)
    z_c = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    p_c = 2 * stats.norm.sf(abs(z_c))
    return b1, b2, est_a, gl, p_a, z_c, p_c, np.diag(D)

# 1. Dados: 50 flores da espécie setosa, 4 medidas (cm)
iris = load_iris()
X = iris.data[iris.target == 0]
n, p = X.shape
print("n =", n, "| p =", p, "| médias:", np.round(X.mean(axis=0), 3))

# 2. Teste de Mardia
b1, b2, est_a, gl, p_a, z_c, p_c, d2 = mardia(X)
print(f"\nAssimetria: b1 = {b1:.4f} | estatística = {est_a:.4f}, gl = {gl}, p-valor = {p_a:.4f}")
print(f"Curtose:    b2 = {b2:.4f} (esperado {p*(p+2)}) | z = {z_c:.4f}, p-valor = {p_c:.4f}")

# 3. Normalidade de cada variável (Shapiro-Wilk)
for nome, col in zip(["Comp. sépala", "Larg. sépala", "Comp. pétala", "Larg. pétala"], X.T):
    W, pw = stats.shapiro(col)
    print(f"{nome:<13} W = {W:.4f}, p-valor = {pw:.4f}")

# 4. As 150 flores juntas (mistura de três espécies)
r = mardia(iris.data)
print(f"\nTodas as espécies: assimetria = {r[2]:.2f} (p = {r[4]:.2e}) | curtose z = {r[5]:.3f} (p = {r[6]:.4f})")

# 5. Exemplo do texto: n = 4, p = 2
Y = np.array([[1, 2], [2, 4], [4, 3], [5, 7]], dtype=float)
ry = mardia(Y)
print(f"\nExemplo pequeno: b1 = {ry[0]:.4f}, b2 = {ry[1]:.4f}, d² = {np.round(ry[7], 3)}")

# 6. Gráfico Q-Q das distâncias de Mahalanobis contra o qui-quadrado
d2_ord = np.sort(d2)
q = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, p)
plt.scatter(q, d2_ord); plt.plot([0, q.max()], [0, q.max()], color="red")
plt.xlabel("Quantis do qui-quadrado (4 gl)"); plt.ylabel("Distância de Mahalanobis ao quadrado"); plt.show()
