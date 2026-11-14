# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def box_m(grupos):
    """Teste M de Box: recebe uma lista de matrizes (n_i x p), uma por grupo."""
    g = len(grupos); p = grupos[0].shape[1]
    ni = np.array([len(x) for x in grupos]); N = ni.sum()
    Si = [np.cov(x, rowvar=False) for x in grupos]
    Sp = sum((n - 1) * S for n, S in zip(ni, Si)) / (N - g)          # covariância combinada
    M = (N - g) * np.log(np.linalg.det(Sp)) - sum((n - 1) * np.log(np.linalg.det(S)) for n, S in zip(ni, Si))
    c = (2*p**2 + 3*p - 1) / (6 * (p + 1) * (g - 1)) * (np.sum(1 / (ni - 1)) - 1 / (N - g))
    X2 = M * (1 - c); gl = p * (p + 1) * (g - 1) // 2
    return M, c, X2, gl, stats.chi2.sf(X2, gl), Si, Sp

# 1. Dados: iris (150 flores, 3 espécies, 4 medidas em cm)
iris = load_iris()
X, y, nomes = iris.data, iris.target, iris.target_names
grupos = [X[y == j] for j in range(3)]

# 2. Determinantes das matrizes de covariâncias
M, c, X2, gl, valor_p, Si, Sp = box_m(grupos)
for nome, S in zip(nomes, Si):
    print(f"{nome:<11} n = 50 | ln|S| = {np.log(np.linalg.det(S)):8.4f} | variâncias = {np.round(np.diag(S), 3)}")
print(f"combinada          | ln|Sp| = {np.log(np.linalg.det(Sp)):7.4f}")

# 3. Teste M de Box
print(f"\nM = {M:.4f} | fator de correção (1 - c) = {1 - c:.5f}")
print(f"Qui-quadrado = {X2:.3f}, gl = {gl}, p-valor = {valor_p:.3e}")

# 4. Exemplo do texto: dois grupos, duas variáveis
A = np.array([[2, 3], [4, 5], [6, 4], [8, 8], [10, 10]], dtype=float)
B = np.array([[1, 9], [5, 2], [6, 12], [9, 3], [14, 9]], dtype=float)
M2, c2, X22, gl2, p2, S2, Sp2 = box_m([A, B])
print("\nExemplo pequeno:")
print("S1 =", np.round(S2[0], 2).tolist(), "| S2 =", np.round(S2[1], 2).tolist(), "| Sp =", np.round(Sp2, 3).tolist())
print(f"|S1| = {np.linalg.det(S2[0]):.3f}, |S2| = {np.linalg.det(S2[1]):.3f}, |Sp| = {np.linalg.det(Sp2):.3f}")
print(f"M = {M2:.4f}, c = {c2:.4f}, qui-quadrado = {X22:.4f}, gl = {gl2}, p-valor = {p2:.4f}")

# 5. Gráfico: dispersão de duas variáveis por espécie
for j, cor in zip(range(3), ["tab:green", "tab:orange", "tab:purple"]):
    plt.scatter(X[y == j, 2], X[y == j, 3], color=cor, label=nomes[j], alpha=0.7)
plt.xlabel("Comprimento da pétala (cm)"); plt.ylabel("Largura da pétala (cm)"); plt.legend(); plt.show()
