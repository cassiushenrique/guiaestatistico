# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def multinom(X, y, K, nomes):
    """Logística multinomial por máxima verossimilhança; a classe 0 é a referência."""
    n = len(y); Xd = np.column_stack([np.ones(n), X]); q = Xd.shape[1]
    def menos_logv(b):
        B = np.vstack([np.zeros(q), b.reshape(K - 1, q)])
        eta = Xd @ B.T; m = eta.max(axis=1, keepdims=True)
        return -np.sum(eta[np.arange(n), y] - m.ravel() - np.log(np.exp(eta - m).sum(axis=1)))
    res = optimize.minimize(menos_logv, np.zeros((K - 1) * q), method="BFGS")
    H = hessiana(menos_logv, res.x); ep = np.sqrt(np.diag(np.linalg.inv(H)))
    b = res.x; z = b / ep; p = 2 * stats.norm.sf(np.abs(z))
    print(f"{'Equação':<26}{'Estimativa':>11}{'Erro-padrão':>13}{'z':>8}{'p-valor':>10}{'OR':>9}")
    for i, nm in enumerate(nomes):
        for j, termo in enumerate(["(Intercepto)"] + [f"x{j}" for j in range(1, q)]):
            k = i * q + j
            print(f"{nm + ': ' + termo:<26}{b[k]:>11.4f}{ep[k]:>13.4f}{z[k]:>8.3f}{p[k]:>10.4f}{np.exp(b[k]):>9.3f}")
    return dict(b=b, ep=ep, logv=-res.fun, q=q, K=K, Xd=Xd)

def hessiana(f, x, h=1e-4):
    """Matriz hessiana por diferenças finitas centrais."""
    p = len(x); H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            e1 = np.zeros(p); e2 = np.zeros(p); e1[i] = h; e2[j] = h
            H[i, j] = (f(x+e1+e2) - f(x+e1-e2) - f(x-e1+e2) + f(x-e1-e2)) / (4 * h * h)
    return H

def probs(m, X):
    Xd = np.column_stack([np.ones(len(X)), X])
    B = np.vstack([np.zeros(m["q"]), m["b"].reshape(m["K"] - 1, m["q"])])
    eta = Xd @ B.T; e = np.exp(eta - eta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# 1. Dados: iris (150 flores de 3 espécies); preditor = comprimento da sépala (cm)
iris = load_iris(); especies = list(iris.target_names)
x = iris.data[:, 0]; y = iris.target; n = len(y)
for j, nome in enumerate(especies):
    print(f"{nome:<11} n = {np.sum(y == j)} | média = {x[y == j].mean():.3f} | DP = {x[y == j].std(ddof=1):.3f} | faixa = {x[y == j].min()}-{x[y == j].max()}")

# 2. Ajuste: setosa é a categoria de referência
m = multinom(x[:, None], y, 3, ["versicolor", "virginica"])
print(f"\nLog-verossimilhança = {m['logv']:.4f} | AIC = {-2 * m['logv'] + 2 * len(m['b']):.3f}")

# 3. Modelo nulo e teste da razão de verossimilhanças
props = np.bincount(y) / n
logv0 = np.sum(np.bincount(y) * np.log(props))
G = 2 * (m["logv"] - logv0)
print(f"Modelo nulo: log-verossimilhança = {logv0:.4f} | G = {G:.3f}, gl = 2, p-valor = {stats.chi2.sf(G, 2):.3e}")
print(f"Pseudo-R² de McFadden = {1 - m['logv'] / logv0:.4f}")

# 4. Probabilidades estimadas em alguns comprimentos
print("\nComprimento   setosa  versicolor  virginica")
for x0 in [4.5, 5.0, 5.5, 6.0, 6.5, 7.5]:
    p = probs(m, np.array([[x0]]))[0]
    print(f"{x0:>9.1f} cm {p[0]:>8.3f} {p[1]:>11.3f} {p[2]:>10.3f}")

# 5. Classificação (nos próprios dados do ajuste: medida otimista)
P = probs(m, x[:, None]); pred = P.argmax(axis=1)
print("\nMatriz de confusão (linhas = observado, colunas = previsto):")
conf = np.array([[np.sum((y == i) & (pred == j)) for j in range(3)] for i in range(3)])
print(conf); print(f"Acurácia = {np.mean(pred == y):.4f} | acerto esperado ao acaso = 0.3333")

# 6. Razão de chances para um aumento de 1 cm
print("\nOR (chance da espécie em relação à setosa, por +1 cm de sépala):")
zc = stats.norm.ppf(0.975)
for i, nome in enumerate(["versicolor", "virginica"]):
    k = i * m["q"] + 1
    print(f"{nome:<11} OR = {np.exp(m['b'][k]):8.3f}   IC 95% = [{np.exp(m['b'][k] - zc*m['ep'][k]):.3f}; {np.exp(m['b'][k] + zc*m['ep'][k]):.3f}]")

# 7. O que acontece com separação perfeita (o comprimento da pétala separa a setosa)
Xp = np.column_stack([np.ones(n), iris.data[:, 2]])
def menos_logv_petala(b):
    B = np.vstack([np.zeros(2), b.reshape(2, 2)]); eta = Xp @ B.T
    m_ = eta.max(axis=1, keepdims=True)
    return -np.sum(eta[np.arange(n), y] - m_.ravel() - np.log(np.exp(eta - m_).sum(axis=1)))
rp = optimize.minimize(menos_logv_petala, np.zeros(4), method="BFGS")
print(f"\nCom o comprimento da pétala: maior coeficiente em valor absoluto = {np.abs(rp.x).max():.1f}")
print("As estimativas crescem sem limite a cada iteração: é o sintoma de separação perfeita.")

# 8. Gráfico: probabilidades estimadas de cada espécie
grade = np.linspace(4, 8.2, 200)
P = probs(m, grade[:, None])
for j, nome in enumerate(especies): plt.plot(grade, P[:, j], label=nome)
plt.xlabel("Comprimento da sépala (cm)"); plt.ylabel("Probabilidade estimada"); plt.legend(); plt.show()
