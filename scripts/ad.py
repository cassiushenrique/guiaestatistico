# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mpg  = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
cyl  = np.array([6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4])
disp = np.array([160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472, 460, 440, 78.7, 75.7, 71.1, 120.1, 318, 304, 350, 400, 79, 120.3, 95.1, 351, 145, 301, 121])

def ad_normal(x):
    """Anderson–Darling para normalidade: retorna A², A* (corrigida) e valor-p
    (aproximação de D'Agostino e Stephens, 1986, como no pacote nortest do R)."""
    xs = np.sort(x); n = len(xs); i = np.arange(1, n + 1)
    p = stats.norm.cdf((xs - xs.mean()) / xs.std(ddof=1))
    A2 = -n - np.mean((2*i - 1) * (np.log(p) + np.log(1 - p[::-1])))
    As = A2 * (1 + 0.75/n + 2.25/n**2)
    if As < 0.2:   pv = 1 - np.exp(-13.436 + 101.14*As - 223.73*As**2)
    elif As < 0.34: pv = 1 - np.exp(-8.318 + 42.796*As - 59.938*As**2)
    elif As < 0.6:  pv = np.exp(0.9177 - 4.279*As - 1.38*As**2)
    elif As < 10:   pv = np.exp(1.2937 - 5.709*As + 0.0186*As**2)
    else:           pv = 3.7e-24
    return A2, As, pv

# 1. Dados: cilindrada (disp) de 32 automóveis
x = disp

# 2. Diagnóstico gráfico
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(x, bins=8, density=True, color="lightgray", edgecolor="white")
t = np.linspace(x.min(), x.max(), 200)
ax[0].plot(t, stats.norm.pdf(t, x.mean(), x.std(ddof=1)), color="red"); ax[0].set_title("Histograma")
stats.probplot(x, dist="norm", plot=ax[1]); ax[1].set_title("Gráfico Q-Q")
plt.show()

# 3. Teste de Anderson–Darling
A2, As, p = ad_normal(x)
print(f"Anderson-Darling (disp): A² = {A2:.5f}, A* = {As:.5f}, p-valor = {p:.4f}")

# 4. Comparação com outros testes e com dados transformados
W, pw = stats.shapiro(x)
print(f"Shapiro-Wilk (disp):    W = {W:.5f}, p-valor = {pw:.4f}")
A2l, Asl, pl = ad_normal(np.log(x))
print(f"Anderson-Darling log(disp): A² = {A2l:.5f}, p-valor = {pl:.4f}")
A2m, Asm, pm = ad_normal(mpg)
print(f"Anderson-Darling (mpg):     A² = {A2m:.5f}, p-valor = {pm:.4f}")

# 5. Exemplo do texto (n = 5; didático — o R exige n >= 8)
y = np.array([12, 17, 19, 22, 28.])
A5, As5, p5 = ad_normal(y)
print(f"\nn = 5: A² = {A5:.4f}, A* = {As5:.4f}, p ≈ {p5:.3f}")
