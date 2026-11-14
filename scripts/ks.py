# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mpg  = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
cyl  = np.array([6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4])
disp = np.array([160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472, 460, 440, 78.7, 75.7, 71.1, 120.1, 318, 304, 350, 400, 79, 120.3, 95.1, 351, 145, 301, 121])

# Funções auxiliares ----
def ks_dist(x, cdf):
    """Retorna D+, D- e D para a CDF teórica informada."""
    xs = np.sort(x); n = len(xs); F = cdf(xs); i = np.arange(1, n + 1)
    dp = np.max(i / n - F); dm = np.max(F - (i - 1) / n)
    return dp, dm, max(dp, dm)

def lillie(x):
    """Lilliefors para normalidade (aproximação de Dallal e Wilkinson, 1986)."""
    xs = np.sort(x); n = len(xs); i = np.arange(1, n + 1)
    p = stats.norm.cdf((xs - xs.mean()) / xs.std(ddof=1))
    K = max(np.max(i / n - p), np.max(p - (i - 1) / n))
    Kd, nd = (K, n) if n <= 100 else (K * (n / 100)**0.49, 100)
    pv = np.exp(-7.01256*Kd**2*(nd + 2.78019) + 2.99587*Kd*np.sqrt(nd + 2.78019)
                - 0.122119 + 0.974598/np.sqrt(nd) + 1.67997/nd)
    if pv > 0.1:
        KK = (np.sqrt(n) - 0.01 + 0.85/np.sqrt(n)) * K
        if KK <= 0.302: pv = 1
        elif KK <= 0.5: pv = 2.76773 - 19.828315*KK + 80.709644*KK**2 - 138.55152*KK**3 + 81.218052*KK**4
        elif KK <= 0.9: pv = -4.901232 + 40.662806*KK - 97.490286*KK**2 + 94.029866*KK**3 - 32.355711*KK**4
        elif KK <= 1.31: pv = 6.198765 - 19.558097*KK + 23.186922*KK**2 - 12.234627*KK**3 + 2.423045*KK**4
        else: pv = 0
    return K, pv

# 1. Dados: consumo (mpg) de 32 automóveis
x = mpg
q = np.percentile(x, [0, 25, 50, 75, 100])
print("Mín, Q1, Mediana, Q3, Máx:", np.round(q, 2), "| Média:", round(x.mean(), 2), "| n:", len(x))

# 2. ECDF versus CDF teórica N(20, 6²)
cdf0 = lambda v: stats.norm.cdf(v, 20, 6)
xs = np.sort(x)
plt.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", label="ECDF")
t = np.linspace(xs.min() - 5, xs.max() + 5, 300)
plt.plot(t, cdf0(t), color="red", label="N(20, 6²)")
plt.legend(); plt.xlabel("mpg"); plt.show()

# 3. KS com parâmetros fixados a priori (valor-p assintótico, pois há empates)
res = stats.kstest(x, cdf0, method="asymp")
print(f"\nKS (N(20, 6²)): D = {res.statistic:.5f}, p-valor = {res.pvalue:.4f}")

# 4. Distância calculada manualmente
dp, dm, D = ks_dist(x, cdf0)
print(f"D+ = {dp:.6f} | D- = {dm:.6f} | D = {D:.6f}")

# 5. Parâmetros estimados da amostra: Lilliefors
for nome, v in [("mpg", mpg), ("disp", disp)]:
    K, pv = lillie(v)
    print(f"Lilliefors ({nome}): D = {K:.4f}, p-valor = {pv:.4f}")

# 6. Exemplo do texto (n = 5) com N(20, 6²)
y = np.array([12, 17, 19, 22, 28.])
dp5, dm5, D5 = ks_dist(y, cdf0)
print(f"\nn = 5: D+ = {dp5:.4f}, D- = {dm5:.4f}, D = {D5:.4f}, p exato = {stats.kstest(y, cdf0).pvalue:.4f}")
