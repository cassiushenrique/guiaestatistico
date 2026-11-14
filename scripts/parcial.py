# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do conjunto mtcars: consumo (mpg), potência (hp) e peso (wt)
mpg = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
hp  = np.array([110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109])
wt  = np.array([2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77, 3.57, 2.78])
n = len(mpg)

# 1. Correlações simples (de ordem zero)
r_xy = stats.pearsonr(hp, mpg).statistic
r_xz = stats.pearsonr(hp, wt).statistic
r_yz = stats.pearsonr(mpg, wt).statistic
print(f"r(hp, mpg) = {r_xy:.4f} | r(hp, wt) = {r_xz:.4f} | r(mpg, wt) = {r_yz:.4f}")

# 2. Correlação parcial entre hp e mpg, controlando wt (fórmula)
r_p = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
k = 1                                           # número de variáveis controladas
gl = n - 2 - k
t = r_p * np.sqrt(gl / (1 - r_p**2))
p = 2 * stats.t.sf(abs(t), gl)
print(f"Correlação parcial r(hp, mpg | wt) = {r_p:.4f} | t = {t:.4f}, gl = {gl}, p-valor = {p:.5f}")

# 3. O mesmo resultado pela correlação entre resíduos
def residuos(v, z):
    b, a = np.polyfit(z, v, 1)
    return v - (a + b * z)
e_hp, e_mpg = residuos(hp, wt), residuos(mpg, wt)
print(f"Correlação entre os resíduos = {stats.pearsonr(e_hp, e_mpg).statistic:.4f}")

# 4. Intervalo de confiança de 95% (z de Fisher com n - 3 - k)
z = np.arctanh(r_p); ep = 1 / np.sqrt(n - 3 - k); zc = stats.norm.ppf(0.975)
ic = np.tanh([z - zc * ep, z + zc * ep])
print(f"IC 95% = [{ic[0]:.4f}; {ic[1]:.4f}] | r parcial ao quadrado = {r_p**2:.4f}")

# 5. Correlação semiparcial (wt removido apenas de hp)
r_sp = (r_xy - r_xz * r_yz) / np.sqrt(1 - r_xz**2)
print(f"Correlação semiparcial = {r_sp:.4f} | ao quadrado = {r_sp**2:.4f}")

# 6. Matriz de correlações parciais pela inversa da matriz de correlações
R = np.corrcoef([mpg, hp, wt]); P = np.linalg.inv(R)
parc = -P / np.sqrt(np.outer(np.diag(P), np.diag(P))); np.fill_diagonal(parc, 1)
print("Matriz de correlações parciais (mpg, hp, wt):\n", np.round(parc, 4))

# 7. Gráficos: dispersão original e dispersão dos resíduos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(hp, mpg); ax[0].set_xlabel("Potência (hp)"); ax[0].set_ylabel("Consumo (mpg)")
ax[1].scatter(e_hp, e_mpg, color="darkgreen"); ax[1].set_xlabel("Resíduo de hp | wt"); ax[1].set_ylabel("Resíduo de mpg | wt")
plt.show()
