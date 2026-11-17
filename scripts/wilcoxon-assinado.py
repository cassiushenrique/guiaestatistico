# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados: conjunto sleep (horas extras de sono de 10 pacientes com dois medicamentos).
# Fonte original: Cushny e Peebles (1905), usados por Student (1908).
medicamento1 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
medicamento2 = np.array([1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])
d = medicamento2 - medicamento1
n = len(d)
print("Diferenças:", d)
print(f"Média = {d.mean():.4f} | mediana = {np.median(d):.2f} | DP = {d.std(ddof=1):.4f}")
print(f"Shapiro-Wilk das diferenças: W = {stats.shapiro(d).statistic:.4f}, p-valor = {stats.shapiro(d).pvalue:.4f}")
print("(foi esse resultado que motivou o uso de uma alternativa ao teste t pareado)")

# 1. Construção dos postos: zeros são descartados e os postos vêm dos VALORES ABSOLUTOS
nz = d[d != 0]
m = len(nz)
postos = stats.rankdata(np.abs(nz))
print(f"\nDiferenças não nulas: {m} de {n} (a diferença 0,0 do paciente 5 é descartada)")
print("|d|     ", np.abs(nz))
print("posto   ", postos)
print("sinal   ", np.where(nz > 0, "+", "-"))

# 2. Somas dos postos positivos e negativos
Wp = postos[nz > 0].sum(); Wm = postos[nz < 0].sum()
print(f"\nW+ = {Wp:.1f} | W- = {Wm:.1f} | soma = {Wp + Wm:.1f} = m(m+1)/2 = {m*(m+1)/2:.1f}")
print("Todas as diferenças não nulas são positivas: W+ atinge o máximo possível.")

# 3. Valor-p exato (válido sem empates)
print(f"\nValor-p exato: {stats.wilcoxon(d, method='exact').pvalue:.6f}  (= 2 / 2^{m} = {2/2**m:.6f})")

# 4. Aproximação normal com correção de continuidade e de empates (o que o R usa quando há empates)
E = m * (m + 1) / 4
emp = np.unique(np.abs(nz), return_counts=True)[1]
V = m * (m + 1) * (2 * m + 1) / 24 - np.sum(emp**3 - emp) / 48
z = (Wp - E - 0.5) / np.sqrt(V)
print(f"\nAproximação normal: E(W+) = {E:.2f} | Var(W+) = {V:.4f} | z = {z:.4f}")
print(f"  valor-p = {2 * stats.norm.sf(z):.5f}")
print(f"scipy (aproximação): {stats.wilcoxon(d, method='approx', correction=True).pvalue:.5f}")
print("Há um empate (duas diferenças iguais a 1,3): por isso o R devolve a aproximação, e não o valor exato.")

# 5. Comparação com o teste t pareado e com o teste do sinal
tt = stats.ttest_1samp(d, 0)
print(f"\nTeste t pareado: t = {tt.statistic:.4f}, p-valor = {tt.pvalue:.6f}")
pos = int(np.sum(d > 0))
print(f"Teste do sinal: {pos} positivas de {m} não nulas | p-valor = {2*stats.binom.sf(pos-1, m, 0.5):.5f}")

# 6. Estimativa de Hodges-Lehmann: mediana das médias de todos os pares (médias de Walsh)
walsh = np.array([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
print(f"\nHodges-Lehmann (mediana das {len(walsh)} médias de Walsh) = {np.median(walsh):.4f} h")
print(f"Mediana das diferenças = {np.median(d):.4f} h | média = {d.mean():.4f} h")

# 7. Tamanho do efeito
print(f"\nr = z / raiz(m) = {z/np.sqrt(m):.4f}")
print(f"Proporção de diferenças positivas = {pos}/{m} = {pos/m:.3f}")

# 8. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cores = ["darkgreen" if v > 0 else "red" for v in nz]
ax[0].barh(range(1, m + 1), np.sort(np.abs(nz)), color=[cores[i] for i in np.argsort(np.abs(nz))])
ax[0].set_xlabel("|diferença| (h)"); ax[0].set_ylabel("Posto")
ax[1].hist(d, bins=6, color="lightgray", edgecolor="white"); ax[1].axvline(0, color="red", lw=2)
ax[1].axvline(np.median(walsh), color="darkgreen", ls="--", lw=2)
ax[1].set_xlabel("Diferença (h)"); ax[1].set_ylabel("Frequência")
plt.show()
