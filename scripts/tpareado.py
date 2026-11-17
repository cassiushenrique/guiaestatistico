# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados: conjunto sleep (horas extras de sono de 10 pacientes com dois medicamentos).
# Fonte original: Cushny e Peebles (1905), usados por Student (1908).
medicamento1 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
medicamento2 = np.array([1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])
n = len(medicamento1)
d = medicamento2 - medicamento1            # a análise é feita sobre as DIFERENÇAS
print("Paciente  medicamento1  medicamento2  diferença")
for i in range(n):
    print(f"{i+1:>8}  {medicamento1[i]:>12}  {medicamento2[i]:>12}  {d[i]:>9.1f}")
print(f"\nMédia das diferenças = {d.mean():.4f} h | DP das diferenças = {d.std(ddof=1):.4f} h")

# 1. Estatística de teste: é um teste t de uma amostra aplicado às diferenças
ep = d.std(ddof=1) / np.sqrt(n)
t = d.mean() / ep; gl = n - 1
print(f"\nErro-padrão = {d.std(ddof=1):.4f} / raiz({n}) = {ep:.4f}")
print(f"t = {d.mean():.4f} / {ep:.4f} = {t:.4f} com {gl} gl | p-valor = {2*stats.t.sf(abs(t), gl):.6f}")
print("scipy (pareado):", stats.ttest_rel(medicamento2, medicamento1))
print("scipy (uma amostra sobre as diferenças):", stats.ttest_1samp(d, 0))

# 2. Intervalo de confiança de 95% para a diferença média
tc = stats.t.ppf(0.975, gl); ic = d.mean() + np.array([-1, 1]) * tc * ep
print(f"IC 95% = [{ic[0]:.4f}; {ic[1]:.4f}] h")

# 3. O que se perderia ignorando o pareamento
t_ind = stats.ttest_ind(medicamento2, medicamento1, equal_var=True)
print(f"\nSe os grupos fossem tratados como independentes: t = {t_ind.statistic:.4f}, p-valor = {t_ind.pvalue:.4f}")
r = stats.pearsonr(medicamento1, medicamento2)
print(f"Correlação entre as duas medidas do mesmo paciente: r = {r.statistic:.4f} (p = {r.pvalue:.5f})")
print("É essa correlação que o pareamento aproveita: ela reduz a variabilidade da diferença.")

# 4. Pressuposto: normalidade das diferenças
W, pw = stats.shapiro(d)
print(f"\nShapiro-Wilk das diferenças: W = {W:.4f}, p-valor = {pw:.4f}")
print("Diferenças ordenadas:", np.sort(d))

# 5. Tamanho do efeito
print(f"\nd de Cohen (pareado) = média / DP das diferenças = {d.mean()/d.std(ddof=1):.4f}")

# 6. Alternativas não paramétricas
print("Wilcoxon (posto sinalizado):", stats.wilcoxon(d).pvalue.round(5))
pos = int(np.sum(d > 0))
print(f"Teste do sinal: {pos} diferenças positivas de {int(np.sum(d != 0))} não nulas | p-valor =",
      round(2 * stats.binom.sf(pos - 1, int(np.sum(d != 0)), 0.5), 5))

# 7. Poder do teste
delta = abs(d.mean() / d.std(ddof=1)) * np.sqrt(n)
print(f"Poder para o efeito observado (alfa = 0,05): {stats.nct.sf(tc, gl, delta) + stats.nct.cdf(-tc, gl, delta):.4f}")

# 8. Gráficos
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for i in range(n): ax[0].plot([1, 2], [medicamento1[i], medicamento2[i]], marker="o", color="gray")
ax[0].plot([1, 2], [medicamento1.mean(), medicamento2.mean()], marker="o", color="red", lw=3)
ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["medicamento 1", "medicamento 2"]); ax[0].set_ylabel("Horas extras de sono")
ax[1].hist(d, bins=6, color="lightgray", edgecolor="white"); ax[1].axvline(0, color="red", lw=2)
ax[1].set_xlabel("Diferença (h)"); ax[1].set_ylabel("Frequência")
stats.probplot(d, dist="norm", plot=ax[2]); ax[2].set_title("Q-Q das diferenças")
plt.show()
