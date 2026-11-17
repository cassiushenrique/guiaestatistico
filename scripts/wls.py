# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados: 20 lotes de produção (dados fictícios).
# teor  = teor de aditivo (%) | n = número de corpos de prova medidos no lote
# resistencia = resistência média do lote (MPa)
teor        = np.array([0.5, 0.7, 1.0, 1.2, 1.4, 1.7, 1.9, 2.2, 2.4, 2.6,
                        2.9, 3.1, 3.3, 3.6, 3.8, 4.1, 4.3, 4.5, 4.8, 5.0])
n           = np.array([2, 20, 12, 18, 18, 2, 12, 4, 15, 20, 5, 3, 10, 4, 10, 3, 5, 8, 15, 8.])
resistencia = np.array([38.24, 32.48, 33.92, 34.87, 34.61, 40.46, 37.01, 34.39, 40.48, 41.85,
                        40.08, 44.83, 42.93, 45.74, 43.56, 44.85, 47.84, 45.71, 49.73, 50.30])
N = len(teor)

def mqp(X, y, w, nomes):
    """Mínimos quadrados ponderados. Com w = 1 para todos, equivale aos mínimos quadrados comuns."""
    XtWX = X.T @ (X * w[:, None])
    b = np.linalg.solve(XtWX, X.T @ (w * y))
    r = y - X @ b
    s2 = np.sum(w * r**2) / (len(y) - X.shape[1])          # variância residual ponderada
    cov = s2 * np.linalg.inv(XtWX); ep = np.sqrt(np.diag(cov))
    t = b / ep; p = 2 * stats.t.sf(np.abs(t), len(y) - X.shape[1])
    tc = stats.t.ppf(0.975, len(y) - X.shape[1])
    print(f"{'Termo':<12}{'Estimativa':>11}{'Erro-padrão':>13}{'t':>8}{'p-valor':>11}{'IC 95%':>22}")
    for nm, bi, ei, ti, pi in zip(nomes, b, ep, t, p):
        print(f"{nm:<12}{bi:>11.4f}{ei:>13.4f}{ti:>8.3f}{pi:>11.2e}   [{bi-tc*ei:>7.4f}; {bi+tc*ei:>7.4f}]")
    sqt = np.sum(w * (y - np.sum(w * y) / np.sum(w))**2)
    print(f"R² ponderado = {1 - np.sum(w * r**2) / sqt:.4f} | erro-padrão residual = {np.sqrt(s2):.4f}")
    return dict(b=b, ep=ep, res=r, s2=s2)

X = np.column_stack([np.ones(N), teor])

# 1. Por que ponderar: cada média vem de um número diferente de corpos de prova
print("Lotes com menos medições (n <= 4):", np.sum(n <= 4), "| com mais (n >= 15):", np.sum(n >= 15))
print(f"Variância de uma média de 2 medições é {20/2:.0f} vezes a de uma média de 20.")

# 2. Mínimos quadrados comuns (todos os lotes com o mesmo peso)
print("\nMínimos quadrados comuns:")
m_ols = mqp(X, resistencia, np.ones(N), ["(Intercepto)", "teor"])

# 3. Mínimos quadrados ponderados (peso = número de medições do lote)
print("\nMínimos quadrados ponderados (peso = n):")
m_wls = mqp(X, resistencia, n, ["(Intercepto)", "teor"])

# 4. Ganho de precisão
print(f"\nErro-padrão da inclinação: comum = {m_ols['ep'][1]:.4f} | ponderado = {m_wls['ep'][1]:.4f}"
      f" | redução de {100 * (1 - m_wls['ep'][1] / m_ols['ep'][1]):.1f}%")
print(f"Largura do IC da inclinação: {2*stats.t.ppf(0.975, N-2)*m_ols['ep'][1]:.4f} -> {2*stats.t.ppf(0.975, N-2)*m_wls['ep'][1]:.4f}")

# 5. Resíduos padronizados: só os ponderados têm variância comparável
r_ols = m_ols["res"] / np.sqrt(m_ols["s2"])
r_wls = m_wls["res"] * np.sqrt(n) / np.sqrt(m_wls["s2"])
print("\nDesvio-padrão dos resíduos padronizados, por tamanho de lote:")
for rot, sel in [("n <= 4 ", n <= 4), ("n >= 15", n >= 15)]:
    print(f"  {rot}: comum = {r_ols[sel].std(ddof=1):.3f} | ponderado = {r_wls[sel].std(ddof=1):.3f}")

# 6. Observações mais e menos influentes no ajuste comum
i = np.argsort(-np.abs(r_ols))[:3]
print("\nMaiores resíduos padronizados no ajuste comum (lote, n, resíduo):")
for k in i: print(f"  lote {k+1:>2} | n = {int(n[k]):>2} | {r_ols[k]:>6.2f}")

# 7. Quando os pesos vêm da variância conhecida de cada média: w = 1 / s²
print("\nPesos iguais ao inverso da variância (s² = 18 / n):")
s2_lote = 18.0 / n
m_var = mqp(X, resistencia, 1 / s2_lote, ["(Intercepto)", "teor"])
print("Multiplicar todos os pesos por uma constante não muda as estimativas nem os erros-padrão;")
print("muda apenas a escala do erro-padrão residual, que agora deve ficar próximo de 1.")

# 8. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(teor, resistencia, s=10 * n, alpha=0.5)
grade = np.linspace(0.3, 5.2, 50)
ax[0].plot(grade, m_ols["b"][0] + m_ols["b"][1] * grade, "--", label="mínimos quadrados comuns")
ax[0].plot(grade, m_wls["b"][0] + m_wls["b"][1] * grade, color="red", label="ponderados")
ax[0].set_xlabel("Teor de aditivo (%)"); ax[0].set_ylabel("Resistência média (MPa)"); ax[0].legend()
ax[1].scatter(n, r_ols, label="comum"); ax[1].scatter(n, r_wls, marker="^", label="ponderado")
ax[1].axhline(0, color="red"); ax[1].set_xlabel("Número de medições do lote"); ax[1].set_ylabel("Resíduo padronizado"); ax[1].legend()
plt.show()
