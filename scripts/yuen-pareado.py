# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def var_winsorizada(x, tr=0.2):
    """Variância winsorizada: substitui as caudas pelos valores que sobraram na borda."""
    n = len(x); g = int(np.floor(tr * n)); xs = np.sort(x)
    w = np.r_[np.repeat(xs[g], g + 1), xs[g+1:n-g-1], np.repeat(xs[n-g-1], g + 1)]
    return w.var(ddof=1), g

def yuen_pareado(d, tr=0.2):
    """Teste de Yuen pareado: aplica a média aparada às DIFERENÇAS."""
    n = len(d); sw, g = var_winsorizada(d, tr); h = n - 2 * g
    var_m = (n - 1) * sw / (h * (h - 1))                    # variância da média aparada
    m = stats.trim_mean(d, tr); ep = np.sqrt(var_m)
    t = m / ep; gl = h - 1
    return dict(t=t, gl=gl, p=2 * stats.t.sf(abs(t), gl), m=m, ep=ep, h=h, g=g, sw=sw)

# 1. Dados: diferença entre dois métodos de medição aplicados às MESMAS 20 peças (dados fictícios).
#    Valores negativos e muito altos correspondem a leituras com falha de registro.
diferenca = np.array([2.0, 3.8, -6.8, 1.3, 16.5, -9.4, 3.5, 4.0, 1.7, 3.3,
                      2.2, 2.9, 2.7, 5.4, 3.1, 4.3, 0.6, 13.2, 2.5, 4.7])
n = len(diferenca)
print(f"n = {n} | média = {diferenca.mean():.4f} | DP = {diferenca.std(ddof=1):.4f}")
print(f"mediana = {np.median(diferenca):.2f} | média aparada 20% = {stats.trim_mean(diferenca, 0.2):.4f}")
print("Diferenças ordenadas:", np.sort(diferenca))

# 2. Normalidade das diferenças
W, pw = stats.shapiro(diferenca)
print(f"\nShapiro-Wilk: W = {W:.4f}, p-valor = {pw:.5f}")

# 3. Teste t pareado comum
tt = stats.ttest_1samp(diferenca, 0)
ep_t = diferenca.std(ddof=1) / np.sqrt(n); tc_t = stats.t.ppf(0.975, n - 1)
ic_t = diferenca.mean() + np.array([-1, 1]) * tc_t * ep_t
print(f"\nTeste t pareado: t = {tt.statistic:.4f}, gl = {n-1}, p-valor = {tt.pvalue:.5f}")
print(f"  erro-padrão = {ep_t:.4f} | IC 95% = [{ic_t[0]:.4f}; {ic_t[1]:.4f}]")

# 4. Teste de Yuen pareado com 20% de aparagem
r = yuen_pareado(diferenca, tr=0.2)
print(f"\nYuen pareado: {r['g']} diferenças descartadas em cada ponta, restando h = {r['h']}")
print(f"  média aparada = {r['m']:.4f} | variância winsorizada = {r['sw']:.4f} (contra {diferenca.var(ddof=1):.4f})")
print(f"  erro-padrão = {r['ep']:.4f} | t = {r['t']:.4f}, gl = {r['gl']}, p-valor = {r['p']:.3e}")
tc = stats.t.ppf(0.975, r["gl"]); ic = r["m"] + np.array([-1, 1]) * tc * r["ep"]
print(f"  IC 95% = [{ic[0]:.4f}; {ic[1]:.4f}]")
print(f"\nLargura do intervalo: {ic_t[1]-ic_t[0]:.4f} (t comum) contra {ic[1]-ic[0]:.4f} (Yuen)")

# 5. Efeito da proporção de aparagem
print("\nAparagem   média aparada        t     gl    p-valor")
for tr in [0.0, 0.1, 0.2, 0.25]:
    if tr == 0:
        print(f"{tr:>7.0%}   {diferenca.mean():>12.4f} {tt.statistic:>8.4f} {n-1:>6} {tt.pvalue:>10.5f}")
    else:
        rr = yuen_pareado(diferenca, tr)
        print(f"{tr:>7.0%}   {rr['m']:>12.4f} {rr['t']:>8.4f} {rr['gl']:>6} {rr['p']:>10.3e}")

# 6. Alternativas não paramétricas
print(f"\nWilcoxon (posto sinalizado): p-valor = {stats.wilcoxon(diferenca).pvalue:.5f}")
pos = int(np.sum(diferenca > 0))
print(f"Teste do sinal: {pos} positivas de {n} | p-valor = {2*stats.binom.sf(pos-1, n, 0.5):.5f}")

# 7. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(diferenca, bins=10, color="lightgray", edgecolor="white")
ax[0].axvline(0, color="red", lw=2); ax[0].axvline(diferenca.mean(), color="blue", ls="--")
ax[0].axvline(stats.trim_mean(diferenca, 0.2), color="darkgreen", ls="--")
ax[0].set_xlabel("Diferença entre métodos"); ax[0].set_ylabel("Frequência")
xs = np.sort(diferenca); cores = ["red"] * 4 + ["darkgreen"] * 12 + ["red"] * 4
ax[1].scatter(range(1, n + 1), xs, color=cores); ax[1].axhline(0, color="gray")
ax[1].set_xlabel("Diferença ordenada"); ax[1].set_ylabel("Valor")
plt.show()
