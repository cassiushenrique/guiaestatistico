# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def var_winsorizada(x, tr=0.2):
    """Variância winsorizada: substitui as caudas pelos valores que sobraram na borda."""
    n = len(x); g = int(np.floor(tr * n)); xs = np.sort(x)
    w = np.r_[np.repeat(xs[g], g + 1), xs[g+1:n-g-1], np.repeat(xs[n-g-1], g + 1)]
    return w.var(ddof=1), g

def yuen(x, y, tr=0.2):
    """Teste de Yuen: compara médias aparadas de dois grupos independentes."""
    nx, ny = len(x), len(y)
    sx, gx = var_winsorizada(x, tr); sy, gy = var_winsorizada(y, tr)
    hx, hy = nx - 2 * gx, ny - 2 * gy                       # observações efetivamente usadas
    dx = (nx - 1) * sx / (hx * (hx - 1)); dy = (ny - 1) * sy / (hy * (hy - 1))
    mx, my = stats.trim_mean(x, tr), stats.trim_mean(y, tr)
    ep = np.sqrt(dx + dy); t = (mx - my) / ep
    gl = (dx + dy)**2 / (dx**2 / (hx - 1) + dy**2 / (hy - 1))
    return dict(t=t, gl=gl, p=2 * stats.t.sf(abs(t), gl), mx=mx, my=my, ep=ep,
                hx=hx, hy=hy, gx=gx, gy=gy, sx=sx, sy=sy)

# 1. Dados: tempo de ciclo (segundos) de duas máquinas, 20 medições cada (dados fictícios).
#    Há paradas ocasionais e leituras falhas, que produzem valores extremos nos dois sentidos.
maquina1 = np.array([51.6, 49.5, 52.6, 50.8, 49.8, 51.2, 28.4, 48.8, 50.2, 52.0,
                     48.0, 53.7, 46.3, 74.8, 48.4, 31.2, 49.2, 47.4, 50.5, 70.6])
maquina2 = np.array([30.1, 55.8, 57.2, 33.5, 52.3, 56.5, 78.2, 54.0, 54.4, 56.2,
                     58.6, 55.5, 55.2, 56.8, 53.4, 57.6, 59.7, 54.8, 83.0, 58.0])
for nome, g in [("máquina 1", maquina1), ("máquina 2", maquina2)]:
    print(f"{nome}: n = {len(g)} | média = {g.mean():.3f} | DP = {g.std(ddof=1):.3f} "
          f"| mediana = {np.median(g):.2f} | média aparada 20% = {stats.trim_mean(g, 0.2):.3f}")

# 2. O problema: caudas pesadas inflam o desvio-padrão
for nome, g in [("máquina 1", maquina1), ("máquina 2", maquina2)]:
    W, pw = stats.shapiro(g)
    print(f"Shapiro-Wilk ({nome}): W = {W:.4f}, p-valor = {pw:.5f} | amplitude = {g.min()} a {g.max()}")

# 3. Teste t de Welch (usa a média e o desvio-padrão completos)
tw = stats.ttest_ind(maquina1, maquina2, equal_var=False)
print(f"\nWelch: t = {tw.statistic:.4f}, gl = {tw.df:.3f}, p-valor = {tw.pvalue:.4f}")

# 4. Teste de Yuen com 20% de aparagem
r = yuen(maquina1, maquina2, tr=0.2)
print(f"\nAparagem de 20%: {r['gx']} valores descartados em cada ponta, restando h = {r['hx']} e {r['hy']}")
print(f"Médias aparadas: {r['mx']:.4f} e {r['my']:.4f} | diferença = {r['mx'] - r['my']:.4f} s")
print(f"Variâncias winsorizadas: {r['sx']:.4f} e {r['sy']:.4f} (contra {maquina1.var(ddof=1):.2f} e {maquina2.var(ddof=1):.2f} nas variâncias comuns)")
print(f"Erro-padrão = {r['ep']:.4f} | t = {r['t']:.4f} | gl = {r['gl']:.3f} | p-valor = {r['p']:.3e}")

# 5. Intervalo de confiança de 95% para a diferença das médias aparadas
tc = stats.t.ppf(0.975, r["gl"]); ic = (r["mx"] - r["my"]) + np.array([-1, 1]) * tc * r["ep"]
print(f"IC 95% = [{ic[0]:.4f}; {ic[1]:.4f}] s")

# 6. Efeito da proporção de aparagem
print("\nAparagem   médias aparadas        t        gl     p-valor")
for tr in [0.0, 0.1, 0.2, 0.25]:
    rr = yuen(maquina1, maquina2, tr=tr) if tr > 0 else None
    if tr == 0:
        print(f"{tr:>7.0%}   {maquina1.mean():.3f} e {maquina2.mean():.3f}   {tw.statistic:>7.4f}  {tw.df:>7.3f}   {tw.pvalue:.5f}")
    else:
        print(f"{tr:>7.0%}   {rr['mx']:.3f} e {rr['my']:.3f}   {rr['t']:>7.4f}  {rr['gl']:>7.3f}   {rr['p']:.3e}")

# 7. Comparação com outras alternativas
print(f"\nMann-Whitney: p-valor = {stats.mannwhitneyu(maquina1, maquina2).pvalue:.5f}")
print(f"Diferença entre as medianas: {np.median(maquina1) - np.median(maquina2):.3f} s")

# 8. Gráficos
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].boxplot([maquina1, maquina2]); ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["máquina 1", "máquina 2"])
for j, g in enumerate([maquina1, maquina2], 1): ax[0].scatter(np.full(len(g), j), g, alpha=0.6)
ax[0].set_ylabel("Tempo de ciclo (s)")
ax[1].bar([0, 1], [maquina1.mean(), maquina2.mean()], width=0.35, label="média")
ax[1].bar([0.4, 1.4], [stats.trim_mean(maquina1, 0.2), stats.trim_mean(maquina2, 0.2)], width=0.35, label="média aparada 20%")
ax[1].set_xticks([0.2, 1.2]); ax[1].set_xticklabels(["máquina 1", "máquina 2"]); ax[1].set_ylabel("Tempo (s)"); ax[1].legend()
plt.show()
