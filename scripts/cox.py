# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

# Dados: ensaio de confiabilidade com 30 bombas centrífugas (dados fictícios)
# tempo: horas até a falha ou até o fim do ensaio | falha: 1 = falhou, 0 = censurado
# carga: 0 = normal, 1 = severa | temp: temperatura média de operação (°C)
tempo = np.array([1347, 790, 748, 1424, 827, 744, 265, 811, 257, 247, 557, 802, 59, 444, 655,
                   266, 678, 932, 136, 249, 23, 83, 80, 1151, 24, 477, 46, 326, 378, 332.])
falha = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.])
carga = np.repeat([0., 1.], 15)
temp  = np.array([65, 67, 72, 70, 72, 72, 81, 58, 77, 80, 76, 67, 83, 56, 79,
                   73, 56, 65, 66, 60, 74, 68, 76, 66, 57, 76, 77, 57, 69, 69.])
n = len(tempo)

def logv_parcial(b, X, t, d):
    """Log-verossimilhança parcial de Cox com a correção de Efron para empates."""
    eta = X @ b; total = 0.0
    for tt in np.unique(t[d == 1]):
        D = (t == tt) & (d == 1); R = t >= tt; m = D.sum()
        soma_risco = np.exp(eta[R]).sum(); soma_falhas = np.exp(eta[D]).sum()
        total += eta[D].sum() - sum(np.log(soma_risco - (l / m) * soma_falhas) for l in range(m))
    return total

def hessiana(f, x, h=1e-4):
    p = len(x); H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            e1 = np.zeros(p); e2 = np.zeros(p); e1[i] = h; e2[j] = h
            H[i, j] = (f(x+e1+e2) - f(x+e1-e2) - f(x-e1+e2) + f(x-e1-e2)) / (4 * h * h)
    return H

def cox(X, t, d, nomes):
    f = lambda b: -logv_parcial(b, X, t, d)
    res = optimize.minimize(f, np.zeros(X.shape[1]), method="BFGS")
    cov = np.linalg.inv(hessiana(f, res.x)); ep = np.sqrt(np.diag(cov))
    b = res.x; z = b / ep; p = 2 * stats.norm.sf(np.abs(z)); zc = stats.norm.ppf(0.975)
    print(f"{'Termo':<10}{'Estimativa':>11}{'Erro-padrão':>13}{'z':>8}{'p-valor':>10}{'HR':>8}{'IC 95% de HR':>22}")
    for nm, bi, ei, zi, pi in zip(nomes, b, ep, z, p):
        print(f"{nm:<10}{bi:>11.4f}{ei:>13.4f}{zi:>8.3f}{pi:>10.4f}{np.exp(bi):>8.3f}   [{np.exp(bi-zc*ei):>6.3f}; {np.exp(bi+zc*ei):>6.3f}]")
    return dict(b=b, ep=ep, cov=cov, logv=-res.fun)

# 1. Descrição dos dados
print(f"n = {n} | falhas = {int(falha.sum())} | censuras = {int(n - falha.sum())}")
for g, nome in [(0, "carga normal"), (1, "carga severa")]:
    m_ = carga == g
    print(f"{nome}: n = {int(m_.sum())} | falhas = {int(falha[m_].sum())} | mediana dos tempos observados = {np.median(tempo[m_]):.0f} h")

# 2. Estimador de Kaplan-Meier por grupo
def kaplan_meier(t, d):
    ts = np.unique(t[d == 1]); S = 1.0; curva = [(0.0, 1.0)]
    for tt in ts:
        em_risco = (t >= tt).sum(); falhas = ((t == tt) & (d == 1)).sum()
        S *= 1 - falhas / em_risco; curva.append((tt, S))
    return np.array(curva)

for g, nome in [(0, "carga normal"), (1, "carga severa")]:
    km = kaplan_meier(tempo[carga == g], falha[carga == g])
    abaixo = km[km[:, 1] <= 0.5]
    mediana = abaixo[0, 0] if len(abaixo) else np.nan
    print(f"{nome}: sobrevivência em 500 h = {km[km[:, 0] <= 500][-1, 1]:.3f} | tempo mediano = {mediana:.0f} h")

# 3. Modelo de Cox com os dois preditores (temperatura centrada em 70 °C)
X = np.column_stack([carga, temp - 70])
print()
m = cox(X, tempo, falha, ["carga", "temp-70"])

# 4. Testes globais
m0_logv = logv_parcial(np.zeros(2), X, tempo, falha)
G = 2 * (m["logv"] - m0_logv)
print(f"\nLog-verossimilhança parcial: nula = {m0_logv:.4f} | modelo = {m['logv']:.4f}")
print(f"Razão de verossimilhanças: G = {G:.3f}, gl = 2, p-valor = {stats.chi2.sf(G, 2):.3e}")
W = m["b"] @ np.linalg.inv(m["cov"]) @ m["b"]
print(f"Teste de Wald: {W:.3f}, gl = 2, p-valor = {stats.chi2.sf(W, 2):.3e}")

# 5. Razão de risco para variações relevantes
print(f"\nHR carga severa / normal = {np.exp(m['b'][0]):.3f}")
print(f"HR por +10 °C = {np.exp(10 * m['b'][1]):.3f}")
print(f"HR de uma bomba com carga severa a 80 °C em relação a carga normal a 60 °C = {np.exp(m['b'][0] + 20 * m['b'][1]):.2f}")

# 6. Sobrevivência estimada (risco basal de Breslow)
def sobrevivencia_basal(b, X, t, d):
    eta = X @ b; ts = np.unique(t[d == 1]); H = 0.0; saida = [(0.0, 1.0)]
    for tt in ts:
        H += ((t == tt) & (d == 1)).sum() / np.exp(eta[t >= tt]).sum()
        saida.append((tt, np.exp(-H)))
    return np.array(saida)

S0 = sobrevivencia_basal(m["b"], X, tempo, falha)
print("\nSobrevivência estimada em 500 h (temperatura de 70 °C):")
S0_500 = S0[S0[:, 0] <= 500][-1, 1]
for g, nome in [(0, "carga normal"), (1, "carga severa")]:
    print(f"  {nome}: {S0_500 ** np.exp(m['b'][0] * g):.3f}")

# 7. Verificação de riscos proporcionais: resíduos de Schoenfeld contra o tempo
def schoenfeld(b, X, t, d):
    eta = X @ b; res = []; tempos = []
    for i in np.where(d == 1)[0]:
        R = t >= t[i]; w = np.exp(eta[R]); media = (X[R] * w[:, None]).sum(axis=0) / w.sum()
        res.append(X[i] - media); tempos.append(t[i])
    return np.array(tempos), np.array(res)

tt, sch = schoenfeld(m["b"], X, tempo, falha)
print("\nResíduos de Schoenfeld: correlação com o posto do tempo")
for j, nm in enumerate(["carga", "temp-70"]):
    r, p = stats.spearmanr(stats.rankdata(tt), sch[:, j])
    print(f"  {nm:<8} rho = {r:>7.4f}, p-valor = {p:.4f}")

# 8. Gráficos: Kaplan-Meier por grupo e resíduos de Schoenfeld
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for g, nome in [(0, "carga normal"), (1, "carga severa")]:
    km = kaplan_meier(tempo[carga == g], falha[carga == g])
    ax[0].step(km[:, 0], km[:, 1], where="post", label=nome)
ax[0].set_xlabel("Tempo (h)"); ax[0].set_ylabel("Sobrevivência estimada"); ax[0].legend()
ax[1].scatter(tt, sch[:, 1]); ax[1].axhline(0, color="red")
ax[1].set_xlabel("Tempo da falha (h)"); ax[1].set_ylabel("Resíduo de Schoenfeld (temperatura)")
plt.show()
