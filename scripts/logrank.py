# Bibliotecas: numpy, scipy e matplotlib (todas já no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# Os mesmos dados da ficha do Kaplan-Meier: Freireich et al. (1963),
# 42 pacientes com leucemia, 6-mercaptopurina contra placebo.
# ===================================================================
t6  = np.array([6,6,6,6,7,9,10,10,11,13,16,17,19,20,22,23,25,32,32,34,35])
e6  = np.array([0,1,1,1,1,0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
tpl = np.array([1,1,2,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23])
epl = np.ones(21, int)
t = np.r_[t6, tpl]; e = np.r_[e6, epl]
gr = np.r_[np.zeros(21), np.ones(21)].astype(int)   # 0 = 6-MP, 1 = placebo
rot = ["6-MP", "placebo"]

def tabela_tempos(t, e, gr):
    """Para cada tempo de evento: n em risco e eventos, no total e por grupo."""
    T = np.unique(t[e == 1])
    linhas = []
    for x in T:
        n_ = np.sum(t >= x); d_ = np.sum((t == x) & (e == 1))
        n1 = np.sum((t >= x) & (gr == 0)); d1 = np.sum((t == x) & (e == 1) & (gr == 0))
        linhas.append((x, n_, d_, n1, d1))
    return np.array(linhas, float)

def testes(t, e, gr, rho=0.0):
    """Família G-rho de Harrington-Fleming: rho = 0 é log-rank,
       rho = 1 é o Peto-Peto, a versão do Wilcoxon para dados censurados."""
    L = tabela_tempos(t, e, gr)
    x, n_, d_, n1, d1 = L.T
    E1 = d_ * n1 / n_                                   # esperado no grupo 1
    V1 = d_ * (n1/n_) * (1 - n1/n_) * (n_ - d_) / np.maximum(n_ - 1, 1)
    # peso: S(t-) elevado a rho, com S o Kaplan-Meier combinado
    Sc = np.r_[1.0, np.cumprod(1 - d_/n_)[:-1]]
    w = Sc ** rho
    O_E = np.sum(w * (d1 - E1)); V = np.sum(w**2 * V1)
    Q = O_E**2 / V
    return L, E1, V1, w, O_E, V, Q, stats.chi2.sf(Q, 1)

print("Freireich et al. (1963): tempo até a recaída, em semanas")
for j in [0, 1]:
    print(f"  {rot[j]:<8} n = {np.sum(gr==j)}, eventos = {int(np.sum(e[gr==j]))}, "
          f"censuras = {int(np.sum(1-e[gr==j]))}")

# ===================================================================
# PARTE 1 - a tabela do teste log-rank, tempo a tempo
# ===================================================================
L, E1, V1, w, O_E, V, Q, p = testes(t, e, gr, rho=0.0)
x, n_, d_, n1, d1 = L.T
print("\nTabela do log-rank: a cada tempo de evento, compara-se o observado no")
print("grupo 6-MP com o esperado sob a hipótese de curvas iguais.")
print(f"{'tempo':>7}{'n total':>9}{'d total':>9}{'n 6-MP':>9}{'d 6-MP':>9}"
      f"{'E(d 6-MP)':>12}{'d - E':>10}{'Var':>10}")
for j in range(len(x)):
    print(f"{x[j]:>7.0f}{n_[j]:>9.0f}{d_[j]:>9.0f}{n1[j]:>9.0f}{d1[j]:>9.0f}"
          f"{E1[j]:>12.4f}{d1[j]-E1[j]:>10.4f}{V1[j]:>10.4f}")
print(f"{'soma':>7}{'':>9}{d_.sum():>9.0f}{'':>9}{d1.sum():>9.0f}"
      f"{E1.sum():>12.4f}{O_E:>10.4f}{V:>10.4f}")
print(f"\nObservado no 6-MP: O1 = {d1.sum():.0f}   |   Esperado: E1 = {E1.sum():.4f}")
print(f"O - E = {O_E:.4f}  |  Variância = {V:.4f}")
print(f"Q = (O - E)² / V = {O_E:.4f}² / {V:.4f} = {Q:.4f}")
print(f"gl = 1  |  valor-p = {p:.4e}")
print("\nO grupo 6-MP teve 9 recaídas onde se esperariam 19,25 sob a hipótese de")
print("que os dois tratamentos são equivalentes. A diferença é grande demais")
print("para ser atribuída ao acaso.")

# ===================================================================
# PARTE 2 - razão de riscos
# ===================================================================
print("\n" + "=" * 70)
O2 = int(np.sum(e[gr == 1])); E2 = d_.sum() - E1.sum()
print("Razão de riscos (hazard ratio), placebo em relação ao 6-MP")
hr_oe = (O2/E2) / (d1.sum()/E1.sum())
print(f"  pela razão O/E:            {O2}/{E2:.4f} dividido por "
      f"{d1.sum():.0f}/{E1.sum():.4f} = {hr_oe:.4f}")
lhr = -O_E / V                                   # estimativa de Peto, log(HR)
ep_lhr = 1/np.sqrt(V)
print(f"  estimativa de Peto:        exp({lhr:.4f}) = {np.exp(lhr):.4f}")
print(f"  IC 95%: [{np.exp(lhr-1.96*ep_lhr):.4f}; {np.exp(lhr+1.96*ep_lhr):.4f}]")
print("  (a ficha da regressão de Cox estima a mesma quantidade por máxima")
print("   verossimilhança parcial e chega a um valor próximo)")
print(f"  Interpretação: a cada instante, quem está no placebo tem cerca de")
print(f"  {np.exp(lhr):.1f} vezes o risco instantâneo de recair de quem está no 6-MP.")

# ===================================================================
# PARTE 3 - log-rank, Gehan e Peto-Peto nos mesmos dados
# ===================================================================
print("\n" + "=" * 70)
print("A família de testes ponderados: o peso decide onde o teste 'olha'.")
def gehan(t, e, gr):
    """Wilcoxon-Gehan: peso igual ao número em risco."""
    L = tabela_tempos(t, e, gr); x, n_, d_, n1, d1 = L.T
    E1 = d_*n1/n_
    V1 = d_*(n1/n_)*(1-n1/n_)*(n_-d_)/np.maximum(n_-1, 1)
    O_E = np.sum(n_*(d1-E1)); V = np.sum(n_**2*V1)
    Q = O_E**2/V
    return O_E, V, Q, stats.chi2.sf(Q, 1)

print(f"\n{'teste':<26}{'peso w(t)':<22}{'Q':>10}{'valor-p':>14}")
_, _, _, _, oe0, v0, q0, p0 = testes(t, e, gr, 0.0)
print(f"{'log-rank (Mantel-Cox)':<26}{'1':<22}{q0:>10.4f}{p0:>14.4e}")
_, _, _, _, oe1, v1, q1, p1 = testes(t, e, gr, 1.0)
print(f"{'Peto-Peto (G-rho, rho=1)':<26}{'S(t-)':<22}{q1:>10.4f}{p1:>14.4e}")
oeg, vg, qg, pg = gehan(t, e, gr)
print(f"{'Wilcoxon-Gehan':<26}{'n em risco':<22}{qg:>10.4f}{pg:>14.4e}")
_, _, _, _, oe5, v5, q5, p5 = testes(t, e, gr, 0.5)
print(f"{'G-rho com rho = 0,5':<26}{'raiz de S(t-)':<22}{q5:>10.4f}{p5:>14.4e}")
print("\nAqui os quatro concordam: a diferença entre os tratamentos é grande e")
print("persiste em toda a faixa de tempo, de modo que nenhuma escolha de peso")
print("muda a conclusão. A ficha do Wilcoxon-Gehan mostra o caso em que muda.")

# ===================================================================
# PARTE 4 - o pressuposto de riscos proporcionais
# ===================================================================
print("\n" + "=" * 70)
print("O log-rank tem poder MÁXIMO quando os riscos são proporcionais, e perde")
print("poder quando não são. Simulação com 2.000 experimentos por cenário,")
print("n = 60 por grupo, 20% de censura administrativa.")

def pwexp(rng, n, a, b, c):
    """Tempos de um risco CONSTANTE POR PARTES: a até c, b depois de c."""
    E = rng.exponential(1, n)                 # risco acumulado até o evento
    return np.where(E <= a*c, E/a, c + (E - a*c)/b)

def simula(cenario, rep=2000, n=60, semente=3):
    rng = np.random.default_rng(semente); cont = np.zeros(3)
    for _ in range(rep):
        T1 = rng.exponential(10, n)                       # risco 0,10 sempre
        if cenario == "nulo":
            T2 = rng.exponential(10, n)                   # idêntico
        elif cenario == "prop":
            T2 = rng.exponential(10/1.8, n)               # risco 0,18 sempre
        elif cenario == "cedo":
            T2 = pwexp(rng, n, 0.20, 0.10, 4.0)           # 0,20 até t = 4
        else:
            T2 = pwexp(rng, n, 0.10, 0.20, 8.0)           # 0,20 depois de t = 8
        C = rng.exponential(40, 2*n)
        TT = np.r_[T1, T2]; X = np.minimum(TT, C); E = (TT <= C).astype(int)
        G = np.r_[np.zeros(n), np.ones(n)].astype(int)
        cont[0] += testes(X, E, G, 0.0)[7] < 0.05
        cont[1] += gehan(X, E, G)[3] < 0.05
        cont[2] += testes(X, E, G, 1.0)[7] < 0.05
    return 100*cont/rep

print(f"\n{'cenário':<46}{'log-rank':>11}{'Gehan':>9}{'Peto-Peto':>12}")
for cen, nome in [("nulo", "H0 verdadeira: risco 0,10 nos dois"),
                  ("prop", "riscos proporcionais: 0,10 contra 0,18"),
                  ("cedo", "risco maior SÓ ATÉ t = 4 (0,20 e depois 0,10)"),
                  ("tardio", "risco maior SÓ APÓS t = 8 (0,10 e depois 0,20)")]:
    a, b, c = simula(cen)
    print(f"{nome:<46}{a:>10.2f}%{b:>8.2f}%{c:>11.2f}%")
print("\nNa primeira linha os três controlam o nível em 5%, como devem. Na")
print("segunda, com riscos proporcionais, o log-rank tem o maior poder: é para")
print("esse caso que ele foi construído, e nenhum teste ponderado o supera ali.")
print("Nas duas últimas a ordem se inverte de forma nítida. Quando a diferença")
print("está no INÍCIO, Gehan e Peto-Peto chegam a 61% e 60% contra 43% do")
print("log-rank, porque dão mais peso aos tempos em que há mais gente em risco.")
print("Quando a diferença está no FIM, o log-rank fica em 22% e os dois ponderados")
print("caem para 7,5% e 8,4% - quase o nível nominal, ou seja, praticamente cegos")
print("à diferença, porque os seus pesos já decaíram quando ela aparece. Note")
print("ainda que o poder é baixo nos três: detectar uma separação tardia exige")
print("acompanhamento longo. Nenhum dos três é o melhor em toda parte, e por isso a")
print("escolha do teste deve ser declarada ANTES de ver os dados.")
