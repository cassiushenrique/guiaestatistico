# Bibliotecas: numpy, scipy e matplotlib (todas já no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# Funções comuns às fichas de sobrevivência
# ===================================================================
def tabela_tempos(t, e, gr):
    T = np.unique(t[e == 1]); L = []
    for x in T:
        L.append((x, np.sum(t >= x), np.sum((t == x) & (e == 1)),
                  np.sum((t >= x) & (gr == 0)), np.sum((t == x) & (e == 1) & (gr == 0))))
    return np.array(L, float)

def ponderado(t, e, gr, peso="logrank"):
    """Família de testes ponderados; o peso é o que distingue um do outro."""
    L = tabela_tempos(t, e, gr); x, n_, d_, n1, d1 = L.T
    E1 = d_*n1/n_
    V1 = d_*(n1/n_)*(1-n1/n_)*(n_-d_)/np.maximum(n_-1, 1)
    Sc = np.r_[1.0, np.cumprod(1 - d_/n_)[:-1]]        # S(t-) combinado
    w = {"logrank": np.ones_like(x), "gehan": n_, "peto": Sc,
         "tarone": np.sqrt(n_), "fleming05": np.sqrt(Sc)}[peso]
    O_E = np.sum(w*(d1-E1)); V = np.sum(w**2*V1); Q = O_E**2/V
    return x, n_, d_, n1, d1, E1, V1, w, O_E, V, Q, stats.chi2.sf(Q, 1)

def km(t, e):
    T = np.unique(t[e == 1])
    nr = np.array([np.sum(t >= x) for x in T])
    d = np.array([np.sum((t == x) & (e == 1)) for x in T])
    return T, np.cumprod(1 - d/nr)

def mann_whitney_censurado(t, e, gr):
    """A estatística de Gehan na forma ORIGINAL: contagem de pares.
       Para cada par (i do grupo 0, j do grupo 1) atribui-se +1 se i
       sobreviveu COMPROVADAMENTE mais, -1 se menos, 0 se indecidível."""
    i0 = np.where(gr == 0)[0]; i1 = np.where(gr == 1)[0]
    U = 0; indec = 0
    for i in i0:
        for j in i1:
            ti, ei, tj, ej = t[i], e[i], t[j], e[j]
            if   ti > tj and ej == 1: U += 1          # j falhou antes
            elif tj > ti and ei == 1: U -= 1          # i falhou antes
            elif ti == tj and ei == 0 and ej == 1: U += 1
            elif ti == tj and ei == 1 and ej == 0: U -= 1
            else: indec += 1
    return U, indec, len(i0)*len(i1)

# ===================================================================
# PARTE 1 - os dados de Freireich, os mesmos das duas fichas anteriores
# ===================================================================
t6  = np.array([6,6,6,6,7,9,10,10,11,13,16,17,19,20,22,23,25,32,32,34,35])
e6  = np.array([0,1,1,1,1,0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
tpl = np.array([1,1,2,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23])
epl = np.ones(21, int)
t = np.r_[t6, tpl]; e = np.r_[e6, epl]
gr = np.r_[np.zeros(21), np.ones(21)].astype(int)

print("Freireich et al. (1963): 6-mercaptopurina contra placebo")
x, n_, d_, n1, d1, E1, V1, w, O_E, V, Q, p = ponderado(t, e, gr, "gehan")
print("\nTabela do teste de Gehan: o peso de cada tempo é o número em risco")
print(f"{'tempo':>7}{'n em risco = w':>16}{'d 6-MP':>9}{'E(d 6-MP)':>12}"
      f"{'w (d - E)':>13}{'w² Var':>12}")
for j in range(len(x)):
    print(f"{x[j]:>7.0f}{n_[j]:>16.0f}{d1[j]:>9.0f}{E1[j]:>12.4f}"
          f"{w[j]*(d1[j]-E1[j]):>13.2f}{w[j]**2*V1[j]:>12.2f}")
print(f"{'soma':>7}{'':>16}{'':>9}{'':>12}{O_E:>13.2f}{V:>12.2f}")
print(f"\nQ = ({O_E:.2f})² / {V:.2f} = {Q:.4f} | gl = 1 | valor-p = {p:.4e}")

U, indec, total = mann_whitney_censurado(t, e, gr)
print(f"\nA mesma estatística pela contagem de pares (forma original de Gehan)")
print(f"  pares comparáveis: {total-indec} de {total}  "
      f"({100*(total-indec)/total:.1f}%)")
print(f"  pares indecidíveis por causa da censura: {indec}")
print(f"  U = (pares em que o 6-MP durou mais) - (pares em que durou menos) = {U}")
print(f"  A soma ponderada da tabela acima é {O_E:.0f}, e U = {U}: são a mesma")
print("  quantidade, escrita de duas maneiras. A forma de pares explica o nome")
print("  'Wilcoxon generalizado': sem censura, ela é exatamente a estatística de")
print("  Mann-Whitney.")

print("\nOs quatro testes nos mesmos dados")
print(f"{'teste':<24}{'peso w(t)':<20}{'Q':>10}{'valor-p':>14}")
for nome, pk, desc in [("log-rank", "logrank", "1"),
                       ("Wilcoxon-Gehan", "gehan", "n em risco"),
                       ("Tarone-Ware", "tarone", "raiz de n em risco"),
                       ("Peto-Peto", "peto", "S(t-)")]:
    *_, Qk, pk_ = ponderado(t, e, gr, pk)
    print(f"{nome:<24}{desc:<20}{Qk:>10.4f}{pk_:>14.4e}")
print("Todos rejeitam com folga: aqui a escolha do peso não muda nada.")

# ===================================================================
# PARTE 2 - o caso em que a escolha do peso MUDA a conclusão
# ===================================================================
print("\n" + "=" * 70)
print("Dados fictícios: 64 plantas de duas cultivares, acompanhadas por 36")
print("meses quanto ao tempo até a primeira infecção por um patógeno de solo.")
print("A cultivar B tem resistência inicial que se perde com o tempo.")
A_t = np.array([1,1,1,1,2,2,3,3,3,3,4,4,5,6,6,7,8,9,9,11,19,19,20,22,24,24,25,31,36,36,36,36])
A_e = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
B_t = np.array([6,7,9,11,11,12,12,13,13,14,14,14,14,15,16,16,16,18,19,20,20,20,20,21,21,22,24,25,25,28,31,33])
B_e = np.ones(32, int)
t2 = np.r_[A_t, B_t]; e2 = np.r_[A_e, B_e]
g2 = np.r_[np.zeros(32), np.ones(32)].astype(int)
print(f"\n  cultivar A: n = 32, infecções = {A_e.sum()}, censuras = {32-A_e.sum()}")
print(f"  cultivar B: n = 32, infecções = {B_e.sum()}, censuras = {32-B_e.sum()}")
TA, SA = km(A_t, A_e); TB, SB = km(B_t, B_e)
print("\nProporção sem infecção, pelo Kaplan-Meier")
print(f"{'mês':>6}{'cultivar A':>14}{'cultivar B':>14}{'diferença':>14}")
for mes in [3, 6, 12, 18, 24, 30, 36]:
    sa = SA[TA <= mes][-1] if np.any(TA <= mes) else 1.0
    sb = SB[TB <= mes][-1] if np.any(TB <= mes) else 1.0
    print(f"{mes:>6}{100*sa:>13.1f}%{100*sb:>13.1f}%{100*(sb-sa):>12.1f} p.p.")
print("A vantagem de B é de 43,8 pontos percentuais aos 6 meses, some entre os")
print("meses 18 e 24, e INVERTE-SE ao fim: é o padrão clássico de riscos não")
print("proporcionais, com curvas que se cruzam.")

print(f"\n{'teste':<24}{'peso w(t)':<20}{'Q':>10}{'valor-p':>12}{'decisão':>14}")
for nome, pk, desc in [("log-rank", "logrank", "1"),
                       ("Wilcoxon-Gehan", "gehan", "n em risco"),
                       ("Tarone-Ware", "tarone", "raiz de n em risco"),
                       ("Peto-Peto", "peto", "S(t-)"),
                       ("G-rho com rho = 0,5", "fleming05", "raiz de S(t-)")]:
    *_, Qk, pk_ = ponderado(t2, e2, g2, pk)
    print(f"{nome:<24}{desc:<20}{Qk:>10.4f}{pk_:>12.4f}"
          f"{('rejeita' if pk_ < 0.05 else 'não rejeita'):>14}")
print("\nO log-rank devolve Q = 0,15 e p = 0,70: nenhuma evidência. O teste de")
print("Gehan devolve p = 0,016 e rejeita. A diferença é real, mas está")
print("concentrada nos primeiros meses, e o log-rank a dilui entre os tempos")
print("tardios, em que a vantagem já se inverteu - as contribuições de sinais")
print("opostos se cancelam. Não há aqui um teste 'certo' e um 'errado': há duas")
print("perguntas diferentes, e escolher o teste depois de ver o gráfico é que")
print("seria errado.")
print("\nRepare que Gehan e Peto-Peto deram EXATAMENTE o mesmo Q, e Tarone-Ware e")
print("o G-rho de 0,5 também. Não é coincidência: como aqui não há censura antes")
print("do último evento, o número em risco é proporcional a S(t-), e a estatística")
print("não muda se todos os pesos forem multiplicados por uma constante. Nos")
print("dados de Freireich, que têm censura no meio do acompanhamento, os dois se")
print("separam: 13,46 contra 14,46.")

# ===================================================================
# PARTE 3 - o que relatar quando a razão de riscos não faz sentido
# ===================================================================
print("\n" + "=" * 70)
print("Com curvas que se cruzam, a razão de riscos deixa de fazer sentido: ela")
print("é maior que 1 no início e menor que 1 no fim. Convém relatar quantidades")
print("que não pressuponham proporcionalidade.")
def rmst(t, e, tau):
    T, S = km(t, e)
    xx = np.r_[0.0, T[T <= tau], tau]; yy = np.r_[1.0, S[T <= tau]]
    return float(np.sum(np.diff(xx)*yy))
print(f"\n{'tau (meses)':>12}{'RMST A':>10}{'RMST B':>10}{'diferença':>12}")
for tau in [6, 12, 24, 36]:
    ra, rb = rmst(A_t, A_e, tau), rmst(B_t, B_e, tau)
    print(f"{tau:>12}{ra:>10.2f}{rb:>10.2f}{rb-ra:>12.2f}")
print("O ganho médio de B depende inteiramente do horizonte escolhido, e declarar")
print("esse horizonte é parte da análise, não um detalhe técnico. A diferença de")
print("proporções em tempos fixos conta a mesma história de outra maneira.")

# PARTE 4 - as três estatísticas sob riscos proporcionais e não proporcionais
# ===================================================================
print("\n" + "=" * 70)
print("Poder dos testes em quatro cenários, 2.000 experimentos cada,")
print("n = 50 por grupo, censura administrativa aos 30 e perdas de seguimento")
print("aleatórias ao longo do estudo (para que Gehan e Peto-Peto não coincidam).")
def pwexp(rng, n, a, b, c):
    E = rng.exponential(1, n)
    return np.where(E <= a*c, E/a, c + (E - a*c)/b)
def simula(cen, rep=2000, n=50, semente=9):
    rng = np.random.default_rng(semente); cont = np.zeros(4)
    for _ in range(rep):
        T1 = rng.exponential(12, n)
        if   cen == "nulo":  T2 = rng.exponential(12, n)
        elif cen == "prop":  T2 = rng.exponential(12/1.7, n)
        elif cen == "cedo":  T2 = pwexp(rng, n, 0.18, 0.0833, 5.0)
        else:                T2 = pwexp(rng, n, 0.0833, 0.18, 10.0)
        TT = np.r_[T1, T2]
        C = np.minimum(rng.exponential(45, 2*n), 30.0)   # perda + fim do estudo
        X = np.minimum(TT, C); E = (TT <= C).astype(int)
        G = np.r_[np.zeros(n), np.ones(n)].astype(int)
        for k, pk in enumerate(["logrank", "gehan", "tarone", "peto"]):
            cont[k] += ponderado(X, E, G, pk)[11] < 0.05
    return 100*cont/rep
print(f"\n{'cenário':<40}{'log-rank':>10}{'Gehan':>9}{'Tarone':>9}{'Peto':>9}")
for cen, nome in [("nulo", "H0 verdadeira"),
                  ("prop", "riscos proporcionais (RR = 1,7)"),
                  ("cedo", "diferença só nos 5 primeiros meses"),
                  ("tardio", "diferença só depois do mês 10")]:
    v = simula(cen)
    print(f"{nome:<40}" + "".join(f"{q:>8.2f}%" for q in v))
print("\nA leitura é a mesma do exemplo das cultivares, agora em termos de poder.")
print("Tarone-Ware, com peso igual à raiz do número em risco, é o meio-termo:")
print("fica próximo do melhor em todos os cenários sem ser o melhor em nenhum.")
print("É por isso que vários autores o recomendam quando não se sabe de antemão")
print("onde a diferença estará. Se houver hipótese prévia sobre ONDE a diferença")
print("deve aparecer, o teste correspondente é mais poderoso - mas essa hipótese")
print("precisa estar no protocolo, não ser inventada depois do gráfico.")
