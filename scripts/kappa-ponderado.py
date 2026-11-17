# Bibliotecas: numpy e scipy (ambas já instaladas no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# Dados fictícios, criados pelo autor, compartilhados pelas fichas de
# concordância: 40 folhas de cafeeiro avaliadas quanto à severidade de
# ferrugem por cinco avaliadores, em escala de 1 (ausente) a 5 (muito alta).
# ===================================================================
SEV = np.array([
 [3,3,4,3,4],[5,4,4,4,5],[4,4,5,3,5],[5,5,5,4,5],[5,5,5,4,3],
 [1,2,2,1,1],[2,2,3,2,4],[1,2,1,1,1],[3,3,4,3,4],[3,4,5,4,2],
 [4,3,4,3,4],[3,4,4,3,3],[1,1,1,1,1],[3,3,5,3,3],[2,3,2,2,1],
 [5,5,5,5,4],[5,5,4,3,5],[4,5,5,4,4],[2,2,5,2,2],[3,4,5,4,3],
 [4,4,4,4,1],[1,2,3,1,3],[3,3,3,3,3],[2,3,5,2,3],[2,1,2,2,2],
 [2,1,3,2,3],[3,3,4,3,1],[2,1,1,1,5],[3,2,4,5,4],[1,1,1,1,2],
 [4,5,5,4,4],[2,2,4,2,1],[1,1,4,1,3],[1,1,2,1,1],[3,3,3,2,3],
 [3,3,3,3,3],[3,5,4,3,2],[2,2,4,2,1],[4,2,5,4,4],[5,5,5,5,2]])
AVAL = ["A", "B", "C", "D", "E"]

def tabela(a, b, k):
    O = np.zeros((k, k), int)
    for x, y in zip(a, b): O[x-1, y-1] += 1
    return O

def kappa_w(O, w):
    """Kappa ponderado. w é a matriz de DESACORDO: 0 na diagonal, 1 no pior caso."""
    O = np.asarray(O, float); N = O.sum(); P = O/N
    E = np.outer(P.sum(1), P.sum(0))
    po = 1 - (w*P).sum(); pe = 1 - (w*E).sum()
    return (po - pe)/(1 - pe), po, pe

def pesos(k, tipo):
    i, j = np.arange(k)[:, None], np.arange(k)[None, :]
    if tipo == "nominal":    return (i != j).astype(float)
    if tipo == "linear":     return np.abs(i - j)/(k - 1)
    if tipo == "quadratico": return (i - j)**2/(k - 1)**2
    raise ValueError(tipo)

def ep_kappa_w(O, w, B=20000, semente=3):
    """Erro-padrão por bootstrap sobre as unidades (simples e geral)."""
    O = np.asarray(O, float); k = O.shape[0]
    cel = np.array([(i, j) for i in range(k) for j in range(k)])
    pr = (O/O.sum()).ravel(); N = int(O.sum())
    rng = np.random.default_rng(semente); out = []
    for _ in range(B):
        c = rng.multinomial(N, pr).reshape(k, k)
        try: out.append(kappa_w(c, w)[0])
        except Exception: pass
    return np.std(out, ddof=1), np.percentile(out, [2.5, 97.5])

def icc_21(Y):
    """ICC(2,1): duas vias, efeito aleatório, medida única, concordância absoluta."""
    Y = np.asarray(Y, float); n, k = Y.shape; gm = Y.mean()
    MSR = k*((Y.mean(1)-gm)**2).sum()/(n-1)
    MSC = n*((Y.mean(0)-gm)**2).sum()/(k-1)
    MSE = ((Y - Y.mean(1)[:,None] - Y.mean(0)[None,:] + gm)**2).sum()/((n-1)*(k-1))
    return (MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n)

# ===================================================================
# PARTE 1 - a mesma tabela, três leituras
# ===================================================================
print("Severidade de ferrugem em 40 folhas, avaliadores A e B, escala de 1 a 5")
O = tabela(SEV[:,0], SEV[:,1], 5)
print(f"\n{'':>12}" + "".join(f"{'B='+str(j+1):>7}" for j in range(5)) + f"{'total':>8}")
for i in range(5):
    print(f"{'A='+str(i+1):>12}" + "".join(f"{O[i,j]:>7d}" for j in range(5))
          + f"{O[i].sum():>8d}")
print(f"{'total':>12}" + "".join(f"{O[:,j].sum():>7d}" for j in range(5))
      + f"{O.sum():>8d}")
conc = np.trace(O); umaclasse = sum(O[i,j] for i in range(5) for j in range(5)
                                    if abs(i-j) == 1)
print(f"\nConcordância exata: {conc} folhas ({100*conc/40:.1f}%)")
print(f"Desacordo de UMA classe: {umaclasse} folhas ({100*umaclasse/40:.1f}%)")
print(f"Desacordo de duas ou mais: {40-conc-umaclasse} folhas "
      f"({100*(40-conc-umaclasse)/40:.1f}%)")
print("Quatro em cada dez folhas têm desacordo de apenas uma classe. O kappa")
print("não ponderado conta esses casos como desacordo pleno, igual a quem")
print("discordou por quatro classes.")

print("\nAs três matrizes de peso (desacordo), para k = 5")
for tipo in ["nominal", "linear", "quadratico"]:
    w = pesos(5, tipo)
    print(f"\n  {tipo}")
    for i in range(5):
        print("    " + "  ".join(f"{w[i,j]:5.3f}" for j in range(5)))

print("\nOs três kappas sobre a MESMA tabela")
print(f"{'peso':<14}{'po':>10}{'pe':>10}{'kappa':>10}{'EP (bootstrap)':>17}"
      f"{'IC 95% percentil':>26}")
res = {}
for tipo in ["nominal", "linear", "quadratico"]:
    w = pesos(5, tipo)
    K, po, pe = kappa_w(O, w); ep, ic = ep_kappa_w(O, w)
    res[tipo] = K
    print(f"{tipo:<14}{po:>10.4f}{pe:>10.4f}{K:>10.4f}{ep:>17.4f}"
          f"   [{ic[0]:6.4f}; {ic[1]:6.4f}]")
print("\nO kappa sobe de 0,43 para 0,83 apenas por mudar o peso: a escolha do")
print("esquema de pesos é uma decisão substantiva, não um detalhe técnico, e")
print("tem de ser declarada junto com o valor.")

# ===================================================================
# PARTE 2 - por que o peso quadrático sobe tanto
# ===================================================================
print("\n" + "=" * 70)
print("O peso quadrático penaliza pouco os desacordos pequenos.")
print(f"\n{'distância entre as classes':<30}{'linear':>10}{'quadrático':>14}"
      f"{'folhas':>9}")
for d in range(5):
    wl = d/4; wq = d**2/16
    nf = sum(O[i,j] for i in range(5) for j in range(5) if abs(i-j) == d)
    print(f"{'desacordo de ' + str(d) + ' classe(s)':<30}{wl:>10.4f}{wq:>14.4f}{nf:>9d}")
print("\nUm desacordo de uma classe vale 0,25 no esquema linear e apenas 0,0625")
print("no quadrático: quatro vezes menos. Como esse é o desacordo mais comum")
print("(16 das 40 folhas), o kappa quadrático é sistematicamente maior.")
print("A pergunta a fazer é: no SEU problema, errar por duas classes é duas")
print("vezes pior que errar por uma (linear) ou quatro vezes pior (quadrático)?")

# ===================================================================
# PARTE 3 - a identidade de Fleiss e Cohen
# ===================================================================
print("\n" + "=" * 70)
print("Fleiss e Cohen (1973): o kappa com pesos QUADRÁTICOS é essencialmente")
print("o coeficiente de correlação intraclasse calculado sobre os mesmos dados.")
kq = res["quadratico"]
ic21 = icc_21(SEV[:, :2])
print(f"\n  kappa quadrático de A e B = {kq:.5f}")
print(f"  ICC(2,1) de A e B         = {ic21:.5f}")
print(f"  diferença                 = {abs(kq-ic21):.5f}")
print("A diferença está na terceira casa decimal e vem só da convenção de")
print("usar n ou n − 1 no denominador das variâncias; as duas quantidades")
print("coincidem quando n cresce. Isso tem uma consequência prática: a escolha")
print("do peso quadrático equivale a tratar a escala ordinal como se fosse")
print("intervalar, e quem não aceita essa suposição deve preferir o linear.")

print("\nA verificação vale para todos os pares de avaliadores:")
print(f"{'par':>6}{'kappa quadrático':>20}{'ICC(2,1)':>12}{'diferença':>12}")
AV = ["A","B","C","D","E"]
for i in range(5):
    for j in range(i+1, 5):
        Oij = tabela(SEV[:,i], SEV[:,j], 5)
        kqij = kappa_w(Oij, pesos(5, "quadratico"))[0]
        icij = icc_21(SEV[:, [i, j]])
        print(f"{AV[i]+'-'+AV[j]:>6}{kqij:>20.5f}{icij:>12.5f}{abs(kqij-icij):>12.5f}")

# ===================================================================
# PARTE 4 - todos os pares, nos três esquemas
# ===================================================================
print("\n" + "=" * 70)
print("Os dez pares de avaliadores, nos três esquemas de peso")
print(f"\n{'par':>6}{'nominal':>11}{'linear':>11}{'quadrático':>13}"
      f"{'ordem nominal':>16}{'ordem quadrático':>19}")
linhas = []
for i in range(5):
    for j in range(i+1, 5):
        Oij = tabela(SEV[:,i], SEV[:,j], 5)
        linhas.append((AV[i]+"-"+AV[j],
                       kappa_w(Oij, pesos(5,"nominal"))[0],
                       kappa_w(Oij, pesos(5,"linear"))[0],
                       kappa_w(Oij, pesos(5,"quadratico"))[0]))
kn = np.array([l[1] for l in linhas]); kq_ = np.array([l[3] for l in linhas])
on = len(kn)+1-stats.rankdata(kn); oq = len(kq_)+1-stats.rankdata(kq_)
for t_, a, b, c in zip(linhas, kn, kq_, range(10)):
    print(f"{t_[0]:>6}{t_[1]:>11.4f}{t_[2]:>11.4f}{t_[3]:>13.4f}"
          f"{int(on[c]):>16d}{int(oq[c]):>19d}")
rho = stats.spearmanr(kn, kq_).statistic
print(f"\nCorrelação de Spearman entre as duas ordenações = {rho:.4f}")
trocas = int(np.sum(on != oq))
print(f"Pares que mudam de posição no ranking: {trocas} de 10")
print("A escolha do peso muda os valores de forma drástica - A-D passa de 0,65")
print("para 0,87 - e também mexe na ORDEM: oito dos dez pares trocam de posição.")
print("O caso mais visível é C-D, que sobe do oitavo para o quinto lugar,")
print("porque os desacordos entre C e D são quase todos de uma classe só; e")
print("C-E, que cai do sétimo para o décimo, porque os desacordos ali são")
print("grandes. A correlação de 0,81 mostra que o quadro geral se mantém, mas")
print("seria errado dizer que o esquema de pesos é indiferente ao ranking.")

# ===================================================================
# PARTE 5 - quando o peso realmente muda a conclusão
# ===================================================================
print("\n" + "=" * 70)
print("Dois padrões de desacordo com a MESMA concordância exata, para mostrar")
print("quando o esquema de pesos decide. Em cada tabela, 20 das 40 folhas")
print("estão na diagonal; o que muda é o tamanho dos desacordos.")
T_perto = np.zeros((5,5), int); T_longe = np.zeros((5,5), int)
for i in range(5):
    T_perto[i,i] = 4; T_longe[i,i] = 4
# desacordos de UMA classe
T_perto[0,1]=5; T_perto[1,2]=5; T_perto[2,3]=5; T_perto[3,4]=5
# desacordos de TRÊS ou QUATRO classes
T_longe[0,4]=5; T_longe[1,4]=5; T_longe[4,0]=5; T_longe[3,0]=5
for nome, T in [("desacordos pequenos (sempre 1 classe)", T_perto),
                ("desacordos grandes (3 ou 4 classes)", T_longe)]:
    kn_ = kappa_w(T, pesos(5,"nominal"))[0]
    kl_ = kappa_w(T, pesos(5,"linear"))[0]
    kqq = kappa_w(T, pesos(5,"quadratico"))[0]
    print(f"\n  {nome}")
    print(f"    concordância exata = {np.trace(T)}/40 = {np.trace(T)/40:.2f}")
    print(f"    kappa nominal = {kn_:.4f} | linear = {kl_:.4f} | "
          f"quadrático = {kqq:.4f}")
print("\nO kappa não ponderado é praticamente o mesmo nas duas tabelas, porque")
print("ele só enxerga a diagonal. Os ponderados separam os dois casos com")
print("clareza, e é exatamente para isso que existem.")
