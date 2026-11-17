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

def kappa(O, w=None):
    """Kappa de Cohen. w = matriz de DESACORDO (0 na diagonal)."""
    O = np.asarray(O, float); k = O.shape[0]; N = O.sum()
    P = O / N; lin = P.sum(1); col = P.sum(0); E = np.outer(lin, col)
    if w is None: w = 1 - np.eye(k)
    po = 1 - (w*P).sum(); pe = 1 - (w*E).sum()
    return (po - pe) / (1 - pe), po, pe

def ep_kappa(O):
    """Erro-padrão assintótico do kappa não ponderado (Fleiss, Cohen e Everitt)."""
    O = np.asarray(O, float); k = O.shape[0]; N = O.sum()
    P = O/N; lin = P.sum(1); col = P.sum(0)
    po = np.trace(P); pe = (lin*col).sum(); K = (po-pe)/(1-pe)
    t1 = sum(P[i,i]*(1-(lin[i]+col[i])*(1-K))**2 for i in range(k))
    t2 = (1-K)**2*sum(P[i,j]*(col[i]+lin[j])**2
                      for i in range(k) for j in range(k) if i != j)
    t3 = (K-pe*(1-K))**2
    return np.sqrt((t1+t2-t3)/(N*(1-pe)**2))

# ===================================================================
# PARTE 1 - duas avaliações, escala dicotomizada
# ===================================================================
print("40 folhas, severidade de ferrugem classificada por cinco avaliadores")
print("Para o kappa de Cohen, que é para DOIS avaliadores e escala nominal,")
print("comparamos A e B, dicotomizando em 'severidade alta' (classes 4 e 5)")
print("contra 'severidade baixa' (classes 1 a 3).")
dA = (SEV[:,0] >= 4).astype(int); dB = (SEV[:,1] >= 4).astype(int)
O = tabela(dA+1, dB+1, 2)
print(f"\n{'':>18}{'B: baixa':>12}{'B: alta':>10}{'total':>9}")
for i, nm in enumerate(["A: baixa", "A: alta"]):
    print(f"{nm:>18}{O[i,0]:>12d}{O[i,1]:>10d}{O[i].sum():>9d}")
print(f"{'total':>18}{O[:,0].sum():>12d}{O[:,1].sum():>10d}{O.sum():>9d}")

K, po, pe = kappa(O)
ep = ep_kappa(O)
print(f"\nConcordância observada  po = ({O[0,0]} + {O[1,1]}) / 40 = {po:.4f}")
print(f"Concordância esperada   pe = {pe:.4f}")
print(f"Kappa = (po - pe)/(1 - pe) = ({po:.4f} - {pe:.4f})/(1 - {pe:.4f}) = {K:.4f}")
print(f"Erro-padrão = {ep:.4f} | IC 95% = [{K-1.96*ep:.4f}; {K+1.96*ep:.4f}]")
print("O que se reporta é o intervalo, e não um teste de kappa = 0: ninguém")
print("espera que dois avaliadores treinados concordem apenas por acaso.")

# ===================================================================
# PARTE 2 - os dois paradoxos do kappa
# ===================================================================
print("\n" + "=" * 70)
print("Os dois paradoxos de Feinstein e Cicchetti (1990)")
print("\n(a) PREVALÊNCIA: concordância altíssima, kappa baixo")
T1 = np.array([[85, 5], [5, 5]])
K1, po1, pe1 = kappa(T1)
print(f"    tabela: {T1.tolist()}")
print(f"    concordância observada = {po1:.4f} ({po1*100:.0f}%)")
print(f"    concordância esperada  = {pe1:.4f}")
print(f"    kappa = {K1:.4f}")
print("    Quase todas as unidades estão na mesma célula: o acaso já explica")
print("    82% da concordância, e sobra pouco para o kappa creditar.")

print("\n(b) VIÉS: a mesma concordância observada, kappas muito diferentes")
for T in [np.array([[45, 15], [25, 15]]), np.array([[25, 35], [5, 35]])]:
    Kx, pox, pex = kappa(T)
    m1 = T[0].sum()/T.sum(); m2 = T[:,0].sum()/T.sum()
    print(f"    tabela: {T.tolist()}  po = {pox:.4f}  pe = {pex:.4f}  "
          f"kappa = {Kx:.4f}")
    print(f"      marginais: avaliador 1 diz 'sim' em {100*m1:.0f}%, "
          f"avaliador 2 em {100*m2:.0f}%")
print("    A mesma concordância de 60% dá kappa 0,13 e 0,26: quanto MAIS")
print("    diferentes forem as marginais, MAIOR o kappa, para po fixo.")

print("\n(c) Índices que contornam o problema")
for nome, T in [("prevalência extrema", np.array([[85,5],[5,5]])),
                ("viés alto", np.array([[25,35],[5,35]])),
                ("A e B desta ficha", O)]:
    N = T.sum(); po_ = np.trace(T)/N; Kx, _, _ = kappa(T)
    pabak = 2*po_ - 1
    ip = abs(T[0,0]-T[1,1])/N          # índice de prevalência
    ib = abs(T[0,1]-T[1,0])/N          # índice de viés
    print(f"    {nome:<22} kappa = {Kx:6.4f}  PABAK = {pabak:6.4f}  "
          f"IP = {ip:.3f}  IV = {ib:.3f}")
print("    O PABAK = 2·po − 1 ignora as marginais; os índices de prevalência")
print("    e de viés dizem QUAL dos dois problemas está em jogo. Reportá-los")
print("    ao lado do kappa é a recomendação de Byrt, Bishop e Carlin (1993).")

# ===================================================================
# PARTE 3 - todos os pares de avaliadores, na escala completa
# ===================================================================
print("\n" + "=" * 70)
print("Kappa NÃO PONDERADO entre todos os pares, nas cinco classes originais")
print(f"\n{'par':>8}{'concordância':>15}{'kappa':>10}{'EP':>9}{'IC 95%':>22}")
pares = [(i, j) for i in range(5) for j in range(i+1, 5)]
ks = []
for i, j in pares:
    Oij = tabela(SEV[:,i], SEV[:,j], 5)
    Kij, poij, peij = kappa(Oij); epij = ep_kappa(Oij); ks.append(Kij)
    print(f"{AVAL[i]+'-'+AVAL[j]:>8}{poij:>15.4f}{Kij:>10.4f}{epij:>9.4f}"
          f"   [{Kij-1.96*epij:6.4f}; {Kij+1.96*epij:6.4f}]")
print(f"\nMédia dos dez kappas pareados (kappa de Light) = {np.mean(ks):.4f}")
print(f"Menor: {min(ks):.4f} | Maior: {max(ks):.4f}")
print("O avaliador C aparece nos pares de menor concordância: ele classifica")
print(f"em média {SEV[:,2].mean():.2f}, contra {SEV[:,[0,1,3,4]].mean():.2f} "
      f"dos outros quatro - é um viés sistemático de severidade.")

print("\nO kappa NÃO PONDERADO trata 'errar por uma classe' e 'errar por")
print("quatro' como o mesmo desacordo. Em escala ordinal isso desperdiça")
print("informação, e é o que a ficha do kappa ponderado corrige: para o par")
i, j = np.arange(5)[:,None], np.arange(5)[None,:]
Oab = tabela(SEV[:,0], SEV[:,1], 5)
print(f"  A-B, kappa não ponderado = {kappa(Oab)[0]:.4f}")
print(f"       kappa com pesos lineares    = {kappa(Oab, np.abs(i-j)/4)[0]:.4f}")
print(f"       kappa com pesos quadráticos = {kappa(Oab, (i-j)**2/16)[0]:.4f}")

# ===================================================================
# PARTE 4 - o que o kappa vale como número: a escala de Landis e Koch
# ===================================================================
print("\n" + "=" * 70)
print("A escala de Landis e Koch (1977) é convenção, não teoria.")
print("Simulação: qual kappa se obtém quando os dois avaliadores classificam")
print("INDEPENDENTEMENTE, mas com as mesmas marginais observadas em A e B?")
rng = np.random.default_rng(5)
mA = np.bincount(SEV[:,0], minlength=6)[1:]/40
mB = np.bincount(SEV[:,1], minlength=6)[1:]/40
sim = []
for _ in range(20000):
    a = rng.choice(5, 40, p=mA)+1; b = rng.choice(5, 40, p=mB)+1
    sim.append(kappa(tabela(a, b, 5))[0])
sim = np.array(sim)
print(f"  média = {sim.mean():.4f} (o valor esperado é 0)")
print(f"  desvio-padrão = {sim.std(ddof=1):.4f}")
print(f"  percentil 95 = {np.percentile(sim, 95):.4f}")
print(f"  proporção de amostras com kappa > 0,20 ('razoável') = "
      f"{100*np.mean(sim > 0.20):.2f}%")
print("Com n = 40 e cinco categorias, avaliadores que não se olham produzem")
print("kappa acima de 0,20 em cerca de 1% das amostras, e o percentil 95 fica")
print("em 0,135. Isso é tranquilizador quanto ao viés do estimador, mas o")
print("desvio-padrão de 0,08 mostra o outro lado: o rótulo verbal não")
print("substitui o intervalo de confiança, que é o que informa a precisão.")

print("\nE o tamanho da amostra? Largura do IC 95% do kappa de A e B,")
print("recalculado em subamostras de tamanhos diferentes (1.000 repetições):")
Oab_full = tabela(SEV[:,0], SEV[:,1], 5)
Kfull = kappa(Oab_full)[0]
print(f"{'n':>6}{'largura média do IC 95%':>28}")
for nn in [20, 40, 100, 400]:
    larg = []
    for _ in range(1000):
        idx = rng.choice(40, nn, replace=True)
        Oi = tabela(SEV[idx,0], SEV[idx,1], 5)
        try: larg.append(2*1.96*ep_kappa(Oi))
        except Exception: pass
    print(f"{nn:>6}{np.mean(larg):>27.4f}")
print(f"(o kappa observado nos 40 originais é {Kfull:.4f})")
print("A largura cai com a raiz de n, como de hábito: para estimar kappa com")
print("precisão de 0,10, são necessárias algumas centenas de unidades.")
