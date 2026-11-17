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

def contagens(X, k):
    n, m = X.shape; N = np.zeros((n, k))
    for i in range(n):
        for v in X[i]: N[i, v-1] += 1
    return N

def po_bar(N, m):
    """Concordância observada média: a mesma dos dois coeficientes."""
    return (((N**2).sum(1) - m)/(m*(m-1))).mean()

def ac1(X, k, w=None):
    """AC1 de Gwet (nominal) ou AC2 (com pesos de CONCORDÂNCIA)."""
    n, m = X.shape; N = contagens(X, k)
    if w is None: w = np.eye(k)                 # peso de CONCORDÂNCIA
    Tw = w.sum()
    po = np.mean([ (N[i] @ w @ N[i] - (np.diag(w) * N[i]).sum())
                   / (m*(m-1)) for i in range(n)])
    pi = N.sum(0)/(n*m)                          # prevalência de cada classe
    pe = Tw/(k*(k-1)) * sum(pi[j]*(1-pi[j]) for j in range(k)) if False else \
         (Tw/(k*(k-1))) * sum(pi[j]*(1-pi[j]) for j in range(k))
    return (po - pe)/(1 - pe), po, pe, pi

def fleiss(X, k):
    n, m = X.shape; N = contagens(X, k)
    pj = N.sum(0)/(n*m); Pb = po_bar(N, m); Pe = (pj**2).sum()
    return (Pb-Pe)/(1-Pe), Pb, Pe

def kappa2(O):
    O = np.asarray(O, float); P = O/O.sum()
    po = np.trace(P); pe = (P.sum(1)*P.sum(0)).sum()
    return (po-pe)/(1-pe), po, pe

def ac1_2(O):
    """AC1 para a tabela 2x2 de dois avaliadores."""
    O = np.asarray(O, float); k = 2; P = O/O.sum()
    po = np.trace(P)
    pi = (P.sum(1) + P.sum(0))/2
    pe = 2*pi[0]*pi[1]          # = (1/(k-1)) * soma pi(1-pi) com k = 2
    return (po-pe)/(1-pe), po, pe, pi

# ===================================================================
# PARTE 1 - o problema que o AC1 resolve
# ===================================================================
print("O paradoxo do kappa, revisitado")
print("Quando quase todas as unidades caem na mesma categoria, a concordância")
print("esperada por acaso do kappa fica enorme, e o coeficiente despenca mesmo")
print("com 90% de concordância observada.")
print(f"\n{'tabela':<22}{'po':>9}{'pe kappa':>11}{'kappa':>9}"
      f"{'pe AC1':>10}{'AC1':>9}")
exemplos = [("equilibrada", np.array([[45.,5.],[5.,45.]])),
            ("moderada", np.array([[60.,5.],[5.,30.]])),
            ("desequilibrada", np.array([[80.,5.],[5.,10.]])),
            ("muito desequilibrada", np.array([[85.,5.],[5.,5.]])),
            ("extrema", np.array([[92.,4.],[4.,0.]]))]
for nome, T in exemplos:
    Kx, pox, pex = kappa2(T); Ax, poa, pea, _ = ac1_2(T)
    print(f"{nome:<22}{pox:>9.4f}{pex:>11.4f}{Kx:>9.4f}{pea:>10.4f}{Ax:>9.4f}")
print("\nA concordância observada é 0,90 nas cinco primeiras linhas e 0,92 na")
print("última. O kappa varia de 0,80 a −0,04; o AC1 fica entre 0,80 e 0,92.")
print("A diferença está inteiramente no pe: o do kappa cresce com o")
print("desequilíbrio, o do AC1 DIMINUI - e é essa a tese de Gwet.")

print("\nPor que o pe do AC1 se comporta assim")
print("O kappa supõe que, ao não saber, o avaliador sorteia a categoria com")
print("as probabilidades das próprias marginais dele. Gwet argumenta que isso")
print("é implausível: um avaliador que não sabe tende a escolher ao acaso")
print("ENTRE AS CATEGORIAS PLAUSÍVEIS, e isso é mais provável quando as")
print("categorias são igualmente prováveis, não quando uma domina.")
print(f"\n{'prevalência da classe 1':>26}{'pe do kappa':>14}{'pe do AC1':>12}")
for p1 in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    pek = p1**2 + (1-p1)**2
    pea = 2*p1*(1-p1)
    print(f"{p1:>26.2f}{pek:>14.4f}{pea:>12.4f}")
print("Com 99% das unidades na classe 1, o kappa supõe que o acaso sozinho")
print("produziria 98% de concordância; o AC1 supõe 2%. A verdade costuma")
print("estar entre as duas, mas a do kappa é claramente a mais extrema.")

# ===================================================================
# PARTE 2 - AC1 nos dados das fichas de concordância
# ===================================================================
print("\n" + "=" * 70)
n, m = SEV.shape; k = 5
print(f"As mesmas {n} folhas e {m} avaliadores das fichas anteriores")
A, poA, peA, pi = ac1(SEV, k)
Kf, PbF, PeF = fleiss(SEV, k)
print(f"\nConcordância observada (idêntica nos dois)  po = {poA:.4f}")
print(f"Concordância esperada do kappa de Fleiss    pe = {PeF:.4f}")
print(f"Concordância esperada do AC1 de Gwet        pe = {peA:.4f}")
print(f"\n  kappa de Fleiss = {Kf:.4f}")
print(f"  AC1 de Gwet     = {A:.4f}")
print(f"  diferença       = {A-Kf:+.4f}")
print("\nAs cinco classes têm prevalências próximas:")
print("  " + "  ".join(f"classe {j+1}: {pi[j]:.3f}" for j in range(k)))
print("Sem uma classe dominante, o pe do AC1 (0,1989) e o do kappa (0,2042)")
print("quase coincidem, e os dois coeficientes também. É o resultado")
print("esperado: o AC1 não é 'um kappa maior', é um kappa com outra hipótese")
print("sobre o acaso, e as duas hipóteses só divergem no desequilíbrio.")

print("\nAgora o mesmo cálculo depois de colapsar a escala em 'sem sintoma'")
print("(classe 1) contra 'com sintoma' (classes 2 a 5), que é fortemente")
print("desequilibrada:")
S2 = np.where(SEV == 1, 1, 2)
A2, poA2, peA2, pi2 = ac1(S2, 2)
K2, Pb2, Pe2 = fleiss(S2, 2)
print(f"  prevalências: sem sintoma {pi2[0]:.3f}, com sintoma {pi2[1]:.3f}")
print(f"  concordância observada = {poA2:.4f}")
print(f"  pe do kappa = {Pe2:.4f}  ->  kappa de Fleiss = {K2:.4f}")
print(f"  pe do AC1   = {peA2:.4f}  ->  AC1 de Gwet     = {A2:.4f}")
print(f"  o AC1 fica {A2-K2:+.4f} acima do kappa")
print("Com 82,5% das avaliações em uma só classe, os dois se separam, e a")
print("separação cresce com o desequilíbrio.")

# ===================================================================
# PARTE 3 - AC2: a versão ponderada, para escalas ordinais
# ===================================================================
print("\n" + "=" * 70)
print("AC2: a versão com pesos, análoga ao kappa ponderado")
i, j = np.arange(k)[:, None], np.arange(k)[None, :]
pesos_conc = {"identidade (AC1)": np.eye(k),
              "linear":      1 - np.abs(i-j)/(k-1),
              "quadrático":  1 - (i-j)**2/(k-1)**2}
print(f"\n{'pesos':<22}{'po':>10}{'pe':>10}{'AC2':>10}")
for nome, w in pesos_conc.items():
    Ax, pox, pex, _ = ac1(SEV, k, w)
    print(f"{nome:<22}{pox:>10.4f}{pex:>10.4f}{Ax:>10.4f}")
print("\nAtenção à convenção: no kappa ponderado os pesos são de DESACORDO")
print("(0 na diagonal); no AC2 são de CONCORDÂNCIA (1 na diagonal). São o")
print("complemento um do outro, e trocá-los inverte o resultado.")

# ===================================================================
# PARTE 4 - os dois coeficientes sob variação de prevalência
# ===================================================================
print("\n" + "=" * 70)
print("Simulação: dois avaliadores com a MESMA qualidade - cada um acerta a")
print("classe verdadeira com probabilidade 0,90 e erra para a outra classe")
print("com 0,10 - aplicados a populações de prevalência crescente.")
print("A QUALIDADE DOS AVALIADORES NÃO MUDA entre as linhas.")
rng = np.random.default_rng(9)
print(f"\n{'prevalência':>12}{'po':>9}{'kappa':>10}{'AC1':>9}{'PABAK':>9}")
for prev in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    ks, as_, pb, ps = [], [], [], []
    for _ in range(3000):
        verdade = (rng.random(300) < prev).astype(int)
        r1 = np.where(rng.random(300) < 0.90, verdade, 1-verdade)
        r2 = np.where(rng.random(300) < 0.90, verdade, 1-verdade)
        T = np.zeros((2,2))
        for a_, b_ in zip(r1, r2): T[a_, b_] += 1
        if T.sum(0).min() == 0 or T.sum(1).min() == 0: continue
        Kx, pox, _ = kappa2(T); Ax, _, _, _ = ac1_2(T)
        ks.append(Kx); as_.append(Ax); pb.append(2*pox-1); ps.append(pox)
    print(f"{prev:>12.2f}{np.mean(ps):>9.4f}{np.mean(ks):>10.4f}"
          f"{np.mean(as_):>9.4f}{np.mean(pb):>9.4f}")
print("\nA simulação isola o fenômeno de forma limpa. A concordância observada")
print("é CONSTANTE em 0,82 nas sete linhas - e tem de ser, porque po = 0,9² +")
print("0,1² = 0,82 qualquer que seja a prevalência, dada a qualidade fixa dos")
print("avaliadores. O PABAK, que só depende de po, também fica constante em")
print("0,64, como deve.")
print("O kappa, porém, cai de 0,64 para 0,07 ao longo da tabela, sem que nada")
print("na qualidade dos avaliadores tenha mudado: é o paradoxo de prevalência")
print("em estado puro. O AC1 vai na direção oposta, de 0,64 para 0,78, porque")
print("o seu pe encolhe quando uma classe domina.")
print("Nenhum dos três é 'o certo'. O kappa responde: quanto da concordância")
print("sobra depois de descontar o que avaliadores com estas marginais")
print("produziriam sorteando? O AC1 responde: quanto sobra depois de descontar")
print("o que produziriam sorteando entre as classes plausíveis? O PABAK nem")
print("desconta. São três perguntas, e a escolha precisa ser declarada.")

# ===================================================================
# PARTE 5 - o que reportar
# ===================================================================
print("\n" + "=" * 70)
print("A recomendação prática: reportar po, a prevalência e DOIS coeficientes")
print("Para os dados desta ficha, em três versões da escala:")
print(f"\n{'escala':<30}{'po':>9}{'classe maior':>14}{'kappa':>9}{'AC1':>9}")
for rot, Z, kk in [("5 classes (original)", SEV, 5),
                   ("3 classes (1-2 / 3 / 4-5)",
                    np.where(SEV <= 2, 1, np.where(SEV == 3, 2, 3)), 3),
                   ("2 classes (1 / 2-5)", np.where(SEV == 1, 1, 2), 2),
                   ("2 classes (1-3 / 4-5)", np.where(SEV <= 3, 1, 2), 2)]:
    Ax, pox, pex, pix = ac1(Z, kk); Kx, _, _ = fleiss(Z, kk)
    print(f"{rot:<30}{pox:>9.4f}{pix.max():>14.3f}{Kx:>9.4f}{Ax:>9.4f}")
print("\nQuanto mais desequilibrada a escala, maior o afastamento entre os")
print("dois. Reportar apenas um deles, sem a prevalência ao lado, deixa o")
print("leitor sem como julgar se o número é alto ou baixo pelo motivo certo.")
