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
    """Matriz n x k com quantos avaliadores puseram cada unidade em cada classe."""
    n, m = X.shape
    N = np.zeros((n, k))
    for i in range(n):
        for v in X[i]: N[i, v-1] += 1
    return N

def fleiss(X, k):
    """Kappa de Fleiss para m avaliadores e k categorias."""
    n, m = X.shape
    N = contagens(X, k)
    pj = N.sum(0)/(n*m)                       # proporção global de cada classe
    Pi = ((N**2).sum(1) - m)/(m*(m-1))        # concordância dentro de cada unidade
    Pb = Pi.mean(); Pe = (pj**2).sum()
    return (Pb-Pe)/(1-Pe), Pb, Pe, pj, Pi, N

def ep_fleiss(X, k):
    """Erro-padrão assintótico de Fleiss (1971), válido sob H0: kappa = 0."""
    n, m = X.shape
    N = contagens(X, k); pj = N.sum(0)/(n*m); Pe = (pj**2).sum()
    num = 2*(Pe - (2*m-3)*Pe**2 + 2*(m-2)*(pj**3).sum())
    return np.sqrt(num/(n*m*(m-1)))/(1-Pe)

def kappa_categoria(X, k, j):
    """Kappa de Fleiss para UMA categoria, colapsando as demais."""
    Xb = np.where(X == j+1, 1, 2)
    return fleiss(Xb, 2)[0]

def kappa_cohen(a, b, k):
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[x-1, y-1] += 1
    P = O/O.sum(); po = np.trace(P); pe = (P.sum(1)*P.sum(0)).sum()
    return (po-pe)/(1-pe)

# ===================================================================
# PARTE 1 - o kappa de Fleiss nos cinco avaliadores
# ===================================================================
n, m = SEV.shape; k = 5
print(f"{n} folhas, {m} avaliadores, {k} classes de severidade")
print("Cada folha foi avaliada pelos CINCO avaliadores: é o delineamento para")
print("o qual o kappa de Fleiss foi construído.")

K, Pb, Pe, pj, Pi, N = fleiss(SEV, k)
print(f"\nProporção global de cada classe (pj)")
for j in range(k):
    print(f"  classe {j+1}: {int(N[:,j].sum()):>3d} atribuições de {n*m}"
          f"  -> pj = {pj[j]:.4f}")
print(f"  soma = {pj.sum():.4f}  |  Pe = soma dos pj² = {Pe:.4f}")

print(f"\nConcordância dentro de cada folha (Pi), primeiras oito folhas")
print(f"{'folha':>6}{'classificações':>22}{'contagens por classe':>26}{'Pi':>9}")
for i in range(8):
    print(f"{i+1:>6}{str(SEV[i].tolist()):>22}"
          f"{str([int(v) for v in N[i]]):>26}{Pi[i]:>9.4f}")
print(f"...")
print(f"  média dos 40 valores: Pbarra = {Pb:.4f}")
print(f"\nKappa de Fleiss = (Pbarra - Pe)/(1 - Pe) = ({Pb:.4f} - {Pe:.4f})/"
      f"(1 - {Pe:.4f}) = {K:.4f}")
ep0 = ep_fleiss(SEV, k)
print(f"Erro-padrão sob H0 = {ep0:.4f} | z = {K/ep0:.4f} | "
      f"p = {2*stats.norm.sf(abs(K/ep0)):.3e}")

# intervalo por bootstrap sobre as folhas
rng = np.random.default_rng(7)
bs = []
for _ in range(5000):
    idx = rng.choice(n, n, replace=True)
    bs.append(fleiss(SEV[idx], k)[0])
print(f"IC 95% por bootstrap das folhas = [{np.percentile(bs,2.5):.4f}; "
      f"{np.percentile(bs,97.5):.4f}]  (EP = {np.std(bs, ddof=1):.4f})")
print("O erro-padrão de Fleiss vale só sob a hipótese de kappa = 0, e serve")
print("para o teste, não para o intervalo. Para o intervalo, use bootstrap.")

# ===================================================================
# PARTE 2 - kappa por categoria: onde está o desacordo
# ===================================================================
print("\n" + "=" * 70)
print("Kappa de cada categoria isoladamente: mostra QUAIS classes são")
print("difíceis de reconhecer.")
print(f"\n{'classe':>24}{'atribuições':>14}{'kappa da classe':>18}")
nomes = ["1 ausente", "2 baixa", "3 média", "4 alta", "5 muito alta"]
for j in range(k):
    print(f"{nomes[j]:>24}{int(N[:,j].sum()):>14d}{kappa_categoria(SEV,k,j):>18.4f}")
print("\nA classe 1, ausência de sintoma, é de longe a mais reconhecível (0,446):")
print("é a única decisão qualitativamente diferente das demais. A classe 4 é a")
print("mais confundida (0,141), e as classes 2, 3 e 5 ficam entre 0,22 e 0,33.")
print("O padrão não é simplesmente 'extremos fáceis, meio difícil': a classe 5")
print("também é confundida, porque a fronteira entre 'alta' e 'muito alta' é")
print("tão subjetiva quanto a do meio da escala. Em um treinamento de")
print("avaliadores, é nessas fronteiras que se investe.")

# ===================================================================
# PARTE 3 - Fleiss, Light e os kappas pareados
# ===================================================================
print("\n" + "=" * 70)
print("Três maneiras de resumir a concordância de cinco avaliadores")
pares = [(i,j) for i in range(5) for j in range(i+1,5)]
kp = np.array([kappa_cohen(SEV[:,i], SEV[:,j], k) for i,j in pares])
print(f"\n  kappa de Fleiss                        = {K:.4f}")
print(f"  kappa de Light (média dos 10 pareados) = {kp.mean():.4f}")
print(f"  menor e maior kappa pareado            = {kp.min():.4f} e {kp.max():.4f}")
print("Fleiss e Light quase coincidem aqui, mas respondem a perguntas")
print("ligeiramente diferentes: Fleiss supõe avaliadores INTERCAMBIÁVEIS e usa")
print("as marginais globais; Light calcula um kappa por par, com as marginais")
print("de cada par, e tira a média. Quando um avaliador tem viés forte, os dois")
print("se separam.")

print("\nDeixando cada avaliador de fora, um de cada vez:")
print(f"{'sem o avaliador':>18}{'kappa de Fleiss':>18}{'variação':>12}")
for j in range(5):
    resto = [c for c in range(5) if c != j]
    Kj = fleiss(SEV[:, resto], k)[0]
    print(f"{AVAL[j]:>18}{Kj:>18.4f}{Kj-K:>+12.4f}")
print(f"\nMédia das classificações de cada avaliador: "
      + ", ".join(f"{AVAL[j]} {SEV[:,j].mean():.2f}" for j in range(5)))
print("Tirar o avaliador C eleva o kappa de forma expressiva: ele classifica")
print("em média 3,67, quase uma classe acima dos demais. Tirar E também eleva,")
print("por outro motivo - E é o mais errático, e não o mais severo.")

# ===================================================================
# PARTE 4 - quantos avaliadores e quantas unidades?
# ===================================================================
print("\n" + "=" * 70)
print("O kappa de Fleiss é afetado pelo NÚMERO DE CATEGORIAS: com mais")
print("classes, a concordância esperada por acaso cai, mas a tarefa fica mais")
print("difícil. Colapsando a escala de 5 para 3 e para 2 classes:")
print(f"\n{'escala':<34}{'Pe':>10}{'Pbarra':>10}{'kappa':>10}")
K5, Pb5, Pe5, *_ = fleiss(SEV, 5)
print(f"{'5 classes (original)':<34}{Pe5:>10.4f}{Pb5:>10.4f}{K5:>10.4f}")
S3 = np.where(SEV <= 2, 1, np.where(SEV == 3, 2, 3))
K3, Pb3, Pe3, *_ = fleiss(S3, 3)
print(f"{'3 classes (1-2 / 3 / 4-5)':<34}{Pe3:>10.4f}{Pb3:>10.4f}{K3:>10.4f}")
S2 = np.where(SEV <= 3, 1, 2)
K2, Pb2, Pe2, *_ = fleiss(S2, 2)
print(f"{'2 classes (1-3 / 4-5)':<34}{Pe2:>10.4f}{Pb2:>10.4f}{K2:>10.4f}")
print("\nColapsar categorias aumenta a concordância observada e também a")
print("esperada; aqui o efeito líquido é de subida. Não se deve escolher o")
print("agrupamento depois de ver qual dá o maior kappa.")

print("\nQuantas unidades e quantos avaliadores? Largura do IC 95% por bootstrap")
print("DAS FOLHAS, com o conjunto de avaliadores fixado (2.000 reamostragens).")
rng = np.random.default_rng(21)
print(f"\n{'avaliadores':<22}{'kappa':>9}{'n = 20':>10}{'n = 40':>10}{'n = 120':>10}")
for jj, rot in [([0,1,2], "A, B, C"), ([0,1,2,3], "A, B, C, D"),
                ([0,1,2,3,4], "A, B, C, D, E"), ([0,1,3,4], "A, B, D, E (sem C)")]:
    linha = []
    for nn in [20, 40, 120]:
        larg = [fleiss(SEV[np.ix_(rng.choice(n, nn, replace=True), jj)], k)[0]
                for _ in range(2000)]
        linha.append(np.percentile(larg, 97.5) - np.percentile(larg, 2.5))
    print(f"{rot:<22}{fleiss(SEV[:, jj], k)[0]:>9.4f}"
          + "".join(f"{v:>10.4f}" for v in linha))
print("\nAs duas vias de ganho rendem quase o mesmo nesta faixa. Com os mesmos")
print("três avaliadores, triplicar as folhas de 40 para 120 estreita o")
print("intervalo em 44% (0,284 para 0,160); com as mesmas 40 folhas, passar de")
print("três para cinco avaliadores estreita 40% (0,284 para 0,171). A escolha")
print("entre as duas é de custo: folhas costumam ser mais baratas que")
print("avaliadores treinados.")
print("O que mais muda o RESULTADO, porém, não é nenhuma das duas: é QUEM")
print("avalia. Trocar o avaliador C por nada - usando A, B, D e E - eleva o")
print("kappa de 0,281 para 0,341 sem coletar um único dado novo. Antes de")
print("aumentar a amostra, vale calibrar quem já está avaliando.")
