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

from collections import Counter

def postos(v):
    """Postos médios, com empates resolvidos pela média."""
    return stats.rankdata(v)

def kendall_w(X, corrigir=True):
    """W de Kendall. X: n unidades (linhas) x m avaliadores (colunas)."""
    X = np.asarray(X, float); n, m = X.shape
    R = np.array([postos(X[:, j]) for j in range(m)])      # m x n
    Rt = R.sum(axis=0)                                     # soma por unidade
    S = ((Rt - Rt.mean())**2).sum()
    if corrigir:
        T = sum(sum(t**3 - t for t in Counter(R[j]).values()) for j in range(m))
    else:
        T = 0.0
    W = 12*S/(m**2*(n**3 - n) - m*T)
    return W, S, T, R, Rt

n, m = SEV.shape
print(f"{n} folhas, {m} avaliadores, severidade ordinal de 1 a 5")
print("O W de Kendall pergunta se os avaliadores ORDENAM as folhas da mesma")
print("maneira, e não se atribuem a mesma classe: é a pergunta certa quando o")
print("que interessa é a hierarquia, e não o rótulo.")

W, S, T, R, Rt = kendall_w(SEV)
print(f"\nSoma dos postos por folha (Rt), primeiras doze folhas")
print(f"{'folha':>6}{'classificações':>22}{'postos dos 5 avaliadores':>36}{'Rt':>9}")
for i in range(12):
    print(f"{i+1:>6}{str(SEV[i].tolist()):>22}"
          f"{'  '.join(f'{R[j,i]:5.1f}' for j in range(m)):>36}{Rt[i]:>9.1f}")
print("...")
print(f"\nMédia de Rt = {Rt.mean():.2f} (é sempre m(n+1)/2 = "
      f"{m*(n+1)/2:.2f})")
print(f"S = soma dos desvios quadráticos de Rt = {S:.2f}")
print(f"Correção para empates: T = {T:.0f}")
print(f"\nW = 12 S / [m²(n³ − n) − m T]")
print(f"W = 12 x {S:.2f} / [{m}² x ({n}³ − {n}) − {m} x {T:.0f}] = {W:.4f}")
Wsc = kendall_w(SEV, corrigir=False)[0]
print(f"Sem a correção para empates, W = {Wsc:.4f} - a correção importa aqui,")
print(f"porque com só cinco classes e {n} folhas há muitos empates.")

# ===================================================================
# PARTE 1 - as três leituras equivalentes de W
# ===================================================================
print("\n" + "=" * 70)
print("W tem três leituras, todas iguais em valor")
print("\n(a) Proporção da variação máxima possível de Rt")
Smax = (m**2*(n**3-n) - m*T)/12
print(f"    S observado = {S:.2f} | S máximo possível = {Smax:.2f}")
print(f"    razão = {S/Smax:.4f} = W")
print("    W = 1 significa concordância perfeita: todos ordenam igual.")
print("    W = 0 exigiria que as somas de postos fossem todas iguais, o que é")
print("    um arranjo MUITO improvável, e não o resultado do acaso: sob")
print("    independência W vale 1/m em média, como a parte 3 mostra.")

print("\n(b) Função da correlação de Spearman média entre os pares")
rs = []
for i in range(m):
    for j in range(i+1, m):
        rs.append(stats.spearmanr(SEV[:,i], SEV[:,j]).statistic)
rs = np.array(rs)
print(f"    {len(rs)} correlações de Spearman pareadas:")
print("    " + "  ".join(f"{v:.3f}" for v in rs))
rbar = rs.mean()
print(f"    média = {rbar:.4f}")
print(f"    W previsto = [(m − 1)·rbar + 1]/m = [{m-1}x{rbar:.4f} + 1]/{m} = "
      f"{((m-1)*rbar+1)/m:.4f}")
print(f"    W calculado = {W:.4f}")
print("    A relação r = (mW − 1)/(m − 1) é uma identidade algébrica, e vale")
print("    também com empates, desde que W use a correção e as correlações")
print("    sejam de Spearman com postos médios.")
print(f"    A identidade é exata até a quarta casa. Note o que ela implica:")
print(f"    com m = {m}, mesmo rbar = 0 daria W = 1/m = {1/m:.4f}.")
print("    W NUNCA é negativo - e por isso 'W = 0,2' não quer dizer o mesmo")
print("    que 'correlação de 0,2'.")

print("\n(c) Relação exata com o teste de Friedman")
chi2_fried = m*(n-1)*W
p_fried = stats.chi2.sf(chi2_fried, n-1)
print(f"    qui² de Friedman = m(n − 1)W = {m} x {n-1} x {W:.4f} = "
      f"{chi2_fried:.4f}")
print(f"    gl = n − 1 = {n-1} | p = {p_fried:.4e}")
amostra = SEV.T
chi2_direto = stats.friedmanchisquare(*[amostra[:, i] for i in range(n)]).statistic
p_direto = stats.friedmanchisquare(*[amostra[:, i] for i in range(n)]).pvalue
print(f"    scipy.stats.friedmanchisquare devolve {chi2_direto:.4f} "
      f"com p = {p_direto:.4e}")
print("    São o MESMO teste: W é a estatística de Friedman reescalonada para")
print("    o intervalo [0, 1]. Friedman dá o valor-p; W dá o tamanho do efeito.")
print("    ATENÇÃO aos papéis: no Friedman da ficha correspondente, os")
print("    tratamentos são as colunas e os blocos as linhas. Aqui é o")
print("    contrário - as FOLHAS fazem o papel de 'tratamentos' e os")
print("    AVALIADORES o de blocos, porque a pergunta é se as folhas são")
print("    ordenadas de modo consistente.")

# ===================================================================
# PARTE 2 - o que W não diz
# ===================================================================
print("\n" + "=" * 70)
print("W mede concordância de ORDEM, não de VALOR.")
print("\nDemonstração: somando 1 classe a tudo o que o avaliador C fez")
S2 = SEV.copy(); S2[:,2] = np.clip(S2[:,2] + 1, 1, 6)
W2 = kendall_w(S2)[0]
print(f"    W original = {W:.4f} | W com +1 em C = {W2:.4f}")
print("    Quase não muda: somar uma constante não altera a ordenação.")
print("    Se o que interessa é o VALOR atribuído, e não a ordem, W é a")
print("    ferramenta errada - use o ICC ou o kappa ponderado.")

print("\nDemonstração oposta: INVERTENDO a ordenação do avaliador C")
S3 = SEV.copy(); S3[:,2] = 6 - S3[:,2]
W3 = kendall_w(S3)[0]
print(f"    W com C invertido = {W3:.4f}")
print(f"    A queda é grande ({W:.4f} para {W3:.4f}), mas W continua bem acima")
print("    de zero, porque os outros quatro avaliadores seguem concordando")
print("    entre si. W é um índice GLOBAL: ele não aponta o dissidente.")

print("\nPara localizar o dissidente, retire um avaliador de cada vez:")
print(f"{'sem o avaliador':>18}{'W':>10}{'variação':>12}")
for j in range(m):
    resto = [c for c in range(m) if c != j]
    Wj = kendall_w(SEV[:, resto])[0]
    print(f"{AVAL[j]:>18}{Wj:>10.4f}{Wj-W:>+12.4f}")
print("Tirar E é o que mais eleva W: ele é o avaliador cuja ORDENAÇÃO mais")
print("destoa. Compare com a ficha do kappa de Fleiss, em que o avaliador C")
print("era o mais problemático - lá a pergunta era sobre o VALOR da classe,")
print("e C tem viés de severidade sem errar a ordem.")

# ===================================================================
# PARTE 3 - distribuição de W sob independência
# ===================================================================
print("\n" + "=" * 70)
print("O que W vale quando não há concordância nenhuma?")
print("Simulação: 20.000 conjuntos em que cada avaliador ordena as 40 folhas")
print("ao acaso, usando as mesmas marginais dos avaliadores reais.")
rng = np.random.default_rng(13)
marg = [np.bincount(SEV[:,j], minlength=6)[1:]/n for j in range(m)]
sim = []
for _ in range(20000):
    Z = np.column_stack([rng.choice(5, n, p=marg[j])+1 for j in range(m)])
    sim.append(kendall_w(Z)[0])
sim = np.array(sim)
print(f"\n  média = {sim.mean():.4f}  (o valor esperado é 1/m = {1/m:.4f})")
print(f"  desvio-padrão = {sim.std(ddof=1):.4f}")
print(f"  percentil 95 = {np.percentile(sim, 95):.4f}")
print(f"  W observado = {W:.4f}, muito acima de qualquer valor simulado")
print(f"  (maior W simulado = {sim.max():.4f})")
print("\nO valor esperado de W sob independência é 1/m, e não 0: com cinco")
print(f"avaliadores, {1/m:.2f}. Um W de 0,25 com cinco avaliadores não é")
print("concordância fraca - é concordância NENHUMA. Esse é o erro de leitura")
print("mais comum do coeficiente.")
print(f"\nValor esperado de W sob independência, por número de avaliadores:")
print("  " + "  ".join(f"m = {mm}: {1/mm:.3f}" for mm in [2, 3, 5, 10, 20]))

# ===================================================================
# PARTE 4 - W com poucas unidades e o teste exato
# ===================================================================
print("\n" + "=" * 70)
print("Com n pequeno, a aproximação qui-quadrado falha e é preciso permutar.")
print("Permutação: embaralham-se os postos DENTRO de cada avaliador, que é o")
print("que a hipótese nula autoriza.")
def p_permuta(X, B=20000, semente=4):
    Wobs = kendall_w(X)[0]; nn, mm = X.shape
    rng = np.random.default_rng(semente); c = 0
    for _ in range(B):
        Z = np.column_stack([rng.permutation(X[:, j]) for j in range(mm)])
        if kendall_w(Z)[0] >= Wobs: c += 1
    return (c+1)/(B+1)
print(f"\n{'subconjunto':<22}{'n':>4}{'W':>9}{'p qui-quadrado':>18}"
      f"{'p permutação':>16}")
for rot, idx in [("todas as folhas", np.arange(40)),
                 ("12 primeiras", np.arange(12)),
                 ("6 primeiras", np.arange(6))]:
    Xs = SEV[idx]; Ws = kendall_w(Xs)[0]; ns = len(idx)
    pq = stats.chi2.sf(m*(ns-1)*Ws, ns-1)
    print(f"{rot:<22}{ns:>4}{Ws:>9.4f}{pq:>18.3e}{p_permuta(Xs):>16.5f}")
print("\nO valor-p por permutação não pode descer abaixo de 1/(B + 1) = 5e-05,")
print("que é o piso de 20.000 reamostragens: com 40 e com 12 folhas, os dois")
print("métodos apenas concordam em que a evidência é esmagadora. Com 6 folhas,")
print("porém, a aproximação qui-quadrado devolve 2,2e-03 e a permutação 2,5e-04,")
print("quase dez vezes menor - a aproximação é grosseira ali. A recomendação")
print("clássica é usá-la apenas com n maior que 7; abaixo disso, tabelas exatas")
print("ou permutação.")
