# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados fictícios: 18 analistas executaram a MESMA tarefa em quatro versões de
# uma interface, e registrou-se se concluíram (1) ou não (0) dentro do tempo.
# Cada analista passou pelas quatro versões: é um delineamento em blocos com
# resposta binária.
dados = np.array([
 [1,1,1,0],[1,1,1,1],[0,1,1,0],[1,1,1,1],[0,0,1,0],[1,1,1,1],
 [0,1,1,0],[1,1,1,1],[0,0,1,0],[1,1,1,1],[0,1,1,1],[1,1,1,1],
 [0,0,1,0],[1,1,1,0],[0,1,1,1],[1,1,1,1],[0,0,0,0],[0,1,1,1]])
versoes = ["versão A", "versão B", "versão C", "versão D"]
n, k = dados.shape
print(f"{n} analistas x {k} versões da interface")
print("versão      sucessos   taxa")
for j, v in enumerate(versoes):
    print(f"{v:<12}{int(dados[:,j].sum()):8d}   {100*dados[:,j].mean():5.1f}%")
Li = dados.sum(axis=1)
print(f"\nSucessos por analista: {Li}")
print(f"Analistas com 0 ou {k} sucessos (não contribuem): "
      f"{int(np.sum((Li==0)|(Li==k)))} de {n}")

# 1. A estatística Q de Cochran
Gj = dados.sum(axis=0)                       # sucessos por tratamento
N = dados.sum()
Q = (k - 1) * (k * np.sum(Gj**2) - N**2) / (k * N - np.sum(Li**2))
p = stats.chi2.sf(Q, k - 1)
print(f"\nG_j (sucessos por versão) = {Gj}   soma = {N}")
print(f"soma de G_j² = {int(np.sum(Gj**2))} | soma de L_i² = {int(np.sum(Li**2))}")
print(f"\nQ = {k-1} x ({k} x {int(np.sum(Gj**2))} - {int(N)}²) / ({k} x {int(N)} - {int(np.sum(Li**2))})")
print(f"Q = {Q:.4f} | gl = {k-1} | p-valor = {p:.3e}")

# 2. Só os blocos com resposta mista contribuem
mistos = dados[(Li > 0) & (Li < k)]
Gm = mistos.sum(axis=0); Lm = mistos.sum(axis=1); Nm = mistos.sum(); nm = len(mistos)
Qm = (k-1)*(k*np.sum(Gm**2) - Nm**2)/(k*Nm - np.sum(Lm**2))
print(f"\nRefazendo a conta apenas com os {nm} analistas de resposta mista:")
print(f"Q = {Qm:.4f}, idêntico. Blocos todos-sucesso ou todos-falha não informam.")

# 3. Com k = 2, Q é o McNemar sem correção
print("\nVerificação: com apenas duas versões (A e B), Q deve ser o McNemar sem correção.")
d2 = dados[:, :2]
L2 = d2.sum(axis=1); G2 = d2.sum(axis=0); N2 = d2.sum()
Q2 = 1 * (2*np.sum(G2**2) - N2**2)/(2*N2 - np.sum(L2**2))
b = int(np.sum((d2[:,0]==1) & (d2[:,1]==0))); c = int(np.sum((d2[:,0]==0) & (d2[:,1]==1)))
print(f"  Q = {Q2:.4f} | McNemar sem correção = (b-c)²/(b+c) = ({b}-{c})²/{b+c} "
      f"= {(b-c)**2/(b+c):.4f}")

# 4. Comparações múltiplas: McNemar par a par com ajuste
print(f"\nComparações pareadas (McNemar exato, ajuste de Bonferroni para {k*(k-1)//2} testes)")
pares = list(itertools.combinations(range(k), 2))
brutos = []
for i, j in pares:
    bi = int(np.sum((dados[:,i]==1) & (dados[:,j]==0)))
    ci = int(np.sum((dados[:,i]==0) & (dados[:,j]==1)))
    pv = 1.0 if bi+ci == 0 else min(1.0, 2*stats.binom.cdf(min(bi,ci), bi+ci, 0.5))
    brutos.append((f"{versoes[j]} - {versoes[i]}", bi, ci, pv))
m = len(pares)
ordem = sorted(range(m), key=lambda t: brutos[t][3])
holm = np.empty(m); ac = 0.0
for pos, idx in enumerate(ordem):
    ac = max(ac, (m - pos)*brutos[idx][3]); holm[idx] = min(1.0, ac)
print(f"{'par':<26}{'b':>4}{'c':>4}{'p bruto':>11}{'Bonferroni':>12}{'Holm':>10}")
for idx in ordem:
    nm_, bi, ci, pv = brutos[idx]
    print(f"{nm_:<26}{bi:4d}{ci:4d}{pv:11.5f}{min(1,m*pv):12.5f}"
          f"{'*' if min(1,m*pv)<0.05 else ' '}{holm[idx]:9.5f}{'*' if holm[idx]<0.05 else ' '}")

# 5. Tamanho de efeito
print(f"\nW de Kendall (concordância entre analistas) = {Q/(n*(k-1)):.4f}")
print(f"Amplitude das taxas de sucesso = {100*(Gj.max()-Gj.min())/n:.1f} pontos percentuais")

# 6. Versão exata por permutação, para amostras pequenas
print("\nValor-p exato por permutação (os rótulos das versões são embaralhados")
print("DENTRO de cada analista, que é o que a hipótese nula autoriza):")
rng = np.random.default_rng(2026)
B = 100000
cont = 0
for _ in range(B):
    perm = np.array([linha[rng.permutation(k)] for linha in dados])
    Gp = perm.sum(axis=0); Lp = perm.sum(axis=1); Np = perm.sum()
    den = k*Np - np.sum(Lp**2)
    if den == 0: continue
    Qp = (k-1)*(k*np.sum(Gp**2) - Np**2)/den
    if Qp >= Q - 1e-9: cont += 1
print(f"  p por permutação ({B} reamostragens) = {(1+cont)/(B+1):.6f}")
print(f"  p pela aproximação qui-quadrado      = {p:.6f}")

# 7. O Q de Cochran é o teste de Friedman aplicado a dados binários
Qf, pf = stats.friedmanchisquare(*[dados[:, j] for j in range(k)])
print(f"\nTeste de Friedman sobre os mesmos dados: Q = {Qf:.4f}, p = {pf:.3e}")
print("Os dois coincidem EXATAMENTE, e isso não é coincidência: com resposta")
print("binária, a estatística de Friedman calculada sobre os postos dentro de cada")
print("bloco reduz-se algebricamente ao Q de Cochran. O Q é a forma direta de")
print("escrever o mesmo teste, dispensando o cálculo dos postos.")
