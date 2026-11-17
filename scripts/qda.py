# Bibliotecas: numpy, scipy, matplotlib e scikit-learn (todas já no Google Colab)
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

np.set_printoptions(suppress=True)

# ===================================================================
# LDA e QDA, implementadas do zero, para que a diferença fique visível
# ===================================================================
def ajusta(X, g):
    X = np.asarray(X, float); g = np.asarray(g)
    niveis = np.unique(g); k = len(niveis); n, p = X.shape
    medias = np.array([X[g == i].mean(axis=0) for i in niveis])
    prior  = np.array([np.mean(g == i) for i in niveis])
    Sg = [np.cov(X[g == i], rowvar=False) for i in niveis]          # por grupo
    ni = np.array([np.sum(g == i) for i in niveis])
    Sp = sum((ni[j]-1)*Sg[j] for j in range(k)) / (n - k)           # agrupada
    return niveis, medias, np.array(Sg), Sp, prior, ni

def escore_lda(X, medias, Sp, prior):
    Si = np.linalg.inv(Sp)
    return (np.asarray(X, float) @ Si @ medias.T
            - 0.5*np.sum((medias @ Si)*medias, axis=1) + np.log(prior))

def escore_qda(X, medias, Sg, prior):
    """d_i(x) = -0,5 ln|S_i| - 0,5 (x-m_i)' S_i^-1 (x-m_i) + ln(pi_i)"""
    X = np.asarray(X, float); out = np.empty((len(X), len(medias)))
    for j in range(len(medias)):
        Si = np.linalg.inv(Sg[j]); dif = X - medias[j]
        out[:, j] = (-0.5*np.log(np.linalg.det(Sg[j]))
                     - 0.5*np.sum((dif @ Si)*dif, axis=1) + np.log(prior[j]))
    return out

def valida_cruzada(X, g, metodo):
    n = len(X); pred = np.empty(n, int)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        nv, me, Sg, Sp, pr, ni = ajusta(X[m], g[m])
        pred[i] = (escore_qda(X[i:i+1], me, Sg, pr) if metodo == "qda"
                   else escore_lda(X[i:i+1], me, Sp, pr)).argmax()
    return pred

def confusao(g, pred, k):
    C = np.zeros((k, k), int)
    for i in range(len(g)): C[g[i], pred[i]] += 1
    return C

def box_m(X, g):
    """M de Box: testa a igualdade das matrizes de covariância."""
    X = np.asarray(X, float); g = np.asarray(g)
    niveis = np.unique(g); k = len(niveis); n, p = X.shape
    ni = np.array([np.sum(g == i) for i in niveis])
    Sg = [np.cov(X[g == i], rowvar=False) for i in niveis]
    Sp = sum((ni[j]-1)*Sg[j] for j in range(k)) / (n - k)
    M = (n-k)*np.log(np.linalg.det(Sp)) - sum((ni[j]-1)*np.log(np.linalg.det(Sg[j]))
                                              for j in range(k))
    c1 = (sum(1/(ni-1)) - 1/(n-k)) * (2*p*p + 3*p - 1)/(6*(p+1)*(k-1))
    gl = p*(p+1)*(k-1)/2
    return M*(1-c1), gl, stats.chi2.sf(M*(1-c1), gl)

# ===================================================================
# PARTE 1 - iris: a QDA onde a LDA já vai bem
# ===================================================================
d = load_iris(); X = d.data; g = d.target
nomes = ["setosa", "versicolor", "virginica"]
niveis, medias, Sg, Sp, prior, ni = ajusta(X, g)
n, p = X.shape; k = 3

print("iris: 150 flores, 4 medidas, 3 espécies")
print("\nAs três matrizes de covariância NÃO são iguais")
print(f"{'espécie':<12}{'determinante':>16}{'traço':>12}")
for j, nm in enumerate(nomes):
    print(f"{nm:<12}{np.linalg.det(Sg[j]):>16.3e}{np.trace(Sg[j]):>12.4f}")
print(f"{'agrupada':<12}{np.linalg.det(Sp):>16.3e}{np.trace(Sp):>12.4f}")
print(f"A maior determinante é {np.linalg.det(Sg[2])/np.linalg.det(Sg[0]):.1f} "
      f"vezes a menor.")
M, gl, pM = box_m(X, g)
print(f"\nM de Box = {M:.4f} | gl = {gl:.0f} | p = {pM:.3e}")
print("A igualdade das covariâncias é rejeitada, que é o pressuposto da LDA.")
print("Em princípio, portanto, a QDA seria a escolha. Veja o que acontece.")

pl = valida_cruzada(X, g, "lda"); pq = valida_cruzada(X, g, "qda")
Cl = confusao(g, pl, k); Cq = confusao(g, pq, k)
print("\nValidação cruzada deixa-um-de-fora")
print(f"{'real \\ predito':<16}" + "".join(f"{nm:>12}" for nm in nomes) + "   | método")
for j, nm in enumerate(nomes):
    print(f"{nm:<16}" + "".join(f"{c:>12d}" for c in Cl[j]) + "   | LDA")
for j, nm in enumerate(nomes):
    print(f"{nm:<16}" + "".join(f"{c:>12d}" for c in Cq[j]) + "   | QDA")
print(f"LDA: {np.trace(Cl)}/{n} = {100*np.trace(Cl)/n:.2f}% de acerto "
      f"({n-np.trace(Cl)} erros)")
print(f"QDA: {np.trace(Cq)}/{n} = {100*np.trace(Cq)/n:.2f}% de acerto "
      f"({n-np.trace(Cq)} erros)")
dif = np.where(pl != pq)[0]
print(f"As duas regras discordam em {len(dif)} das {n} flores: {dif+1}")
print("\nO M de Box rejeita, mas a QDA não melhora o resultado. A razão é que a")
print("QDA estima k matrizes de covariância em vez de uma: em iris são 3 x 10 =")
print("30 parâmetros de covariância, contra 10 da LDA, com os mesmos 150 dados.")
print("A variância extra das estimativas consome o ganho de flexibilidade.")

# ===================================================================
# PARTE 2 - um caso em que a QDA é claramente superior
# ===================================================================
print("\n" + "=" * 70)
print("Dados fictícios em que a diferença aparece: dois grupos com a MESMA")
print("média e dispersões muito diferentes (um compacto dentro de um disperso).")
rng = np.random.default_rng(7)
nA = nB = 200
A = rng.multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 1.0]], nA)       # compacto
B = rng.multivariate_normal([0, 0], [[9.0, 0.0], [0.0, 9.0]], nB)       # disperso
Y = np.vstack([A, B]); gy = np.r_[np.zeros(nA), np.ones(nB)].astype(int)
nv2, me2, Sg2, Sp2, pr2, ni2 = ajusta(Y, gy)
print(f"\nMédias: grupo A {np.round(me2[0],3)} | grupo B {np.round(me2[1],3)}")
print(f"Determinantes: A {np.linalg.det(Sg2[0]):.4f} | B {np.linalg.det(Sg2[1]):.4f}")
M2, gl2, pM2 = box_m(Y, gy)
print(f"M de Box = {M2:.2f} | gl = {gl2:.0f} | p = {pM2:.3e}")
pl2 = valida_cruzada(Y, gy, "lda"); pq2 = valida_cruzada(Y, gy, "qda")
Cl2 = confusao(gy, pl2, 2); Cq2 = confusao(gy, pq2, 2)
print(f"\n{'real \\ predito':<16}{'A':>8}{'B':>8}   | método")
for j, nm in enumerate(["A", "B"]):
    print(f"{nm:<16}{Cl2[j,0]:>8d}{Cl2[j,1]:>8d}   | LDA")
for j, nm in enumerate(["A", "B"]):
    print(f"{nm:<16}{Cq2[j,0]:>8d}{Cq2[j,1]:>8d}   | QDA")
print(f"LDA: {100*np.trace(Cl2)/len(gy):.2f}% de acerto")
print(f"QDA: {100*np.trace(Cq2)/len(gy):.2f}% de acerto")
print("As duas médias POPULACIONAIS são iguais (as amostrais diferem apenas por")
print("flutuação), de modo que nenhuma fronteira LINEAR consegue separar os")
print("grupos: a LDA fica perto do acaso, com 57,5% onde 50% é o chute. A QDA")
print("desenha uma fronteira CIRCULAR em torno da origem, atribuindo ao grupo")
print("compacto o que está perto do centro, e chega a 84,3%.")

# ===================================================================
# PARTE 3 - a troca entre viés e variância, em função de n
# ===================================================================
print("\n" + "=" * 70)
print("Quando a QDA compensa? A resposta depende de n e do QUANTO as")
print("covariâncias diferem. Dois grupos normais em p = 5 dimensões, médias")
print("separadas por 0,7 em cada eixo, covariâncias I e c x I.")
print("Erro medido em 3.000 observações novas, média de 300 repetições.")
rng = np.random.default_rng(5)
for c, rotulo in [(1.4, "diferença MODERADA (c = 1,4)"),
                  (3.0, "diferença GRANDE (c = 3,0)")]:
    print(f"\n{rotulo}")
    print(f"{'n por grupo':>12}{'erro LDA':>12}{'erro QDA':>12}{'vencedor':>12}")
    for nn in [12, 20, 40, 70, 100, 400]:
        mu = np.zeros(5); mu2 = mu + 0.7
        S1 = np.eye(5); S2 = c*np.eye(5)
        el = []; eq = []
        for _ in range(300):
            Z = np.vstack([rng.multivariate_normal(mu, S1, nn),
                           rng.multivariate_normal(mu2, S2, nn)])
            gz = np.r_[np.zeros(nn), np.ones(nn)].astype(int)
            T = np.vstack([rng.multivariate_normal(mu, S1, 1500),
                           rng.multivariate_normal(mu2, S2, 1500)])
            gt = np.r_[np.zeros(1500), np.ones(1500)].astype(int)
            nv3, me3, Sg3, Sp3, pr3, ni3 = ajusta(Z, gz)
            el.append(np.mean(escore_lda(T, me3, Sp3, pr3).argmax(axis=1) != gt))
            eq.append(np.mean(escore_qda(T, me3, Sg3, pr3).argmax(axis=1) != gt))
        a, b = 100*np.mean(el), 100*np.mean(eq)
        print(f"{nn:>12}{a:>11.2f}%{b:>11.2f}%{('QDA' if b < a else 'LDA'):>12}")
print("\nCom diferença moderada entre as covariâncias, a LDA VENCE enquanto n é")
print("pequeno, mesmo sabendo-se que o modelo dela é falso: estimar duas")
print("matrizes de covariância em vez de uma custa variância, e esse custo supera")
print("o viés do modelo errado. O ponto de virada fica entre n = 70 e n = 100 por")
print("grupo. Com diferença grande, o viés da LDA é alto demais desde o início e")
print("a QDA vence em toda a faixa.")
print("\nParâmetros de covariância a estimar:")
for pp in [2, 5, 10, 20]:
    print(f"  p = {pp:>2}: LDA {pp*(pp+1)//2:>4}  |  QDA com k = 2: "
          f"{2*pp*(pp+1)//2:>4}  |  QDA com k = 3: {3*pp*(pp+1)//2:>4}")
print("O número cresce com p², e é por isso que o ponto de virada sobe depressa")
print("com a dimensão. A análise discriminante REGULARIZADA interpola entre as")
print("duas, encolhendo cada covariância de grupo em direção à agrupada.")
