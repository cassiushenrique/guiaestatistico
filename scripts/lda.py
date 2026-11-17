# Bibliotecas: numpy, scipy, matplotlib e scikit-learn (todas já no Google Colab)
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

np.set_printoptions(suppress=True)

# ===================================================================
# A análise discriminante linear, implementada do zero
# ===================================================================
def lda_ajusta(X, g):
    """Devolve médias por grupo, covariância agrupada, priores e autovetores."""
    X = np.asarray(X, float); g = np.asarray(g)
    niveis = np.unique(g); k = len(niveis); n, p = X.shape
    medias = np.array([X[g == i].mean(axis=0) for i in niveis])
    prior  = np.array([np.mean(g == i) for i in niveis])
    S = np.zeros((p, p))
    for j, i in enumerate(niveis):
        Xi = X[g == i]
        S += (Xi - medias[j]).T @ (Xi - medias[j])
    S /= (n - k)                                  # covariância AGRUPADA
    return niveis, medias, S, prior

def lda_escores(X, medias, S, prior):
    """Função discriminante linear de cada grupo, avaliada em cada observação."""
    Si = np.linalg.inv(S)
    # d_i(x) = x' S^-1 m_i - 0,5 m_i' S^-1 m_i + ln(pi_i)
    return (np.asarray(X, float) @ Si @ medias.T
            - 0.5 * np.sum((medias @ Si) * medias, axis=1) + np.log(prior))

def lda_eixos(X, g):
    """Eixos canônicos: autovetores de W^-1 B, normalizados em W."""
    X = np.asarray(X, float); g = np.asarray(g)
    niveis = np.unique(g); k = len(niveis); n, p = X.shape
    gm = X.mean(axis=0); W = np.zeros((p, p)); B = np.zeros((p, p))
    for i in niveis:
        Xi = X[g == i]; mi = Xi.mean(axis=0)
        W += (Xi - mi).T @ (Xi - mi)
        B += len(Xi) * np.outer(mi - gm, mi - gm)
    val, vec = np.linalg.eig(np.linalg.inv(W) @ B)
    ordem = np.argsort(val.real)[::-1]
    val = val.real[ordem]; vec = vec.real[:, ordem]
    # normalização da MASS::lda(): variância agrupada unitária nos escores
    for j in range(vec.shape[1]):
        esc = (X - gm) @ vec[:, j]
        sw = np.sqrt(sum(np.var(esc[g == i], ddof=1) * (np.sum(g == i) - 1)
                         for i in niveis) / (n - k))
        vec[:, j] = vec[:, j] / sw
        if vec[0, j] < 0: vec[:, j] = -vec[:, j]   # o sinal é arbitrário
    return val[:min(k - 1, p)], vec[:, :min(k - 1, p)], gm

# ===================================================================
# PARTE 1 - iris: 150 flores, quatro medidas, três espécies
# ===================================================================
d = load_iris(); X = d.data; g = d.target
nomes = ["setosa", "versicolor", "virginica"]
vars_ = ["compr. sépala", "larg. sépala", "compr. pétala", "larg. pétala"]
niveis, medias, S, prior = lda_ajusta(X, g)
n, p = X.shape; k = len(niveis)

print("iris: 150 flores, 4 medidas, 3 espécies (50 de cada)")
print("\nMédias por grupo (cm)")
print(f"{'espécie':<12}" + "".join(f"{v:>16}" for v in vars_))
for j, nm in enumerate(nomes):
    print(f"{nm:<12}" + "".join(f"{m:>16.3f}" for m in medias[j]))

print("\nMatriz de covariância AGRUPADA (gl = 150 - 3 = 147)")
print(np.round(S, 4))

# --- os eixos canônicos
val, vec, gm = lda_eixos(X, g)
prop = val / val.sum()
print(f"\nAutovalores de W^-1 B: {np.round(val, 4)}")
print(f"Proporção do poder discriminante: {np.round(100*prop, 2)} %")
print("\nCoeficientes dos discriminantes lineares (escala de MASS::lda)")
print(f"{'variável':<16}{'LD1':>10}{'LD2':>10}")
for j, v in enumerate(vars_):
    print(f"{v:<16}{vec[j,0]:>10.4f}{vec[j,1]:>10.4f}")
print("O sinal de cada coluna é arbitrário; aqui foi fixado para que o primeiro")
print("coeficiente seja positivo, o que reproduz a saída de MASS::lda() em iris.")
print("A soma dos dois autovalores reproduz o traço de Hotelling-Lawley da")
print(f"ficha da MANOVA: {val.sum():.4f}")

# --- classificação por reclassificação (otimista)
esc = lda_escores(X, medias, S, prior)
pred = esc.argmax(axis=1)
conf = np.zeros((k, k), int)
for i in range(n): conf[g[i], pred[i]] += 1
print("\nMatriz de confusão por RECLASSIFICAÇÃO (otimista)")
print(f"{'real \\ predito':<16}" + "".join(f"{nm:>12}" for nm in nomes))
for j, nm in enumerate(nomes):
    print(f"{nm:<16}" + "".join(f"{c:>12d}" for c in conf[j]))
acerto = np.trace(conf) / n
print(f"Acerto aparente = {np.trace(conf)}/{n} = {100*acerto:.2f}%  "
      f"(erro {100*(1-acerto):.2f}%)")

# --- validação cruzada deixa-um-de-fora
pred_vc = np.empty(n, int)
for i in range(n):
    m = np.ones(n, bool); m[i] = False
    nv, me, Sv, pr = lda_ajusta(X[m], g[m])
    pred_vc[i] = lda_escores(X[i:i+1], me, Sv, pr).argmax()
conf_vc = np.zeros((k, k), int)
for i in range(n): conf_vc[g[i], pred_vc[i]] += 1
print("\nMatriz de confusão por VALIDAÇÃO CRUZADA deixa-um-de-fora")
print(f"{'real \\ predito':<16}" + "".join(f"{nm:>12}" for nm in nomes))
for j, nm in enumerate(nomes):
    print(f"{nm:<16}" + "".join(f"{c:>12d}" for c in conf_vc[j]))
ac_vc = np.trace(conf_vc) / n
print(f"Acerto por validação cruzada = {np.trace(conf_vc)}/{n} = {100*ac_vc:.2f}%  "
      f"(erro {100*(1-ac_vc):.2f}%)")
print(f"A diferença entre os dois é de {100*(acerto-ac_vc):.2f} ponto percentual.")
print("Aqui os dois coincidem, mas isso não é a regra: a reclassificação usa cada")
print("flor para estimar o classificador que depois a julga, e por isso é otimista.")
print("Com n = 150 e p = 4, e grupos tão separados, o viés é desprezível. A")
print("simulação do fim desta ficha mostra o que acontece quando não é.")

# --- as flores mal classificadas
erradas = np.where(pred_vc != g)[0]
print(f"\nAs {len(erradas)} flores mal classificadas (validação cruzada)")
print(f"{'obs.':>5}{'real':>12}{'predita':>12}" + "".join(f"{v:>15}" for v in vars_))
for i in erradas:
    print(f"{i+1:>5}{nomes[g[i]]:>12}{nomes[pred_vc[i]]:>12}"
          + "".join(f"{x:>15.1f}" for x in X[i]))

# --- um caso resolvido à mão
i0 = int(erradas[0])
print(f"\nO cálculo para a flor {i0+1}, passo a passo")
print(f"  medidas: {X[i0]}")
Si = np.linalg.inv(S)
for j, nm in enumerate(nomes):
    lin = X[i0] @ Si @ medias[j]
    quad = -0.5 * medias[j] @ Si @ medias[j]
    print(f"  d({nm:<10}) = {lin:9.4f} {quad:9.4f} + ln({prior[j]:.4f}) "
          f"= {lin+quad+np.log(prior[j]):9.4f}")
print(f"  O maior é o do grupo '{nomes[esc[i0].argmax()]}', e é essa a predição.")
pos = np.exp(esc[i0] - esc[i0].max()); pos = pos / pos.sum()
print(f"  Probabilidades a posteriori: "
      + "  ".join(f"{nm} {100*q:.2f}%" for nm, q in zip(nomes, pos)))

# --- o que as probabilidades a posteriori mostram
POS = np.exp(esc - esc.max(axis=1, keepdims=True))
POS = POS / POS.sum(axis=1, keepdims=True)
maxpos = POS.max(axis=1)
print(f"\nProbabilidade a posteriori do grupo escolhido")
print(f"  mediana = {np.median(maxpos):.4f} | flores com p < 0,90: "
      f"{int(np.sum(maxpos < 0.90))} de {n}")
print("  São as flores da fronteira entre versicolor e virginica.")

# ===================================================================
# PARTE 2 - a suposição de covariâncias iguais, e quando ela falha
# ===================================================================
print("\n" + "=" * 70)
print("A LDA supõe a MESMA matriz de covariância em todos os grupos.")
print("Covariâncias por grupo em iris (variâncias na diagonal)")
for j, nm in enumerate(nomes):
    Xi = X[g == j]; Sj = np.cov(Xi, rowvar=False)
    print(f"  {nm:<12} determinante = {np.linalg.det(Sj):.3e}  "
          f"traço = {np.trace(Sj):.4f}")
print(f"  {'agrupada':<12} determinante = {np.linalg.det(S):.3e}  "
      f"traço = {np.trace(S):.4f}")
print("A ficha do M de Box mostra que essa igualdade é REJEITADA em iris")
print("(qui² = 140,943). Ainda assim a LDA vai bem aqui, porque as médias")
print("estão muito separadas; a ficha da QDA mostra o caso em que não vai.")

# --- a fronteira é linear: demonstração
print("\nPor que a fronteira é um hiperplano")
dif = medias[1] - medias[2]
w = Si @ dif
c = 0.5 * (medias[1] + medias[2]) @ w + np.log(prior[2]/prior[1])
print("  d_versicolor(x) - d_virginica(x) = w'x - c, com")
print(f"  w = {np.round(w, 4)}")
print(f"  c = {c:.4f}")
print("  Como w não depende de x, a fronteira {x: w'x = c} é um hiperplano.")
print("  É a subtração dos termos quadráticos, iguais nos dois grupos, que o")
print("  torna linear; na QDA eles não se cancelam.")

# ===================================================================
# PARTE 3 - o efeito das probabilidades a priori
# ===================================================================
print("\n" + "=" * 70)
print("As probabilidades a priori deslocam a fronteira.")
print("Suponha que, na população de interesse, 80% sejam virginica e as")
print("outras duas espécies tenham 10% cada.")
pr2 = np.array([0.10, 0.10, 0.80])
esc2 = lda_escores(X, medias, S, pr2)
pred2 = esc2.argmax(axis=1)
conf2 = np.zeros((k, k), int)
for i in range(n): conf2[g[i], pred2[i]] += 1
print(f"{'real \\ predito':<16}" + "".join(f"{nm:>12}" for nm in nomes))
for j, nm in enumerate(nomes):
    print(f"{nm:<16}" + "".join(f"{c:>12d}" for c in conf2[j]))
print(f"Acerto aparente = {100*np.trace(conf2)/n:.2f}%  "
      f"(era {100*acerto:.2f}% com priores iguais)")
mud = int(np.sum(pred2 != pred))
print(f"{mud} flores mudaram de classificação, todas de versicolor para virginica.")
print("O acerto global CAI porque a amostra tem 1/3 de cada espécie, e não a")
print("composição suposta. Em uma população realmente 80% virginica, a regra")
print("com priores corretos é a que minimiza o erro esperado.")

# ===================================================================
# PARTE 4 - por que a reclassificação é otimista
# ===================================================================
print("\n" + "=" * 70)
print("O viés da reclassificação, medido em dados SEM nenhum sinal.")
print("Dois grupos de n/2 observações cada, com p variáveis geradas da MESMA")
print("distribuição normal padrão: por construção, nenhum classificador pode")
print("acertar mais que 50% em dados novos.")
rng = np.random.default_rng(11)
print(f"\n{'n':>6}{'p':>6}{'acerto aparente':>20}{'acerto val. cruzada':>22}")
for nn, pp in [(40, 2), (40, 10), (40, 18), (100, 10), (400, 10)]:
    ap = []; vc = []
    for _ in range(200):
        Z = rng.standard_normal((nn, pp)); gz = np.r_[np.zeros(nn//2), np.ones(nn//2)].astype(int)
        nv, me, Sz, pr = lda_ajusta(Z, gz)
        ap.append(np.mean(lda_escores(Z, me, Sz, pr).argmax(axis=1) == gz))
        pv = np.empty(nn, int)
        for i in range(nn):
            m = np.ones(nn, bool); m[i] = False
            _, me2, S2, pr2_ = lda_ajusta(Z[m], gz[m])
            pv[i] = lda_escores(Z[i:i+1], me2, S2, pr2_).argmax()
        vc.append(np.mean(pv == gz))
    print(f"{nn:>6}{pp:>6}{100*np.mean(ap):>19.2f}%{100*np.mean(vc):>21.2f}%")
print("\nCom p grande em relação a n, a reclassificação chega a 81% de acerto em")
print("dados que são puro ruído, enquanto a validação cruzada fica em torno de")
print("50%, que é a verdade. Reportar o acerto aparente como desempenho do")
print("classificador é o erro mais comum nesta análise.")
