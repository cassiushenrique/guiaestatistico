# Bibliotecas: numpy e scipy (ambas já instaladas no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# Dados fictícios, criados pelo autor, compartilhados pelas fichas de
# avaliação de modelos: 120 talhões de milho, com a produtividade (t/ha)
# e quatro variáveis: dose de nitrogênio (kg/ha), densidade de plantio
# (mil plantas/ha), matéria orgânica do solo (%) e precipitação no ciclo (mm).
# Colunas: y, N, densidade, materia_organica, chuva
# ===================================================================
MILHO = np.array([
 [7.65,40,56.3,1.86,572], [9.35,200,65.2,3.67,736], [6.8,160,71.1,2.46,372], [7.52,80,62.6,3.54,340],
 [6.39,0,64,3.17,528], [8.54,160,58.2,2.81,433], [9.29,80,60.3,3.48,530], [6.85,120,59.8,2.58,375],
 [8.76,120,60,3.54,532], [8,40,57.8,1.87,638], [5.86,80,64.8,2.27,620], [9.23,200,71.1,3.65,421],
 [7.37,160,55.4,1.32,492], [8.55,200,71.8,3.13,600], [5.73,0,59.2,2.51,684], [6.3,80,59.4,2.41,434],
 [5.66,0,46.9,2.31,443], [10.2,200,45,3.7,621], [8.84,120,67.1,1.72,488], [7.22,40,54.4,3.38,428],
 [8.62,160,67.6,3.13,615], [8.97,160,62.4,2.5,369], [5.58,40,63.8,2.74,458], [8.52,160,71.3,3.29,607],
 [6.97,0,61.5,2.38,463], [7.8,80,39.1,1.73,589], [8.8,120,49.8,3,410], [6.51,0,68.8,2.21,590],
 [8.1,160,76,2.51,562], [10.13,200,61.3,2.84,666], [8.7,160,57.1,3.32,498], [9.12,120,52.3,4.07,488],
 [9.21,200,65.8,3.04,511], [9.56,80,49.4,2.88,614], [7.59,80,53.3,3.37,453], [9,80,63,3,507],
 [7.84,80,62.7,1.62,695], [7.23,120,60.2,2.38,590], [5.79,80,69.4,1.93,585], [7.6,80,53.3,3.29,473],
 [10.46,120,53.7,3.69,597], [8.81,120,59.1,3.05,415], [6.29,0,62.5,2.08,386], [8.3,80,59.6,3.36,435],
 [8.54,80,75.9,2.13,648], [6.05,0,57.8,2.98,535], [3.93,0,66.4,1.96,436], [6.95,40,57.9,1.83,488],
 [5.84,40,64.2,1.82,561], [8.31,200,58.9,2.81,626], [7.77,200,59.2,2.34,373], [8.83,160,73.4,3.11,640],
 [7.99,80,59.6,1.89,604], [9.28,160,74,3.14,457], [9.28,160,56.8,2.37,484], [10.55,80,58.7,3.68,564],
 [6.44,80,69.8,3.51,494], [8.6,200,44.6,2.65,532], [9.92,80,50.3,2.88,425], [8.44,200,62.1,2.97,356],
 [7.78,80,65.3,3.12,739], [6.18,40,66,2.61,535], [8.88,200,59.7,3.57,540], [8.22,120,51.9,2.44,566],
 [9.58,120,67.8,1.45,640], [8.23,200,65.1,2.97,604], [4.71,0,54,2.72,427], [10.63,160,62.6,3.38,510],
 [6.94,40,74.7,3.22,526], [8.58,80,62.9,2.89,589], [8.7,200,53.6,2.88,466], [9.05,80,59,2.89,554],
 [5.96,40,59.4,2.34,540], [9.37,120,64.4,1.95,644], [7.89,160,68.4,2.23,285], [9,200,55.3,2.74,527],
 [4.34,0,57.4,1.21,551], [9.67,160,69.2,2.49,581], [8.81,160,67.2,2.11,519], [7.04,80,71.6,2.74,445],
 [6.92,40,51.6,2.23,580], [7.32,80,55.2,2.84,553], [7.51,40,69.1,2.64,444], [7.03,0,55.2,3.71,397],
 [7.94,40,64.3,2.56,570], [7.95,80,61.6,2.63,442], [9.43,120,56.7,2.82,578], [5.06,40,72.4,1.53,656],
 [6.3,40,61.9,2.05,539], [6.17,0,58.4,3.81,541], [8.22,80,60.8,3.16,531], [6.58,0,63.7,2.86,645],
 [7.46,40,50.8,3.56,646], [6.37,40,55.8,2.67,528], [6.78,0,52.9,2.63,587], [8.54,160,47,2.59,524],
 [3.94,0,68.9,1.56,475], [5.5,40,72.2,3.46,589], [6.46,80,70.4,2.67,512], [8.17,120,49.4,2.73,444],
 [8.73,200,64.8,2.57,385], [7.1,120,66.8,2.36,609], [5.78,40,57.9,1.22,433], [8.98,200,67.8,2.91,420],
 [9.57,160,54,4.33,496], [8.76,80,54.3,3.49,587], [8.66,80,56,2.6,365], [8.29,160,56.8,2.15,577],
 [4.35,0,54.8,2.82,555], [4.89,0,68.8,3.06,411], [4.53,0,54.4,2.72,417], [9.05,120,58.7,3.38,504],
 [8.82,80,57.4,2.13,694], [9.44,200,77.6,3.55,497], [9.43,120,55.6,2.91,618], [6.77,0,65.3,2.42,605],
 [8.24,200,74.9,2.6,377], [10.17,200,53.6,3.04,520], [7.96,40,56.1,3.33,609], [5.79,0,59.6,2.85,475]])

y = MILHO[:, 0]; Xv = MILHO[:, 1:]
nomes = ["nitrogênio", "densidade", "mat. orgânica", "chuva"]
n = len(y)

def monta(X, grau_n=1, grau_d=1):
    """Matriz de delineamento: intercepto, as quatro variáveis e termos
       quadráticos opcionais para nitrogênio e densidade."""
    cols = [np.ones(len(X))] + [X[:, j] for j in range(4)]
    if grau_n >= 2: cols.append(X[:, 0]**2)
    if grau_d >= 2: cols.append(X[:, 1]**2)
    return np.column_stack(cols)

def ajusta(Xd, yy):
    return np.linalg.lstsq(Xd, yy, rcond=None)[0]

def eqm(obs, prev):
    return float(np.mean((obs - prev)**2))


def cv_k(Xd, yy, k, semente=0):
    nn = len(yy); rng = np.random.default_rng(semente)
    dob = np.array_split(rng.permutation(nn), k); e = np.empty(nn)
    for d in dob:
        m = np.ones(nn, bool); m[d] = False
        b = ajusta(Xd[m], yy[m]); e[d] = (yy[d] - Xd[d] @ b)**2
    return e.mean()

def loocv_forca_bruta(Xd, yy):
    """n ajustes, um por observação omitida."""
    nn = len(yy); e = np.empty(nn)
    for i in range(nn):
        m = np.ones(nn, bool); m[i] = False
        b = ajusta(Xd[m], yy[m]); e[i] = (yy[i] - Xd[i] @ b)**2
    return e

def loocv_atalho(Xd, yy):
    """UM ajuste só, usando os valores de alavancagem (PRESS)."""
    b = ajusta(Xd, yy); r = yy - Xd @ b
    H = Xd @ np.linalg.pinv(Xd.T @ Xd) @ Xd.T
    h = np.diag(H)
    return (r/(1 - h))**2, h

# ===================================================================
# PARTE 1 - a deixa-um-de-fora, e o atalho que a torna gratuita
# ===================================================================
print(f"{n} talhões de milho. Modelo com as quatro variáveis e os dois")
print("termos quadráticos (p = 7 parâmetros).")
Xd = monta(Xv, 2, 2)

import time
t0 = time.time()
for _ in range(20): e_bruta = loocv_forca_bruta(Xd, y)
t_b = (time.time() - t0)/20
t0 = time.time()
for _ in range(20): e_atalho, h = loocv_atalho(Xd, y)
t_a = (time.time() - t0)/20
print(f"\nA deixa-um-de-fora é a validação cruzada com k = n: são {n} ajustes,")
print("cada um omitindo uma observação.")
print(f"\n  EQM por força bruta ({n} ajustes): {e_bruta.mean():.6f}  "
      f"({t_b*1000:.1f} ms)")
print(f"  EQM pelo atalho (1 ajuste só):    {e_atalho.mean():.6f}  "
      f"({t_a*1000:.1f} ms)")
print(f"  diferença máxima entre as {n} parcelas: "
      f"{np.abs(e_bruta-e_atalho).max():.2e}")
print(f"  o atalho é cerca de {t_b/t_a:.0f} vezes mais rápido, e exato")
print("  (tempos médios de 20 execuções; a vantagem cresce com n, porque a")
print("   força bruta faz n ajustes e o atalho sempre faz um)")
print("\nO atalho vale para QUALQUER modelo linear ajustado por mínimos")
print("quadrados, e é uma identidade algébrica, não uma aproximação:")
print("  resíduo deixa-um-de-fora = resíduo comum / (1 − h_i)")
print("em que h_i é a alavancagem da observação i, o i-ésimo elemento da")
print("diagonal da matriz chapéu H = X(X'X)⁻¹X'.")
print(f"\n  soma das alavancagens = {h.sum():.4f} = p = {Xd.shape[1]}  "
      "(é sempre assim)")
print(f"  alavancagem média = p/n = {h.mean():.4f}")
print(f"  menor {h.min():.4f} | maior {h.max():.4f} "
      f"| observações com h > 2p/n: {int(np.sum(h > 2*Xd.shape[1]/n))}")

print("\nAs cinco observações de maior alavancagem")
o = np.argsort(h)[::-1][:5]
print(f"{'obs.':>6}{'N':>7}{'densid.':>10}{'M.O.':>8}{'chuva':>8}"
      f"{'h':>9}{'resíduo':>10}{'res. LOO':>10}")
b = ajusta(Xd, y); r = y - Xd @ b
for i in o:
    print(f"{i+1:>6}{Xv[i,0]:>7.0f}{Xv[i,1]:>10.1f}{Xv[i,2]:>8.2f}"
          f"{Xv[i,3]:>8.0f}{h[i]:>9.4f}{r[i]:>10.4f}{r[i]/(1-h[i]):>10.4f}")
print("Quanto maior a alavancagem, mais o resíduo cresce ao se remover a")
print("observação: ela puxava o ajuste para si.")

# ===================================================================
# PARTE 2 - comparação com os demais k
# ===================================================================
print("\n" + "=" * 70)
print("A deixa-um-de-fora no contexto dos demais valores de k")
print(f"\n{'k':>6}{'n de treino':>14}{'EQM':>11}{'ajustes':>10}{'viés*':>10}")
loo = e_atalho.mean()
for k in [2, 3, 5, 10, 20, 40, n]:
    vals = [cv_k(Xd, y, k, semente=s) for s in range(200)] if k < n else [loo]
    print(f"{k:>6}{n - n//k:>14}{np.mean(vals):>11.4f}"
          f"{k:>10}{np.mean(vals)-loo:>10.4f}")
print(f"* diferença em relação à deixa-um-de-fora, tomada como referência")
print("\nA deixa-um-de-fora é a de MENOR VIÉS, porque treina com n − 1")
print("observações - praticamente o conjunto inteiro. É essa a sua virtude,")
print("e é a razão de ela ser a referência natural.")

# ===================================================================
# PARTE 3 - a variância: a afirmação clássica, posta à prova
# ===================================================================
print("\n" + "=" * 70)
print("'A deixa-um-de-fora tem variância alta.' A afirmação é correta, mas")
print("refere-se a uma fonte de variância específica, e vale separar as duas.")
print("\nFONTE 1 - o sorteio das dobras, com os DADOS FIXOS.")
print(f"{'k':>6}{'DP entre 500 sorteios':>26}")
for k in [2, 5, 10, 20, n]:
    if k == n:
        print(f"{k:>6}{0.0:>26.4f}   (não há sorteio)")
    else:
        v = [cv_k(Xd, y, k, semente=s) for s in range(500)]
        print(f"{k:>6}{np.std(v, ddof=1):>26.4f}")
print("Nessa fonte, a deixa-um-de-fora é a de MENOR variância: zero, porque")
print("não há nada a sortear. Não é a essa variância que a afirmação se refere.")

print("\nFONTE 2 - a variabilidade entre AMOSTRAS diferentes da mesma")
print("população. É esta que importa, e é esta que a afirmação descreve.")
print("Simulação: 2.000 conjuntos de 120 talhões gerados do mesmo processo")
print("(o mesmo que originou os dados desta ficha), e em cada um calculam-se")
print("as estimativas por k = 5, k = 10 e deixa-um-de-fora.")
def gera(rng, nn=120):
    N_ = rng.choice([0,40,80,120,160,200], nn)
    de = np.round(rng.normal(62,7,nn),1)
    mo = np.round(np.clip(rng.normal(2.8,0.7,nn),1.0,None),2)
    ch = np.round(rng.normal(520,90,nn),0)
    mu = (3.2 + 0.0385*N_ - 0.000105*N_**2 + 0.055*(de-45)
          - 0.0022*(de-45)**2 + 0.62*mo + 0.0032*(ch-400))
    return np.column_stack([N_, de, mo, ch]), mu + rng.normal(0, 0.85, nn)
rng = np.random.default_rng(77)
col = {5: [], 10: [], "loo": []}
for _ in range(2000):
    Xs, ys = gera(rng); Xds = monta(Xs, 2, 2)
    col[5].append(cv_k(Xds, ys, 5, semente=1))
    col[10].append(cv_k(Xds, ys, 10, semente=1))
    col["loo"].append(loocv_atalho(Xds, ys)[0].mean())
print(f"\n{'método':>22}{'média':>11}{'DP entre amostras':>22}{'CV':>9}")
for rot, v in [("k = 5", col[5]), ("k = 10", col[10]),
               ("deixa-um-de-fora", col["loo"])]:
    v = np.array(v)
    print(f"{rot:>22}{v.mean():>11.4f}{v.std(ddof=1):>22.4f}"
          f"{100*v.std(ddof=1)/v.mean():>8.1f}%")
print(f"\n(o valor verdadeiro do erro irredutível é σ² = {0.85**2:.4f})")
print("\nAqui a diferença de variância entre os três é pequena, e a")
print("deixa-um-de-fora sequer é a pior. O resultado contraria a versão")
print("simplificada da afirmação clássica, e a razão é que o modelo é LINEAR e")
print("ESTÁVEL: remover uma observação de 120 quase não muda o ajuste, de modo")
print("que os n erros deixa-um-de-fora não ficam tão correlacionados quanto o")
print("argumento supõe. Com modelos instáveis - árvores, vizinhos mais")
print("próximos, seleção de variáveis - a correlação é forte e a variância da")
print("deixa-um-de-fora cresce de fato.")

# ===================================================================
# PARTE 4 - onde a deixa-um-de-fora FALHA mesmo
# ===================================================================
print("\n" + "=" * 70)
print("O caso em que a deixa-um-de-fora falha de forma espetacular:")
print("um classificador treinado em dados SEM NENHUM SINAL, com as duas")
print("classes em proporções iguais.")
print("\nRegra: classificar tudo na classe MAJORITÁRIA do treino.")
print("Com n par e classes 50/50, remover uma observação SEMPRE deixa a outra")
print("classe em maioria no treino - e a predição é SEMPRE errada.")
rng = np.random.default_rng(9)
for nn in [20, 50, 100]:
    acl, ak5 = [], []
    for _ in range(2000):
        yy = np.r_[np.ones(nn//2), np.zeros(nn//2)]
        yy = rng.permutation(yy)
        # deixa-um-de-fora
        err = 0
        for i in range(nn):
            m = np.ones(nn, bool); m[i] = False
            maj = 1.0 if yy[m].mean() > 0.5 else 0.0
            err += (maj != yy[i])
        acl.append(1 - err/nn)
        # 5-fold
        dob = np.array_split(rng.permutation(nn), 5); e5 = 0
        for d in dob:
            m = np.ones(nn, bool); m[d] = False
            maj = 1.0 if yy[m].mean() > 0.5 else 0.0
            e5 += np.sum(maj != yy[d])
        ak5.append(1 - e5/nn)
    print(f"  n = {nn:>3}: acurácia estimada por deixa-um-de-fora = "
          f"{np.mean(acl):.4f} | por 5-fold = {np.mean(ak5):.4f} "
          f"| verdadeira = 0,5000")
print("\nA deixa-um-de-fora estima acurácia ZERO para um classificador cuja")
print("acurácia verdadeira é 50%. O 5-fold também subestima - 0,34 a 0,42,")
print("conforme n - mas não colapsa, e o seu viés diminui com n enquanto o da")
print("deixa-um-de-fora permanece total. É um exemplo construído,")
print("mas expõe o mecanismo: os n conjuntos de treino da deixa-um-de-fora são")
print("quase idênticos entre si, e qualquer dependência do modelo em relação")
print("ao conjunto inteiro se propaga a todas as n predições de uma vez.")

# ===================================================================
# PARTE 5 - a seleção de modelo pela deixa-um-de-fora
# ===================================================================
print("\n" + "=" * 70)
print("Seleção dos seis modelos candidatos, pela deixa-um-de-fora")
modelos = [("só nitrogênio", lambda X: np.column_stack([np.ones(len(X)), X[:,0]])),
           ("N + densidade", lambda X: np.column_stack([np.ones(len(X)), X[:,0], X[:,1]])),
           ("as quatro, linear", lambda X: monta(X, 1, 1)),
           ("com N²", lambda X: monta(X, 2, 1)),
           ("com densidade²", lambda X: monta(X, 1, 2)),
           ("com os dois quadráticos", lambda X: monta(X, 2, 2))]
print(f"\n{'modelo':<26}{'p':>4}{'resubst.':>11}{'LOOCV':>10}"
      f"{'5-fold':>10}{'AIC':>11}{'BIC':>11}")
for rot, f in modelos:
    Xm = f(Xv); p_ = Xm.shape[1]
    bb = ajusta(Xm, y); sq = np.sum((y - Xm@bb)**2)
    lo = loocv_atalho(Xm, y)[0].mean()
    k5 = np.mean([cv_k(Xm, y, 5, semente=s) for s in range(200)])
    aic = n*np.log(sq/n) + 2*p_
    bic = n*np.log(sq/n) + np.log(n)*p_
    print(f"{rot:<26}{p_:>4}{sq/n:>11.4f}{lo:>10.4f}{k5:>10.4f}"
          f"{aic:>11.2f}{bic:>11.2f}")
print("\nOs quatro critérios concordam em preferir o modelo 'com N²'. A")
print("deixa-um-de-fora e o AIC são assintoticamente equivalentes para modelos")
print("lineares (Stone, 1977), o que explica a coincidência; o BIC penaliza")
print("mais a complexidade e, com n maior, poderia divergir deles.")
