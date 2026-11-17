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


def cv_k(Xd, yy, k, semente=0, repeticoes=1, por_dobra=False):
    """Validação cruzada k-fold. Devolve o EQM de cada repetição."""
    nn = len(yy); rng = np.random.default_rng(semente); out = []; pd_ = []
    for _ in range(repeticoes):
        ordem = rng.permutation(nn)
        dobras = np.array_split(ordem, k)
        erros = np.empty(nn); md = []
        for d in dobras:
            m = np.ones(nn, bool); m[d] = False
            b = ajusta(Xd[m], yy[m])
            erros[d] = (yy[d] - Xd[d] @ b)**2
            md.append(erros[d].mean())
        out.append(erros.mean()); pd_.append(md)
    return (np.array(out), np.array(pd_)) if por_dobra else np.array(out)

# ===================================================================
# PARTE 1 - como funciona, e o que ela entrega
# ===================================================================
print(f"{n} talhões de milho. Modelo com as quatro variáveis e os dois")
print("termos quadráticos (p = 7 parâmetros).")
Xd = monta(Xv, 2, 2)

rng = np.random.default_rng(0)
ordem = rng.permutation(n); dobras = np.array_split(ordem, 5)
print(f"\nValidação cruzada com k = 5: as {n} observações são divididas em")
print("cinco dobras, e cada uma serve de teste uma vez.")
print(f"\n{'dobra':>7}{'n teste':>10}{'n treino':>11}{'EQM da dobra':>15}")
tot = np.empty(n)
for i, d in enumerate(dobras):
    m = np.ones(n, bool); m[d] = False
    b = ajusta(Xd[m], y[m]); e = (y[d] - Xd[d] @ b)**2
    tot[d] = e
    print(f"{i+1:>7}{len(d):>10}{int(m.sum()):>11}{e.mean():>15.4f}")
print(f"{'média':>7}{'':>10}{'':>11}{tot.mean():>15.4f}")
print("\nTodas as 120 observações foram testadas exatamente uma vez, e cada")
print("modelo foi ajustado com 96 delas. A média dos erros é a estimativa.")
print(f"\nPara comparar: o EQM de RESUBSTITUIÇÃO, calculado nos próprios dados")
b_full = ajusta(Xd, y)
print(f"de ajuste, é {eqm(y, Xd@b_full):.4f} - cerca de "
      f"{100*(1-eqm(y, Xd@b_full)/tot.mean()):.0f}% menor que o da validação cruzada.")

# ===================================================================
# PARTE 2 - a escolha de k
# ===================================================================
print("\n" + "=" * 70)
print("Qual k usar? Cada valor é um compromisso entre viés, variância e custo.")
print("A simulação repete cada k 500 vezes, com sorteios diferentes das dobras.")
print(f"\n{'k':>5}{'n de treino':>13}{'EQM médio':>12}{'DP entre':>11}"
      f"{'ajustes':>10}")
print(f"{'':>5}{'por dobra':>13}{'':>12}{'sorteios':>11}{'':>10}")
for k in [2, 3, 5, 10, 20, 60, 120]:
    v = cv_k(Xd, y, k, semente=k, repeticoes=(500 if k < 120 else 1))
    ntr_ = n - n//k
    dp = v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"{k:>5}{ntr_:>13}{v.mean():>12.4f}{dp:>11.4f}{k:>10}")
print("\nCom k pequeno, cada modelo é ajustado com menos dados e o erro é")
print("SUPERESTIMADO: k = 2 treina com 60 observações e devolve 0,8372 contra")
print("0,7854 de k = 120. É o viés, e ele desaparece à medida que k cresce.")
print("\nA coluna do DP merece cuidado, porque ela mede UMA das duas fontes de")
print("variância, e não a que a literatura costuma discutir. Aqui os dados são")
print("FIXOS e só o sorteio das dobras muda: essa variabilidade cai com k, e é")
print("zero para k = n, porque a deixa-um-de-fora não tem sorteio nenhum.")
print("A afirmação clássica de que a deixa-um-de-fora tem variância ALTA refere-se")
print("à outra fonte - a variabilidade da estimativa entre AMOSTRAS diferentes da")
print("mesma população -, e a ficha da deixa-um-de-fora a examina em separado.")
print("\nA recomendação usual de k = 5 ou k = 10 é um compromisso entre viés e")
print("custo computacional, e nestes dados os dois dão praticamente o mesmo.")

# ===================================================================
# PARTE 3 - validação cruzada REPETIDA
# ===================================================================
print("\n" + "=" * 70)
print("Uma única rodada de 10-fold ainda depende do sorteio das dobras.")
v10 = cv_k(Xd, y, 10, semente=99, repeticoes=1000)
print(f"\n1.000 rodadas independentes de 10-fold nos mesmos dados:")
print(f"  média {v10.mean():.4f} | DP {v10.std(ddof=1):.4f}")
print(f"  mínimo {v10.min():.4f} | máximo {v10.max():.4f} "
      f"| amplitude {v10.max()-v10.min():.4f}")
print(f"  coeficiente de variação {100*v10.std(ddof=1)/v10.mean():.2f}%")
print("\nCompare com a separação única 70/30 da ficha do holdout, cujo EQM")
print("variou de 0,37 a 1,34 (CV de 17%). A validação cruzada reduz a")
print("variabilidade do sorteio por um fator de oito, mas não a elimina.")
print("\nA validação cruzada REPETIDA é a solução: rode r vezes e tire a média.")
print(f"\n{'repetições':>12}{'DP da estimativa':>20}{'redução':>12}")
rng2 = np.random.default_rng(55); dps = []
for r in [1, 3, 5, 10, 25]:
    medias = [cv_k(Xd, y, 10, semente=int(rng2.integers(1e6)),
                   repeticoes=r).mean() for _ in range(300)]
    dps.append(np.std(medias, ddof=1))
for r, d in zip([1, 3, 5, 10, 25], dps):
    print(f"{r:>12}{d:>20.4f}{d/dps[0]:>12.2f}")
print("A redução segue a raiz de r, como qualquer média. Com 10 repetições, a")
print("variabilidade do sorteio deixa de ser relevante diante do erro amostral.")

# ===================================================================
# PARTE 4 - a seleção de modelo, feita direito
# ===================================================================
print("\n" + "=" * 70)
print("Comparando os seis modelos candidatos por 10-fold repetida 20 vezes")
modelos = [("só nitrogênio", lambda X: np.column_stack([np.ones(len(X)), X[:,0]])),
           ("N + densidade", lambda X: np.column_stack([np.ones(len(X)), X[:,0], X[:,1]])),
           ("as quatro, linear", lambda X: monta(X, 1, 1)),
           ("com N²", lambda X: monta(X, 2, 1)),
           ("com densidade²", lambda X: monta(X, 1, 2)),
           ("com os dois quadráticos", lambda X: monta(X, 2, 2))]
print("(o erro-padrão é calculado ENTRE AS DOBRAS, que é o que a regra de um")
print(" erro-padrão usa - e não entre as repetições)")
print(f"\n{'modelo':<26}{'p':>4}{'resubstituição':>16}{'EQM 10-fold':>14}"
      f"{'EP':>9}")
res = []
for rot, f in modelos:
    Xm = f(Xv)
    v, pd_ = cv_k(Xm, y, 10, semente=4, repeticoes=20, por_dobra=True)
    # erro-padrão ENTRE DOBRAS, que é o usado na regra de um erro-padrão
    ep = pd_.mean(axis=0).std(ddof=1)/np.sqrt(10)
    bb = ajusta(Xm, y)
    res.append((rot, Xm.shape[1], v.mean(), ep))
    print(f"{rot:<26}{Xm.shape[1]:>4}{eqm(y, Xm@bb):>16.4f}{v.mean():>14.4f}"
          f"{ep:>9.4f}")
j = int(np.argmin([r[2] for r in res]))
print(f"\nMenor EQM de validação cruzada: '{res[j][0]}' com {res[j][2]:.4f}.")
print("Note que o EQM de resubstituição cai monotonicamente com p - ele sempre")
print("cai - enquanto o da validação cruzada tem um mínimo. É essa diferença")
print("que torna a validação cruzada útil para selecionar modelos.")

# a regra de um erro-padrão
lim = res[j][2] + res[j][3]
simples = [r for r in res if r[2] <= lim]
js = min(simples, key=lambda r: r[1])
print(f"\nRegra de UM ERRO-PADRÃO: entre os modelos cujo EQM não excede")
print(f"{res[j][2]:.4f} + {res[j][3]:.4f} = {lim:.4f}, escolha o mais simples.")
print(f"  modelos elegíveis: " + ", ".join(f"{r[0]} (p={r[1]})" for r in simples))
print(f"  escolhido: '{js[0]}', com p = {js[1]}")
print("A regra privilegia a parcimônia quando a diferença de desempenho está")
print("dentro do ruído da própria estimativa.")

# ===================================================================
# PARTE 5 - o erro mais caro: selecionar e avaliar na MESMA validação
# ===================================================================
print("\n" + "=" * 70)
print("Se a mesma validação cruzada escolhe o modelo E estima o erro, a")
print("estimativa fica otimista. A solução é a validação cruzada ANINHADA:")
print("um laço interno escolhe, um laço externo avalia.")

def cv_simples(Xlist, yy, k=5, semente=0):
    """Escolhe o modelo e devolve o MENOR EQM - o procedimento errado."""
    e = [cv_k(Xm, yy, k, semente)[0] for Xm in Xlist]
    return min(e), int(np.argmin(e))

def cv_aninhada(flist, Xraw, yy, kext=5, kint=5, semente=0):
    nn = len(yy); rng = np.random.default_rng(semente)
    dob = np.array_split(rng.permutation(nn), kext)
    erros = np.empty(nn)
    for d in dob:
        m = np.ones(nn, bool); m[d] = False
        # laço INTERNO: escolhe o modelo usando só o treino externo
        eint = [cv_k(f(Xraw[m]), yy[m], kint, semente+1)[0] for f in flist]
        jbest = int(np.argmin(eint))
        Xb = flist[jbest](Xraw)
        b = ajusta(Xb[m], yy[m])
        erros[d] = (yy[d] - Xb[d] @ b)**2
    return erros.mean()

flist = [f for _, f in modelos]
rng = np.random.default_rng(31)
a_err, b_err = [], []
for _ in range(300):
    s = int(rng.integers(1e6))
    Xl = [f(Xv) for f in flist]
    e_sel, _ = cv_simples(Xl, y, 5, s)
    a_err.append(e_sel)
    b_err.append(cv_aninhada(flist, Xv, y, 5, 5, s))
a_err = np.array(a_err); b_err = np.array(b_err)
print(f"\n300 repetições, com 5 dobras externas e 5 internas:")
print(f"  EQM da validação cruzada SIMPLES (seleciona e avalia) = {a_err.mean():.4f}")
print(f"  EQM da validação cruzada ANINHADA (separa as duas)    = {b_err.mean():.4f}")
print(f"  otimismo = {b_err.mean()-a_err.mean():.4f} "
      f"({100*(b_err.mean()/a_err.mean()-1):.1f}%)")
print("\nO otimismo é modesto aqui porque há apenas seis candidatos e um deles")
print("é claramente melhor. Com dezenas de modelos, ou com ajuste fino de")
print("hiperparâmetros contínuos, ele cresce e pode dominar a estimativa.")
print("Regra prática: toda escolha feita olhando os dados - variáveis,")
print("transformações, hiperparâmetros - tem de estar DENTRO do laço de")
print("validação, e não antes dele.")
