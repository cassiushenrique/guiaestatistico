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


# ===================================================================
# PARTE 1 - uma separação treino-teste, feita uma vez
# ===================================================================
print(f"{n} talhões de milho, produtividade média {y.mean():.3f} t/ha "
      f"(DP {y.std(ddof=1):.3f})")
print("Modelo: produtividade ~ N + densidade + mat. orgânica + chuva,")
print("com termos quadráticos para N e densidade.")

rng = np.random.default_rng(1)
idx = rng.permutation(n)
ntr = int(0.70*n)
tr, te = idx[:ntr], idx[ntr:]
print(f"\nSeparação 70/30: {len(tr)} talhões no treino, {len(te)} no teste")

Xd = monta(Xv, 2, 2)
b = ajusta(Xd[tr], y[tr])
eqm_tr = eqm(y[tr], Xd[tr] @ b)
eqm_te = eqm(y[te], Xd[te] @ b)
print(f"\n{'conjunto':<12}{'n':>6}{'EQM':>12}{'REQM':>10}{'R²':>9}")
for rot, m_, ee in [("treino", tr, eqm_tr), ("teste", te, eqm_te)]:
    r2 = 1 - ee/np.mean((y[m_] - y[tr].mean())**2)
    print(f"{rot:<12}{len(m_):>6}{ee:>12.4f}{np.sqrt(ee):>10.4f}{r2:>9.4f}")
print(f"\nO EQM de treino é {100*(1-eqm_tr/eqm_te):.1f}% menor que o de teste.")
print("A diferença é o otimismo da reclassificação: o modelo foi ajustado")
print("para minimizar o erro justamente nas observações do treino.")

# ===================================================================
# PARTE 2 - o problema: a estimativa depende da separação sorteada
# ===================================================================
print("\n" + "=" * 70)
print("A MESMA análise, repetida com 2.000 separações 70/30 diferentes")
rng = np.random.default_rng(7)
est = []
for _ in range(2000):
    i2 = rng.permutation(n); t2, s2 = i2[:ntr], i2[ntr:]
    bb = ajusta(Xd[t2], y[t2])
    est.append(eqm(y[s2], Xd[s2] @ bb))
est = np.array(est)
print(f"\n  EQM de teste: média {est.mean():.4f}, DP {est.std(ddof=1):.4f}")
print(f"  mínimo {est.min():.4f} | percentil 5 {np.percentile(est,5):.4f} "
      f"| mediana {np.median(est):.4f}")
print(f"  percentil 95 {np.percentile(est,95):.4f} | máximo {est.max():.4f}")
print(f"  razão entre o maior e o menor: {est.max()/est.min():.2f} vezes")
print(f"  coeficiente de variação: {100*est.std(ddof=1)/est.mean():.1f}%")
print("\nA mesma análise, nos mesmos dados, com o mesmo modelo, produz um EQM")
print(f"de teste entre {est.min():.2f} e {est.max():.2f} conforme o sorteio. Reportar UMA")
print("separação é reportar um número que depende da semente aleatória.")

# ===================================================================
# PARTE 3 - a proporção de treino: o compromisso
# ===================================================================
print("\n" + "=" * 70)
print("Qual proporção usar? O compromisso entre VIÉS e VARIÂNCIA da estimativa.")
print("Treino pequeno: o modelo fica pior do que ficaria com todos os dados,")
print("e o EQM é superestimado. Teste pequeno: a estimativa fica instável.")
rng = np.random.default_rng(11)
b_full = ajusta(Xd, y)
print(f"\n{'treino':>8}{'teste':>8}{'EQM médio':>13}{'DP do EQM':>12}"
      f"{'viés aparente':>16}")
# referência: erro verdadeiro do modelo ajustado com TODOS os dados,
# estimado em dados novos simulados do mesmo processo
sig = np.sqrt(np.sum((y - Xd @ b_full)**2)/(n - Xd.shape[1]))
for frac in [0.50, 0.60, 0.70, 0.80, 0.90]:
    k = int(frac*n); vals = []
    for _ in range(3000):
        i2 = rng.permutation(n); t2, s2 = i2[:k], i2[k:]
        bb = ajusta(Xd[t2], y[t2])
        vals.append(eqm(y[s2], Xd[s2] @ bb))
    vals = np.array(vals)
    print(f"{k:>8}{n-k:>8}{vals.mean():>13.4f}{vals.std(ddof=1):>12.4f}"
          f"{vals.mean()-sig**2:>16.4f}")
print(f"\n(σ² estimado com todos os {n} dados = {sig**2:.4f})")
print("Treinos maiores reduzem o viés e AUMENTAM a variância da estimativa,")
print("porque sobra menos para testar. Não há proporção ótima universal; a")
print("faixa de 70% a 80% é a convenção, e a validação cruzada existe")
print("justamente para não ter de escolher.")

# ===================================================================
# PARTE 4 - o uso indevido mais comum: o teste usado duas vezes
# ===================================================================
print("\n" + "=" * 70)
print("Se o conjunto de teste for usado para ESCOLHER o modelo, ele deixa")
print("de estimar o erro. Demonstração: seis modelos candidatos, e o teste")
print("usado para escolher o melhor deles.")
modelos = [("só nitrogênio", lambda X: np.column_stack([np.ones(len(X)), X[:,0]])),
           ("N + densidade", lambda X: np.column_stack([np.ones(len(X)), X[:,0], X[:,1]])),
           ("as quatro, linear", lambda X: monta(X, 1, 1)),
           ("com N²", lambda X: monta(X, 2, 1)),
           ("com densidade²", lambda X: monta(X, 1, 2)),
           ("com os dois quadráticos", lambda X: monta(X, 2, 2))]
rng = np.random.default_rng(3)
i2 = rng.permutation(n); t2, s2 = i2[:ntr], i2[ntr:]
print(f"\n{'modelo':<26}{'p':>4}{'EQM treino':>13}{'EQM teste':>12}")
eq_te = []
for rot, f in modelos:
    Xm = f(Xv); bb = ajusta(Xm[t2], y[t2])
    a, t_ = eqm(y[t2], Xm[t2]@bb), eqm(y[s2], Xm[s2]@bb)
    eq_te.append(t_)
    print(f"{rot:<26}{Xm.shape[1]:>4}{a:>13.4f}{t_:>12.4f}")
jmin = int(np.argmin(eq_te))
print(f"\nO menor EQM de teste é o do modelo '{modelos[jmin][0]}': "
      f"{eq_te[jmin]:.4f}.")
print("Esse número NÃO é uma estimativa honesta do erro desse modelo em dados")
print("novos: ele é o mínimo de seis estimativas ruidosas, e portanto")
print("enviesado para baixo. Quanto? A simulação a seguir mede.")

rng = np.random.default_rng(23)
vies = []
for _ in range(2000):
    i3 = rng.permutation(n)
    t3, v3, s3 = i3[:72], i3[72:96], i3[96:]      # treino, validação, teste
    eqv, eqs = [], []
    for rot, f in modelos:
        Xm = f(Xv); bb = ajusta(Xm[t3], y[t3])
        eqv.append(eqm(y[v3], Xm[v3]@bb))
        eqs.append(eqm(y[s3], Xm[s3]@bb))
    j = int(np.argmin(eqv))
    vies.append((eqv[j], eqs[j]))
vies = np.array(vies)
print(f"\nSeparação em TRÊS partes (72 treino, 24 validação, 24 teste),")
print("2.000 repetições. O modelo é escolhido pela validação e avaliado")
print("no teste, que nunca foi usado para escolher nada.")
print(f"  EQM no conjunto de VALIDAÇÃO (usado para escolher) = {vies[:,0].mean():.4f}")
print(f"  EQM no conjunto de TESTE     (nunca usado)         = {vies[:,1].mean():.4f}")
print(f"  otimismo = {vies[:,1].mean()-vies[:,0].mean():.4f} "
      f"({100*(vies[:,1].mean()/vies[:,0].mean()-1):.1f}%)")
print("\nO conjunto usado para escolher subestima o erro em cerca de 9%, e o")
print("efeito cresce com o número de modelos comparados.")
print("É por isso que, quando há seleção de modelo, a separação precisa ser")
print("em TRÊS partes: treino para ajustar, validação para escolher, teste")
print("para estimar o erro - e o teste é aberto uma única vez, no fim.")

# ===================================================================
# PARTE 5 - estratificação, e quando ela importa
# ===================================================================
print("\n" + "=" * 70)
print("Em problemas de CLASSIFICAÇÃO com classes raras, uma separação")
print("aleatória pode produzir conjuntos muito desequilibrados.")
rng = np.random.default_rng(5)
N_, prev = 200, 0.10
print(f"\nSimulação: {N_} unidades, {100*prev:.0f}% da classe rara, "
      f"separação 70/30, 10.000 sorteios")
rotulos = np.r_[np.ones(int(N_*prev)), np.zeros(N_-int(N_*prev))]
prop = []
for _ in range(10000):
    i4 = rng.permutation(N_); s4 = i4[int(0.7*N_):]
    prop.append(rotulos[s4].sum())
prop = np.array(prop)
print(f"  casos da classe rara no teste: esperado "
      f"{prev*N_*0.3:.1f}, observado de {prop.min():.0f} a {prop.max():.0f}")
print(f"  em {100*np.mean(prop <= 2):.2f}% dos sorteios o teste tem 2 casos ou menos")
print(f"  em {100*np.mean(prop == 0):.3f}% dos sorteios o teste não tem NENHUM")
print("A separação ESTRATIFICADA sorteia dentro de cada classe e garante a")
print("mesma proporção nos dois conjuntos. Com classes raras, séries")
print("temporais ou dados agrupados, a separação puramente aleatória não serve.")
