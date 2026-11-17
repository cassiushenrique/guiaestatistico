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
# PARTE 1 - a decomposição, verificada numericamente
# ===================================================================
print("O erro quadrático esperado de uma predição se decompõe em três")
print("parcelas, e a decomposição é uma IDENTIDADE, não uma aproximação:")
print("  E[(y − f̂(x))²] = [viés de f̂(x)]² + Var[f̂(x)] + σ²")
print("\nVerificação numérica em um processo conhecido: y = sen(2x) + ε,")
print("com x em [0, 3] e ε ~ N(0; 0,40²). Ajusta-se um polinômio de grau g")
print("a 2.000 amostras independentes de n = 50 pontos, e avalia-se tudo em")
print("uma grade fixa de x. A base polinomial é ortogonal (Legendre), para")
print("que o que se vê seja estatística e não erro de arredondamento.")

def verdade(x): return np.sin(2*x)

def base(x, grau):
    """Base polinomial ORTOGONAL (Legendre) em [0, 3], bem condicionada.
       Usar a matriz de Vandermonde crua produziria instabilidade numérica
       com grau alto, que não é um fenômeno estatístico e sim de aritmética."""
    return np.polynomial.legendre.legvander(2*np.asarray(x)/3 - 1, grau)
SIG = 0.40
grade = np.linspace(0.15, 2.85, 60)
rng = np.random.default_rng(1)

def experimento(grau, reps=2000, nn=50):
    P = np.empty((reps, len(grade)))
    for r in range(reps):
        xs = rng.uniform(0, 3, nn)
        ys = verdade(xs) + rng.normal(0, SIG, nn)
        A = base(xs, grau); B = base(grade, grau)
        co = np.linalg.lstsq(A, ys, rcond=None)[0]
        P[r] = B @ co
    med = P.mean(axis=0)
    vies2 = float(np.mean((med - verdade(grade))**2))
    var = float(np.mean(P.var(axis=0)))
    return vies2, var, P

print(f"\n{'grau':>6}{'viés²':>11}{'variância':>12}{'σ²':>9}"
      f"{'soma':>10}{'EQM simulado':>15}{'dif.':>10}")
res = []
for g in range(0, 10):
    v2, va, P = experimento(g)
    # EQM simulado: erro de predição em pontos novos sobre a grade
    rng2 = np.random.default_rng(100+g)
    eq = float(np.mean((P - (verdade(grade) +
                             rng2.normal(0, SIG, P.shape)))**2))
    res.append((g, v2, va, v2+va+SIG**2, eq))
    print(f"{g:>6}{v2:>11.4f}{va:>12.4f}{SIG**2:>9.4f}"
          f"{v2+va+SIG**2:>10.4f}{eq:>15.4f}{eq-(v2+va+SIG**2):>10.4f}")
print("\nA coluna 'soma' e a coluna 'EQM simulado' coincidem até o erro de")
print("Monte Carlo: a identidade se verifica. O que ela diz é que não há")
print("como reduzir o erro abaixo de σ² = 0,1600, por melhor que seja o")
print("modelo - é o ERRO IRREDUTÍVEL, a variação da própria natureza.")

jm = int(np.argmin([r[4] for r in res]))
print(f"\nO menor erro total é o do grau {res[jm][0]}, com {res[jm][4]:.4f}.")
print("O viés² cai de 0,5620 para praticamente zero entre os graus 0 e 3, e")
print("a variância sobe de 0,0135 para 0,0605 entre os graus 0 e 9. A soma")
print("desenha a curva em U, com mínimo no grau 3 - que é, não por acaso, o")
print("menor grau capaz de reproduzir uma senoide nesse intervalo.")

print("\nO MESMO experimento com n = 25, metade dos dados. Aqui a MÉDIA da")
print("variância deixa de resumir bem, e vale olhar também a mediana e a")
print("cauda: o desastre é raro, mas quando acontece é enorme.")
print(f"\n{'grau':>6}{'variância média':>18}{'erro MEDIANO':>15}"
      f"{'% de ajustes com':>19}")
print(f"{'':>6}{'':>18}{'por amostra':>15}{'erro > 1':>19}")
for g in [0, 3, 5, 7, 8, 9]:
    v2, va, P = experimento(g, reps=1500, nn=25)
    porAmostra = ((P - verdade(grade))**2).mean(axis=1)
    print(f"{g:>6}{va:>18.4f}{np.median(porAmostra):>15.4f}"
          f"{100*np.mean(porAmostra > 1):>18.1f}%")
print("No grau 9 com n = 25, metade dos ajustes tem erro mediano modesto e")
print("uma fração deles erra por ordens de grandeza - é isso que faz a média")
print("da variância saltar para as centenas. Não é erro numérico: é o")
print("estimador de mínimos quadrados fazendo o que faz quando há poucos")
print("dados por parâmetro, sobretudo perto das bordas do domínio. A lição é")
print("que, com modelos de alta variância, a média do erro pode ser um")
print("resumo enganoso, e convém olhar a distribuição inteira.")

# ===================================================================
# PARTE 2 - o que acontece em cada regime
# ===================================================================
print("\n" + "=" * 70)
print("Os três regimes, lado a lado")
print(f"\n{'':<22}{'grau 0':>12}{'grau 3':>12}{'grau 9':>12}")
tab = {g: res[g] for g in [0, 3, 9]}
for rot, i in [("viés²", 1), ("variância", 2), ("erro total", 4)]:
    print(f"{rot:<22}" + "".join(f"{tab[g][i]:>12.4f}" for g in [0, 3, 9]))
for rot, i in [("% do erro: viés²", 1), ("% do erro: variância", 2)]:
    print(f"{rot:<22}" + "".join(f"{100*tab[g][i]/tab[g][4]:>11.1f}%"
                                 for g in [0, 3, 9]))
print(f"{'% do erro: σ²':<22}" + "".join(f"{100*SIG**2/tab[g][4]:>11.1f}%"
                                         for g in [0, 3, 9]))
print("\nO grau 0 - uma constante - erra por rigidez: 76% do erro é viés.")
print("O grau 9 erra por instabilidade: o viés some por completo e a")
print("variância passa a responder por 27% do erro, contra 2% no grau 0.")
print("No grau 3, o erro irredutível responde por 92% do total, e esse é o")
print("sinal de que não há mais nada a ganhar ajustando a complexidade:")
print("o que resta é a variação da própria natureza.")

# ===================================================================
# PARTE 3 - o efeito do tamanho da amostra
# ===================================================================
print("\n" + "=" * 70)
print("A complexidade ótima DEPENDE de n. Com mais dados, a variância de um")
print("modelo flexível cai, e vale a pena usá-lo.")
print(f"\n{'n':>6}" + "".join(f"{'g='+str(g):>10}" for g in range(1, 10))
      + f"{'melhor':>9}")
for nn in [15, 25, 50, 100, 400]:
    linha = []
    for g in range(1, 10):
        if g + 1 > nn - 1: linha.append(np.nan); continue
        v2, va, _ = experimento(g, reps=600, nn=nn)
        linha.append(v2 + va + SIG**2)
    best = int(np.nanargmin(linha)) + 1
    print(f"{nn:>6}" + "".join(f"{v:>10.4f}" if np.isfinite(v) else f"{'-':>10}"
                               for v in linha) + f"{best:>9}")
print("\nCom n = 15 o melhor grau é baixo; com n = 400, graus altos deixam de")
print("ser penalizados, porque a variância cai com 1/n. Não existe")
print("'complexidade certa' de um modelo: existe complexidade certa PARA UM")
print("TAMANHO DE AMOSTRA.")

# ===================================================================
# PARTE 4 - o mesmo fenômeno em k vizinhos mais próximos
# ===================================================================
print("\n" + "=" * 70)
print("O compromisso não é sobre polinômios: é sobre flexibilidade. Em k")
print("vizinhos mais próximos, quem controla a flexibilidade é o k, e ao")
print("CONTRÁRIO do grau, flexibilidade maior corresponde a k MENOR.")

def exp_knn(k, reps=1500, nn=80):
    P = np.empty((reps, len(grade)))
    for r in range(reps):
        xs = rng.uniform(0, 3, nn); ys = verdade(xs) + rng.normal(0, SIG, nn)
        d = np.abs(grade[:, None] - xs[None, :])
        viz = np.argsort(d, axis=1)[:, :k]
        P[r] = ys[viz].mean(axis=1)
    med = P.mean(axis=0)
    return float(np.mean((med - verdade(grade))**2)), float(np.mean(P.var(axis=0)))

print(f"\n{'k':>5}{'graus de liberdade ≈ n/k':>26}{'viés²':>10}"
      f"{'variância':>12}{'erro total':>13}")
melhor = None
for k in [1, 2, 3, 5, 10, 20, 40, 80]:
    v2, va = exp_knn(k)
    tot = v2 + va + SIG**2
    if melhor is None or tot < melhor[1]: melhor = (k, tot)
    print(f"{k:>5}{80/k:>26.1f}{v2:>10.4f}{va:>12.4f}{tot:>13.4f}")
print(f"\nO melhor k é {melhor[0]}, com erro {melhor[1]:.4f}.")
print("Com k = 1 a predição é o valor do vizinho mais próximo: viés quase")
print(f"nulo ({exp_knn(1)[0]:.4f}) e variância máxima. Com k = 80 a predição é a")
print("média de tudo: variância mínima e viés enorme. É a mesma curva em U,")
print("percorrida na direção oposta.")

# ===================================================================
# PARTE 5 - o que a decomposição NÃO diz
# ===================================================================
print("\n" + "=" * 70)
print("Três leituras equivocadas, e o que os dados mostram.")

print("\n(a) 'O erro de TREINO também tem curva em U.' Não tem: ele cai")
print("    monotonicamente com a complexidade.")
print("    (uma única amostra de n = 25; o erro de predição é calculado")
print("     contra a função verdadeira, sobre a grade, mais σ²)")
print(f"\n{'grau':>6}{'p':>4}{'EQM de treino':>16}{'EQM de predição':>18}")
rng3 = np.random.default_rng(55)
xs = rng3.uniform(0, 3, 25); ys = verdade(xs) + rng3.normal(0, SIG, 25)
for g in [0, 1, 2, 3, 5, 9, 15, 20, 24]:
    A = base(xs, g); B = base(grade, g)
    co = np.linalg.lstsq(A, ys, rcond=None)[0]
    tre = float(np.mean((ys - A @ co)**2))
    pre = float(np.mean((B @ co - verdade(grade))**2)) + SIG**2
    print(f"{g:>6}{g+1:>4}{tre:>16.4f}{pre:>18.4f}")
print("    Com grau 24 e 25 pontos há tantos parâmetros quanto observações:")
print("    o polinômio passa por todos os pontos e o erro de treino é ZERO.")
print("    O erro de predição, nesse ponto, é o maior de toda a tabela.")

print("\n(b) 'Viés é sempre ruim.' Não é: a regressão RIDGE introduz viés de")
print("    propósito, e reduz o erro total quando a redução de variância")
print("    compensa.")
lam_g = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
print(f"\n{'λ (ridge, grau 9)':>20}{'viés²':>10}{'variância':>12}{'erro total':>13}")
def exp_ridge(lam, grau=9, reps=1500, nn=25):
    # n = 25 de propósito: é onde o grau 9 é instável e a regularização paga
    P = np.empty((reps, len(grade)))
    for r in range(reps):
        xs_ = rng.uniform(0, 3, nn); ys_ = verdade(xs_) + rng.normal(0, SIG, nn)
        A = base(xs_, grau); B = base(grade, grau)
        co = np.linalg.solve(A.T @ A + lam*np.eye(grau+1), A.T @ ys_)
        P[r] = B @ co
    med = P.mean(axis=0)
    return float(np.mean((med - verdade(grade))**2)), float(np.mean(P.var(axis=0)))
for lam in lam_g:
    v2, va = exp_ridge(lam)
    print(f"{lam:>20g}{v2:>10.4f}{va:>12.4f}{v2+va+SIG**2:>13.4f}")
print("    O λ que minimiza o erro NÃO é zero. Aceitar um pouco de viés em")
print("    troca de muito menos variância é o princípio de toda a")
print("    regularização.")

print("\n(c) 'Mais flexibilidade sempre aumenta a variância.' Em média sim,")
print("    mas a decomposição é PONTUAL: ela vale em cada x, e o perfil")
print("    muda ao longo do domínio.")
v2g, vag, Pg = experimento(3, reps=2000)
med = Pg.mean(axis=0)
b2x = (med - verdade(grade))**2; vax = Pg.var(axis=0)
print(f"\n{'região de x':<22}{'viés² médio':>14}{'variância média':>18}")
for rot, m_ in [("bordas (x<0,6 ou >2,4)", (grade < 0.6) | (grade > 2.4)),
                ("miolo (0,6 ≤ x ≤ 2,4)", (grade >= 0.6) & (grade <= 2.4))]:
    print(f"{rot:<22}{b2x[m_].mean():>14.4f}{vax[m_].mean():>18.4f}")
print("    Nas bordas do domínio a variância é cerca de 70% maior que no")
print("    miolo, com o mesmo viés, porque há menos dados de cada lado para")
print("    ancorar o ajuste. Com grau mais alto a diferença é muito maior.")
print("    É por isso que extrapolar é perigoso mesmo com o modelo certo.")
