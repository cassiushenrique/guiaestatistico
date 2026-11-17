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


def boot(dados, estat, B=10000, semente=0):
    """Reamostragem com reposição das UNIDADES."""
    rng = np.random.default_rng(semente); nn = len(dados)
    return np.array([estat(dados[rng.integers(0, nn, nn)]) for _ in range(B)])

def ic_percentil(bs, alfa=0.05):
    return np.percentile(bs, [100*alfa/2, 100*(1-alfa/2)])

def ic_basico(bs, t0, alfa=0.05):
    lo, hi = np.percentile(bs, [100*alfa/2, 100*(1-alfa/2)])
    return 2*t0 - hi, 2*t0 - lo

def ic_normal(bs, t0, alfa=0.05):
    z = stats.norm.ppf(1-alfa/2); vies = bs.mean() - t0
    return t0 - vies - z*bs.std(ddof=1), t0 - vies + z*bs.std(ddof=1)

def ic_bca(dados, estat, bs, t0, alfa=0.05):
    """Viés corrigido e acelerado (BCa), de Efron (1987)."""
    nn = len(dados)
    z0 = stats.norm.ppf(np.mean(bs < t0))
    jk = np.array([estat(np.delete(dados, i, axis=0)) for i in range(nn)])
    d = jk.mean() - jk
    a = (d**3).sum()/(6*((d**2).sum())**1.5)
    z = stats.norm.ppf([alfa/2, 1-alfa/2])
    q = stats.norm.cdf(z0 + (z0 + z)/(1 - a*(z0 + z)))
    return np.percentile(bs, 100*q), z0, a

# ===================================================================
# PARTE 1 - a ideia, em uma estatística sem erro-padrão em forma fechada
# ===================================================================
print(f"{n} talhões de milho. A pergunta: qual a precisão da MEDIANA da")
print("produtividade, uma estatística sem fórmula simples de erro-padrão?")
med0 = float(np.median(y))
print(f"\n  mediana observada = {med0:.4f} t/ha")
print(f"  média observada   = {y.mean():.4f} t/ha  "
      f"(EP pela fórmula = {y.std(ddof=1)/np.sqrt(n):.4f})")

bs_med = boot(y, np.median, 10000, 1)
print(f"\nBootstrap com B = 10.000 reamostras da mediana")
print(f"  média das reamostras = {bs_med.mean():.4f}")
print(f"  viés estimado        = {bs_med.mean()-med0:+.4f}")
print(f"  ERRO-PADRÃO estimado = {bs_med.std(ddof=1):.4f}")
print(f"  IC 95% percentil     = [{ic_percentil(bs_med)[0]:.4f}; "
      f"{ic_percentil(bs_med)[1]:.4f}]")
print("\nO erro-padrão da mediana não tem fórmula elementar; o bootstrap o")
print("obtém sem supor nenhuma distribuição. Para comparar, a aproximação")
print("assintótica 1/(2·f(m)·raiz(n)), com a densidade estimada por núcleo:")
kde = stats.gaussian_kde(y)
print(f"  aproximação assintótica = {1/(2*float(kde(med0)[0])*np.sqrt(n)):.4f}")

print("\nQuantos B são necessários? O bootstrap tem DUAS fontes de erro: a")
print("amostral, que B não reduz, e a de Monte Carlo, que B reduz.")
print(f"\n{'B':>8}{'EP estimado':>14}{'variação entre 50 execuções':>31}")
for B in [50, 200, 1000, 5000, 20000]:
    eps = [boot(y, np.median, B, s).std(ddof=1) for s in range(50)]
    print(f"{B:>8}{np.mean(eps):>14.4f}{np.std(eps, ddof=1):>31.4f}")
print("Com B = 1.000 a terceira casa já está estável. Para INTERVALOS, que")
print("dependem de percentis extremos, convém B = 10.000 ou mais.")

# ===================================================================
# PARTE 2 - os quatro intervalos, e quando eles divergem
# ===================================================================
print("\n" + "=" * 70)
print("Quatro maneiras de construir o intervalo, aplicadas a três estatísticas")
dados = MILHO
def est_media(d):  return float(np.mean(d[:, 0]))
def est_mediana(d): return float(np.median(d[:, 0]))
def est_cv(d):      return float(np.std(d[:, 0], ddof=1)/np.mean(d[:, 0]))
def est_r(d):       return float(np.corrcoef(d[:, 0], d[:, 3])[0, 1])
def est_raz(d):     return float(np.mean(d[d[:,1] >= 120, 0])/np.mean(d[d[:,1] < 120, 0]))

for nome, f, extra in [("média da produtividade", est_media, True),
                       ("mediana da produtividade", est_mediana, False),
                       ("coeficiente de variação", est_cv, False),
                       ("correlação y × mat. orgânica", est_r, False),
                       ("razão N alto / N baixo", est_raz, False)]:
    t0 = f(dados); bsv = boot(dados, f, 10000, 2)
    bca, z0, a = ic_bca(dados, f, bsv, t0)
    print(f"\n  {nome}: estimativa = {t0:.4f}")
    print(f"    {'percentil':<14}[{ic_percentil(bsv)[0]:8.4f}; "
          f"{ic_percentil(bsv)[1]:8.4f}]   largura {np.diff(ic_percentil(bsv))[0]:.4f}")
    bb = ic_basico(bsv, t0)
    print(f"    {'básico':<14}[{bb[0]:8.4f}; {bb[1]:8.4f}]   largura {bb[1]-bb[0]:.4f}")
    nn_ = ic_normal(bsv, t0)
    print(f"    {'normal':<14}[{nn_[0]:8.4f}; {nn_[1]:8.4f}]   largura {nn_[1]-nn_[0]:.4f}")
    print(f"    {'BCa':<14}[{bca[0]:8.4f}; {bca[1]:8.4f}]   largura {bca[1]-bca[0]:.4f}"
          f"   (z0 = {z0:+.4f}, a = {a:+.4f})")
    if extra:
        t_ = stats.t.ppf(0.975, n-1)*np.std(dados[:,0], ddof=1)/np.sqrt(n)
        print(f"    {'teoria (t)':<14}[{t0-t_:8.4f}; {t0+t_:8.4f}]   "
              f"largura {2*t_:.4f}")
print("\nPara a média, os quatro bootstraps e a fórmula clássica praticamente")
print("coincidem - é o caso em que o bootstrap não traz nada de novo. Para a")
print("correlação e a razão de médias, que têm distribuições assimétricas,")
print("os intervalos se separam, e os parâmetros z0 e a do BCa medem")
print("exatamente essa assimetria: z0 corrige o viés da mediana das")
print("reamostras e a corrige a variação do erro-padrão com o parâmetro.")

# ===================================================================
# PARTE 3 - a cobertura real dos quatro intervalos
# ===================================================================
print("\n" + "=" * 70)
print("Qual deles realmente cobre 95% das vezes? Simulação com 2.000")
print("amostras de uma população CONHECIDA, em quatro situações.")
rng = np.random.default_rng(13)
def cobertura(gera, estat, verdade, nn, reps=2000, B=1200):
    c = np.zeros(4); larg = np.zeros(4)
    for _ in range(reps):
        d = gera(nn)
        t0 = estat(d); bsv = boot(d, estat, B, int(rng.integers(1e6)))
        ics = [ic_percentil(bsv), ic_basico(bsv, t0), ic_normal(bsv, t0),
               ic_bca(d, estat, bsv, t0)[0]]
        for j, (lo, hi) in enumerate(ics):
            c[j] += (lo <= verdade <= hi); larg[j] += hi - lo
    return 100*c/reps, larg/reps

print(f"\n{'situação':<34}{'percentil':>11}{'básico':>9}{'normal':>9}{'BCa':>8}")
# (a) média de uma normal, n = 30
g1 = lambda nn: rng.normal(10, 2, nn).reshape(-1,1)
c1, l1 = cobertura(g1, lambda d: float(d[:,0].mean()), 10.0, 30)
print(f"{'média, normal, n = 30':<34}" + "".join(f"{v:>8.1f}%" for v in c1[:1])
      + "".join(f"{v:>8.1f}%" for v in c1[1:]))
# (b) média de uma exponencial (assimétrica), n = 20
g2 = lambda nn: rng.exponential(5, nn).reshape(-1,1)
c2, l2 = cobertura(g2, lambda d: float(d[:,0].mean()), 5.0, 20)
print(f"{'média, exponencial, n = 20':<34}" + "".join(f"{v:>8.1f}%" for v in c2))
# (c) variância de uma normal, n = 25
g3 = lambda nn: rng.normal(0, 3, nn).reshape(-1,1)
c3, l3 = cobertura(g3, lambda d: float(np.var(d[:,0], ddof=1)), 9.0, 25)
print(f"{'variância, normal, n = 25':<34}" + "".join(f"{v:>8.1f}%" for v in c3))
# (d) mediana de uma log-normal, n = 40
g4 = lambda nn: np.exp(rng.normal(0, 1, nn)).reshape(-1,1)
c4, l4 = cobertura(g4, lambda d: float(np.median(d[:,0])), 1.0, 40)
print(f"{'mediana, log-normal, n = 40':<34}" + "".join(f"{v:>8.1f}%" for v in c4))
print("\nNenhum método atinge 95% em todas as situações, e a leitura exige")
print("cuidado. Na média de uma normal, os quatro ficam juntos em 92-93%: o")
print("déficit vem do bootstrap substituir a distribuição t pela normal, e")
print("desaparece com n maior. Na variância com n = 25, TODOS falham, entre")
print("87% e 89% - a distribuição amostral da variância é assimétrica demais")
print("para n tão pequeno. Na mediana de uma log-normal, percentil e BCa")
print("acertam (95,2%) e o BÁSICO desaba para 84,9%, porque ele reflete a")
print("distribuição bootstrap em torno da estimativa e a assimetria se inverte.")
print("O BCa é o que mais vezes fica próximo do nominal, e é por isso que é o")
print("padrão recomendado - mas ele custa n avaliações extras por intervalo.")

# ===================================================================
# PARTE 4 - o bootstrap aplicado ao erro de predição
# ===================================================================
print("\n" + "=" * 70)
print("O bootstrap também estima o erro de predição - mas a versão ingênua")
print("é fortemente otimista, e é preciso corrigi-la.")
Xd = monta(Xv, 2, 2)
b_full = ajusta(Xd, y)
res_sub = eqm(y, Xd @ b_full)
rng = np.random.default_rng(21)
B = 1000
err_ing, err_632 = [], []
oob_soma = np.zeros(n); oob_cont = np.zeros(n)
for _ in range(B):
    ii = rng.integers(0, n, n)
    bb = ajusta(Xd[ii], y[ii])
    err_ing.append(eqm(y, Xd @ bb))                      # testa em TODOS
    fora = np.setdiff1d(np.arange(n), ii)
    oob_soma[fora] += (y[fora] - Xd[fora] @ bb)**2
    oob_cont[fora] += 1
oob = float(np.mean(oob_soma[oob_cont > 0]/oob_cont[oob_cont > 0]))
e632 = 0.368*res_sub + 0.632*oob
print(f"\n  EQM de resubstituição                     = {res_sub:.4f}")
print(f"  bootstrap INGÊNUO (testa em todos)        = {np.mean(err_ing):.4f}")
print(f"  bootstrap fora-da-bolsa (out-of-bag)      = {oob:.4f}")
print(f"  estimador 0,632 = 0,368·resub + 0,632·oob = {e632:.4f}")
print(f"  validação cruzada deixa-um-de-fora        = 0.7854  (ficha loocv)")
print(f"\n  proporção média de observações fora da bolsa = "
      f"{np.mean(oob_cont)/B:.4f}")
print(f"  valor teórico (1 − 1/n)^n → 1/e = {1/np.e:.4f}")
print("\nO bootstrap ingênuo subestima porque a maioria das observações de")
print("teste também está na reamostra de treino. O fora-da-bolsa corrige")
print("isso, mas passa a SUPERESTIMAR, porque cada modelo é ajustado com")
print("cerca de 63% das observações distintas. O estimador 0,632 de Efron")
print("combina os dois extremos e fica próximo da validação cruzada.")

# ===================================================================
# PARTE 5 - onde o bootstrap FALHA
# ===================================================================
print("\n" + "=" * 70)
print("O bootstrap não é universal. Três situações em que ele falha:")
print("\n(a) ESTATÍSTICAS DE EXTREMOS. O máximo da amostra.")
rng = np.random.default_rng(31)
cob = 0; reps = 3000
for _ in range(reps):
    d = rng.uniform(0, 10, 40).reshape(-1, 1)
    t0 = d.max(); bsv = boot(d, lambda z: float(z.max()), 600,
                             int(rng.integers(1e6)))
    lo, hi = ic_percentil(bsv)
    cob += (lo <= 10.0 <= hi)
print(f"    cobertura do IC percentil 95% para o máximo = "
      f"{100*cob/reps:.1f}%  (deveria ser 95%)")
print("    O máximo da reamostra nunca excede o da amostra, de modo que a")
print("    distribuição bootstrap é truncada à direita e o verdadeiro fica")
print("    quase sempre fora. Nenhum B resolve.")

print("\n(b) DADOS DEPENDENTES. Reamostrar observações de uma série temporal")
print("    destrói a autocorrelação.")
rng = np.random.default_rng(41)
cob_i, cob_b = 0, 0
for _ in range(2000):
    e = rng.normal(0, 1, 200); s = np.empty(200); s[0] = e[0]
    for i in range(1, 200): s[i] = 0.7*s[i-1] + e[i]     # AR(1)
    t0 = s.mean()
    bsv = boot(s.reshape(-1,1), lambda z: float(z.mean()), 400,
               int(rng.integers(1e6)))
    lo, hi = ic_percentil(bsv); cob_i += (lo <= 0 <= hi)
    # bootstrap em BLOCOS de 20
    bl = s.reshape(10, 20)
    bb = np.array([bl[rng.integers(0,10,10)].ravel().mean() for _ in range(400)])
    lo, hi = np.percentile(bb, [2.5, 97.5]); cob_b += (lo <= 0 <= hi)
print(f"    cobertura reamostrando OBSERVAÇÕES = {100*cob_i/2000:.1f}%")
print(f"    cobertura reamostrando BLOCOS de 20 = {100*cob_b/2000:.1f}%")
print("    O bootstrap comum supõe independência; com séries temporais é")
print("    preciso reamostrar blocos, que preservam a estrutura local.")

print("\n(c) AMOSTRAS MUITO PEQUENAS. Com n = 5, há no máximo 126 reamostras")
print("    distintas, e a distribuição bootstrap é grosseiramente discreta.")
from math import comb
for nn in [5, 10, 20, 50]:
    print(f"    n = {nn:>2}: {comb(2*nn-1, nn):>14,} reamostras distintas"
          .replace(',', '.'))
print("    Com n pequeno o bootstrap não cria informação que os dados não têm.")
