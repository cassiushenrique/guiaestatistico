# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: chickwts (R). Peso (g) de 71 pintinhos aos 6 dias, segundo seis rações.
# É o mesmo conjunto das fichas de Bonferroni, Šidák, Tukey e Duncan.
dados = {
 "horsebean": [179,160,136,227,217,168,108,124,143,140],
 "linseed":   [309,229,181,141,260,203,148,169,213,257,244,271],
 "soybean":   [243,230,248,327,329,250,193,271,316,267,199,171,158,248],
 "sunflower": [423,340,392,339,341,226,320,295,334,322,297,318],
 "meatmeal":  [325,257,303,315,380,153,263,242,206,344,258],
 "casein":    [368,390,379,260,404,318,352,359,216,222,283,332]}
nomes = list(dados)
gs = [np.array(v, float) for v in dados.values()]
k = len(gs); ns = np.array([len(g) for g in gs], float); N = int(ns.sum())
medias = np.array([g.mean() for g in gs]); vars_ = np.array([g.var(ddof=1) for g in gs])
QM = np.sum((ns - 1) * vars_) / (N - k); gl = N - k
alfa = 0.05
ordem = np.argsort(-medias)                      # do maior para o menor

print("ração        n    média     DP")
for i in ordem:
    print(f"{nomes[i]:<12}{int(ns[i]):2d}  {medias[i]:7.3f}  {np.sqrt(vars_[i]):6.2f}")
print(f"\nANOVA: F({k-1}; {gl}) = {stats.f_oneway(*gs)[0]:.4f}, p = {stats.f_oneway(*gs)[1]:.3e}")
print(f"QM do resíduo = {QM:.4f} | gl = {gl}")

# 1. O nível de significância que o REGW usa em cada etapa
def alfa_regw(p, k, alfa=0.05):
    return alfa if p >= k - 1 else 1 - (1 - alfa)**(p / k)
print("\nNíveis de significância por etapa (p = número de médias no subconjunto)")
print(" p   alfa_p (REGW)   q crítico   F crítico   q de Tukey (referência)")
for p in range(2, k + 1):
    ap = alfa_regw(p, k)
    print(f" {p}    {ap:.6f}     {stats.studentized_range.ppf(1-ap, p, gl):8.4f}    "
          f"{stats.f.ppf(1-ap, p-1, gl):8.4f}   {stats.studentized_range.ppf(1-alfa, k, gl):10.4f}")
print("Para p = k e p = k-1 o nível é o próprio alfa; abaixo disso ele é reduzido,")
print("de modo que a taxa de erro do conjunto permanece controlada.")

# 2. O procedimento descendente
def passo_descendente(estatistica):
    """Testa subconjuntos de médias consecutivas, do maior para o menor.
    Um subconjunto só é testado se nenhum subconjunto que o contém foi declarado homogêneo."""
    difere = np.zeros((k, k), bool)
    homogeneos = []
    passos = []
    for p in range(k, 1, -1):
        for a in range(0, k - p + 1):
            bloco = ordem[a:a + p]
            cj = frozenset(bloco)
            if any(cj <= h for h in homogeneos):
                passos.append((p, [nomes[i] for i in bloco], None, None, "não testado"))
                continue
            est, crit = estatistica(bloco, p)
            if est > crit:
                difere[bloco[0], bloco[-1]] = difere[bloco[-1], bloco[0]] = True
                passos.append((p, [nomes[i] for i in bloco], est, crit, "heterogêneo"))
            else:
                homogeneos.append(cj)
                passos.append((p, [nomes[i] for i in bloco], est, crit, "HOMOGÊNEO: para aqui"))
    return difere, homogeneos, passos

def est_q(bloco, p):
    i, j = bloco[0], bloco[-1]
    ep = np.sqrt(QM / 2 * (1/ns[i] + 1/ns[j]))
    return abs(medias[i] - medias[j]) / ep, stats.studentized_range.ppf(1 - alfa_regw(p, k), p, gl)

def est_F(bloco, p):
    mb = np.sum(ns[bloco] * medias[bloco]) / np.sum(ns[bloco])
    SQ = np.sum(ns[bloco] * (medias[bloco] - mb)**2)
    return (SQ / (p - 1)) / QM, stats.f.ppf(1 - alfa_regw(p, k), p - 1, gl)

for titulo, est in [("REGWQ (amplitude estudentizada)", est_q), ("REGWF (estatística F)", est_F)]:
    print(f"\n{titulo}")
    dif, hom, passos = passo_descendente(est)
    print(f"{'p':>2}  {'subconjunto (do maior ao menor)':<52}{'est.':>8}{'crítico':>9}  situação")
    for p, bl, e, c, sit in passos:
        bls = " > ".join(bl)
        if e is None:
            print(f"{p:>2}  {bls:<52}{'':>8}{'':>9}  {sit}")
        else:
            print(f"{p:>2}  {bls:<52}{e:8.4f}{c:9.4f}  {sit}")
    # conjuntos homogêneos maximais -> pares que NÃO diferem
    naodifere = np.zeros((k, k), bool)
    for h in hom:
        for x, y in itertools.combinations(h, 2):
            naodifere[x, y] = naodifere[y, x] = True
    pares = [(nomes[a], nomes[b]) for a, b in itertools.combinations(range(k), 2) if not naodifere[a, b]]
    print(f"Pares que diferem: {len(pares)}")
    print("  " + "; ".join(f"{b} - {a}" for a, b in pares))

# 3. Comparação com os demais procedimentos
print("\nComparação dos procedimentos (número de pares declarados diferentes, de 15)")
tuk = [(a, b) for a, b in itertools.combinations(range(k), 2)
       if stats.studentized_range.sf(abs(medias[b]-medias[a])/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl) < alfa]
lsd = [(a, b) for a, b in itertools.combinations(range(k), 2)
       if 2*stats.t.sf(abs(medias[b]-medias[a])/np.sqrt(QM*(1/ns[a]+1/ns[b])), gl) < alfa]
dif_q, hom_q, _ = passo_descendente(est_q)
dif_f, hom_f, _ = passo_descendente(est_F)
def conta(hom):
    nd = np.zeros((k, k), bool)
    for h in hom:
        for x, y in itertools.combinations(h, 2): nd[x, y] = nd[y, x] = True
    return sum(1 for a, b in itertools.combinations(range(k), 2) if not nd[a, b])
print(f"  LSD (sem ajuste): {len(lsd)}")
print(f"  Duncan:           12   (ver a ficha do teste de Duncan)")
print(f"  REGWQ:            {conta(hom_q)}")
print(f"  REGWF:            {conta(hom_f)}")
print(f"  Tukey:            {len(tuk)}")
print("  Bonferroni:       8")

# 4. O que o REGW encontra e o Tukey não
nd = np.zeros((k, k), bool)
for h in hom_q:
    for x, y in itertools.combinations(h, 2): nd[x, y] = nd[y, x] = True
extra = [(a, b) for a, b in itertools.combinations(range(k), 2)
         if not nd[a, b] and (a, b) not in tuk]
print("\nPares em que REGWQ difere do Tukey:")
for a, b in extra:
    pt = stats.studentized_range.sf(abs(medias[b]-medias[a])/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl)
    print(f"  {nomes[b]} - {nomes[a]}: diferença = {medias[b]-medias[a]:+.2f} g | "
          f"p de Tukey = {pt:.4f} | declarado significativo pelo REGWQ")

# 5. Simulação da taxa de erro do conjunto
print("\nSimulação: 10.000 experimentos com 6 grupos de médias IGUAIS (n = 12 cada)")
rng = np.random.default_rng(99)
M = 10000
glr = 6 * 11
q_tuk = stats.studentized_range.ppf(1 - alfa, 6, glr)
t_lsd = stats.t.ppf(1 - alfa/2, glr)
q_reg = {p: stats.studentized_range.ppf(1 - alfa_regw(p, 6), p, glr) for p in range(2, 7)}
q_dun = {p: stats.studentized_range.ppf(0.95**(p-1), p, glr) for p in range(2, 7)}
cont = dict(lsd=0, duncan=0, regwq=0, tukey=0)
G = rng.normal(0, 1, (M, 6, 12))
MM = G.mean(axis=2); VV = G.var(axis=2, ddof=1).mean(axis=1)
EP = np.sqrt(VV / 12)
for e in range(M):
    mm = MM[e]; ep = EP[e]
    ordm = np.argsort(-mm)
    dif = np.abs(mm[:, None] - mm[None, :]) / ep
    iu = np.triu_indices(6, 1)
    if (dif[iu] / np.sqrt(2) > t_lsd).any(): cont['lsd'] += 1
    if (dif[iu] > q_tuk).any(): cont['tukey'] += 1
    for chave, qc in (('regwq', q_reg), ('duncan', q_dun)):
        achou = False; hom = []
        for p in range(6, 1, -1):
            for a in range(0, 7 - p):
                bl = ordm[a:a+p]; cj = frozenset(bl)
                if any(cj <= h for h in hom): continue
                if dif[bl[0], bl[-1]] > qc[p]: achou = True
                else: hom.append(cj)
        if achou: cont[chave] += 1
for nome, c in [("LSD (sem ajuste)", cont['lsd']), ("Duncan", cont['duncan']),
                ("REGWQ", cont['regwq']), ("Tukey", cont['tukey'])]:
    print(f"  {nome:<18}{100*c/M:5.2f}% dos experimentos com ao menos um falso positivo")
print("  (o alvo é 5%; Duncan fica claramente acima, REGWQ e Tukey ficam no alvo)")
