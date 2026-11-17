# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: chickwts (R). Peso (g) de 71 pintinhos aos 6 dias, segundo seis rações.
# É o mesmo conjunto das fichas de Bonferroni, Šidák, Tukey e REGW.
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
ordem = np.argsort(-medias)

print("ração        n    média     DP")
for i in ordem:
    print(f"{nomes[i]:<12}{int(ns[i]):2d}  {medias[i]:7.3f}  {np.sqrt(vars_[i]):6.2f}")
print(f"\nANOVA: F({k-1}; {gl}) = {stats.f_oneway(*gs)[0]:.4f}, p = {stats.f_oneway(*gs)[1]:.3e}")
print(f"QM do resíduo = {QM:.4f} | gl = {gl}")

# 1. O nível de proteção de Duncan
print("\nO nível que Duncan adota para um subconjunto de p médias consecutivas")
print(" p   nível de proteção   alfa_p = 1 - 0,95^(p-1)   q de Duncan   q de Tukey   q do REGW")
for p in range(2, k + 1):
    prot = 0.95**(p - 1)
    ap = 1 - prot
    ar = alfa if p >= k - 1 else 1 - (1 - alfa)**(p / k)
    print(f" {p}       {prot:.6f}          {ap:.6f}            "
          f"{stats.studentized_range.ppf(prot, p, gl):8.4f}     "
          f"{stats.studentized_range.ppf(1-alfa, k, gl):8.4f}   "
          f"{stats.studentized_range.ppf(1-ar, p, gl):8.4f}")
print("Duncan AUMENTA o nível conforme o subconjunto cresce; o REGW faz o contrário.")
print("É essa escolha que faz Duncan perder o controle da taxa de erro do conjunto.")

# 2. Amplitudes mínimas significativas
print("\nAmplitudes mínimas significativas (grupos balanceados em n = 12, para referência)")
print(" p   q de Duncan   AMS (g)")
for p in range(2, k + 1):
    q = stats.studentized_range.ppf(0.95**(p-1), p, gl)
    print(f" {p}     {q:8.4f}     {q*np.sqrt(QM/12):7.3f}")

# 3. O procedimento descendente
def passo_descendente(qcrit):
    difere = np.zeros((k, k), bool); homogeneos = []; passos = []
    for p in range(k, 1, -1):
        for a in range(0, k - p + 1):
            bloco = ordem[a:a + p]; cj = frozenset(bloco)
            if any(cj <= h for h in homogeneos):
                passos.append((p, [nomes[i] for i in bloco], None, None, "não testado")); continue
            i, j = bloco[0], bloco[-1]
            ep = np.sqrt(QM / 2 * (1/ns[i] + 1/ns[j]))
            q = abs(medias[i] - medias[j]) / ep
            c = qcrit(p)
            if q > c:
                difere[i, j] = difere[j, i] = True
                passos.append((p, [nomes[i] for i in bloco], q, c, "heterogêneo"))
            else:
                homogeneos.append(cj)
                passos.append((p, [nomes[i] for i in bloco], q, c, "HOMOGÊNEO: para aqui"))
    return homogeneos, passos

print("\nTeste de Duncan, passo a passo")
hom, passos = passo_descendente(lambda p: stats.studentized_range.ppf(0.95**(p-1), p, gl))
print(f"{'p':>2}  {'subconjunto (do maior ao menor)':<52}{'q':>8}{'crítico':>9}  situação")
for p, bl, q, c, sit in passos:
    bls = " > ".join(bl)
    print(f"{p:>2}  {bls:<52}" + (f"{'':>8}{'':>9}" if q is None else f"{q:8.4f}{c:9.4f}") + f"  {sit}")
nd = np.zeros((k, k), bool)
for h in hom:
    for x, y in itertools.combinations(h, 2): nd[x, y] = nd[y, x] = True
pares = [(nomes[a], nomes[b]) for a, b in itertools.combinations(range(k), 2) if not nd[a, b]]
print(f"\nPares declarados diferentes por Duncan: {len(pares)}")
print("  " + "; ".join(f"{b} - {a}" for a, b in pares))

# 4. Comparação com os demais
tuk = [(a, b) for a, b in itertools.combinations(range(k), 2)
       if stats.studentized_range.sf(abs(medias[b]-medias[a])/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl) < alfa]
print(f"\nQuais pares Duncan acrescenta ao que o Tukey já encontra:")
for a, b in itertools.combinations(range(k), 2):
    if not nd[a, b] and (a, b) not in tuk:
        pt = stats.studentized_range.sf(abs(medias[b]-medias[a])/np.sqrt(QM/2*(1/ns[a]+1/ns[b])), k, gl)
        pb = min(1, 15 * 2*stats.t.sf(abs(medias[b]-medias[a])/np.sqrt(QM*(1/ns[a]+1/ns[b])), gl))
        print(f"  {nomes[b]} - {nomes[a]}: {medias[b]-medias[a]:+7.2f} g | "
              f"p de Tukey = {pt:.4f} | p de Bonferroni = {pb:.4f}")
print("São exatamente as quatro comparações que os métodos que controlam a taxa de erro")
print("do conjunto classificam como achados compatíveis com o acaso.")

# 5. Letras
print("\nApresentação em letras (Duncan)")
cliques = []
for r in range(k, 0, -1):
    for comb in itertools.combinations(ordem, r):
        if all(nd[x, y] for x, y in itertools.combinations(comb, 2)):
            cj = set(comb)
            if not any(cj <= c for c in cliques): cliques.append(cj)
letras = [""] * k
for idx, c in enumerate(cliques):
    for j in c: letras[j] += chr(ord("a") + idx)
print("ração          média   letras")
for i in ordem:
    print(f"{nomes[i]:<12}{medias[i]:9.2f}   {''.join(sorted(letras[i]))}")

# 6. A simulação que mostra o problema
print("\nSimulação: 10.000 experimentos com médias IGUAIS, para 3, 4, 5 e 6 grupos (n = 12)")
rng = np.random.default_rng(321)
M = 10000
print(" k grupos    LSD    Duncan   REGWQ   Tukey   (alvo: 5%)")
for kk in (3, 4, 5, 6):
    glr = kk * 11
    q_t = stats.studentized_range.ppf(1 - alfa, kk, glr)
    t_l = stats.t.ppf(1 - alfa/2, glr)
    q_d = {p: stats.studentized_range.ppf(0.95**(p-1), p, glr) for p in range(2, kk+1)}
    q_r = {p: stats.studentized_range.ppf(1 - (alfa if p >= kk-1 else 1-(1-alfa)**(p/kk)), p, glr)
           for p in range(2, kk+1)}
    G = rng.normal(0, 1, (M, kk, 12))
    MM = G.mean(axis=2); EP = np.sqrt(G.var(axis=2, ddof=1).mean(axis=1) / 12)
    c = dict(lsd=0, dun=0, reg=0, tuk=0)
    iu = np.triu_indices(kk, 1)
    for e in range(M):
        mm = MM[e]; ep = EP[e]; om = np.argsort(-mm)
        dif = np.abs(mm[:, None] - mm[None, :]) / ep
        if (dif[iu] / np.sqrt(2) > t_l).any(): c['lsd'] += 1
        if (dif[iu] > q_t).any(): c['tuk'] += 1
        for chave, qc in (('dun', q_d), ('reg', q_r)):
            achou = False; hm = []
            for p in range(kk, 1, -1):
                for a in range(0, kk - p + 1):
                    bl = om[a:a+p]; cj = frozenset(bl)
                    if any(cj <= h for h in hm): continue
                    if dif[bl[0], bl[-1]] > qc[p]: achou = True
                    else: hm.append(cj)
            if achou: c[chave] += 1
    print(f"   {kk:3d}      {100*c['lsd']/M:5.2f}%  {100*c['dun']/M:5.2f}%  "
          f"{100*c['reg']/M:5.2f}%  {100*c['tuk']/M:5.2f}%")
print("Duncan só controla a taxa de erro quando k = 2. A partir daí ela cresce,")
print("e é por isso que o procedimento é considerado liberal.")
