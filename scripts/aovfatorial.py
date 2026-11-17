# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados: warpbreaks (R). Número de rupturas do fio em teares, segundo o tipo de
# lã (A, B) e o nível de tensão do fio (L = baixa, M = média, H = alta).
# Fonte original: Tippett (1950). Delineamento 2 x 3 com 9 repetições por célula.
celulas = {
    ("A", "L"): [26, 30, 54, 25, 70, 52, 51, 26, 67],
    ("A", "M"): [18, 21, 29, 17, 12, 18, 35, 30, 36],
    ("A", "H"): [36, 21, 24, 18, 10, 43, 28, 15, 26],
    ("B", "L"): [27, 14, 29, 19, 29, 31, 41, 20, 44],
    ("B", "M"): [42, 26, 19, 16, 39, 28, 21, 39, 29],
    ("B", "H"): [20, 21, 24, 17, 13, 15, 15, 16, 28],
}
las = ["A", "B"]
tensoes = ["L", "M", "H"]
a, b = len(las), len(tensoes)
r = 9                                     # repetições por célula
N = a * b * r
y = {kv: np.array(v, dtype=float) for kv, v in celulas.items()}
todos = np.concatenate([y[(i, j)] for i in las for j in tensoes])
M = np.array([[y[(i, j)].mean() for j in tensoes] for i in las])    # médias das células

print("Médias das células (número de rupturas)")
print("lã      L       M       H     média da lã")
for ii, i in enumerate(las):
    print(f" {i}   {M[ii,0]:6.3f}  {M[ii,1]:6.3f}  {M[ii,2]:6.3f}     {M[ii].mean():6.3f}")
print(f"média   {M[:,0].mean():6.3f}  {M[:,1].mean():6.3f}  {M[:,2].mean():6.3f}     {todos.mean():6.3f}")
print("\nDesvios-padrão das células")
for i in las:
    print(f" {i}   " + "  ".join(f"{y[(i,j)].std(ddof=1):6.3f}" for j in tensoes))

# 1. Decomposição da soma de quadrados (delineamento balanceado: as somas são ortogonais)
mg = todos.mean()
SQ_total = ((todos - mg)**2).sum()
SQ_la = b * r * ((M.mean(axis=1) - mg)**2).sum()
SQ_tensao = a * r * ((M.mean(axis=0) - mg)**2).sum()
SQ_celulas = r * ((M - mg)**2).sum()
SQ_inter = SQ_celulas - SQ_la - SQ_tensao
SQ_res = SQ_total - SQ_celulas
gl_la, gl_tensao, gl_inter, gl_res = a - 1, b - 1, (a - 1) * (b - 1), N - a * b
QM_res = SQ_res / gl_res
linhas = [("Lã", gl_la, SQ_la), ("Tensão", gl_tensao, SQ_tensao), ("Lã:Tensão", gl_inter, SQ_inter)]
print(f"\nSQ total = {SQ_total:.2f} = {SQ_la:.2f} (lã) + {SQ_tensao:.2f} (tensão) "
      f"+ {SQ_inter:.2f} (interação) + {SQ_res:.2f} (resíduo)")
print("\nFonte          GL        SQ         QM        F     p-valor")
for nome, gl, sq in linhas:
    qm = sq / gl
    Fv = qm / QM_res
    print(f"{nome:<12} {gl:3d}  {sq:8.2f}  {qm:9.3f}  {Fv:6.4f}  {stats.f.sf(Fv, gl, gl_res):.5f}")
print(f"{'Resíduo':<12} {gl_res:3d}  {SQ_res:8.2f}  {QM_res:9.3f}")
print(f"{'Total':<12} {N-1:3d}  {SQ_total:8.2f}")
print(f"DP do resíduo = {np.sqrt(QM_res):.3f} rupturas")

# 2. Tamanho de efeito
print("\nefeito        eta² parcial   omega² parcial")
for nome, gl, sq in linhas:
    e2p = sq / (sq + SQ_res)
    w2p = (sq - gl * QM_res) / (sq + (N - gl) * QM_res)
    print(f"{nome:<12}    {e2p:.4f}         {w2p:.4f}")

# 3. Pressupostos
res = np.concatenate([y[(i, j)] - M[ii, jj] for ii, i in enumerate(las) for jj, j in enumerate(tensoes)])
print(f"\nShapiro-Wilk dos resíduos: W = {stats.shapiro(res)[0]:.4f}, p-valor = {stats.shapiro(res)[1]:.4f}")
gs = [y[(i, j)] for i in las for j in tensoes]
print(f"Levene nas 6 células (mediana): W = {stats.levene(*gs, center='median')[0]:.4f}, "
      f"p-valor = {stats.levene(*gs, center='median')[1]:.4f}")
vs = [g.var(ddof=1) for g in gs]
print(f"Razão entre a maior e a menor variância das células = {max(vs)/min(vs):.2f}")
print("A célula A-L tem variância muito acima das demais: a raiz quadrada estabiliza.")

# 3b. O mesmo modelo na escala da raiz quadrada, remédio usual para contagens
raiz = {kv: np.sqrt(v) for kv, v in y.items()}
Mr = np.array([[raiz[(i, j)].mean() for j in tensoes] for i in las])
tr = np.concatenate([raiz[(i, j)] for i in las for j in tensoes])
mgr = tr.mean()
SQt_r = ((tr - mgr)**2).sum()
SQ_la_r = b * r * ((Mr.mean(axis=1) - mgr)**2).sum()
SQ_te_r = a * r * ((Mr.mean(axis=0) - mgr)**2).sum()
SQ_cel_r = r * ((Mr - mgr)**2).sum()
SQ_in_r = SQ_cel_r - SQ_la_r - SQ_te_r
QM_res_r = (SQt_r - SQ_cel_r) / gl_res
gsr = [raiz[(i, j)] for i in las for j in tensoes]
print(f"  Levene na escala da raiz: p-valor = {stats.levene(*gsr, center='median')[1]:.4f}")
vr = [g.var(ddof=1) for g in gsr]
print(f"  razão entre variâncias na escala da raiz = {max(vr)/min(vr):.2f}")
for nome, gl, sq in [("Lã", gl_la, SQ_la_r), ("Tensão", gl_tensao, SQ_te_r), ("Lã:Tensão", gl_inter, SQ_in_r)]:
    Fv = (sq / gl) / QM_res_r
    print(f"  {nome:<10} F({gl}; {gl_res}) = {Fv:6.4f} | p-valor = {stats.f.sf(Fv, gl, gl_res):.5f}")

# 4. Como a interação se manifesta: efeitos simples da tensão dentro de cada lã
print("\nEfeitos simples da tensão, dentro de cada lã")
for ii, i in enumerate(las):
    SQ_s = r * ((M[ii] - M[ii].mean())**2).sum()
    Fv = (SQ_s / (b - 1)) / QM_res
    print(f"  lã {i}: SQ = {SQ_s:8.3f} | F({b-1}; {gl_res}) = {Fv:7.4f} | p-valor = {stats.f.sf(Fv, b-1, gl_res):.3e}")
print("\nEfeitos simples da lã, dentro de cada nível de tensão")
for jj, j in enumerate(tensoes):
    dif = M[1, jj] - M[0, jj]
    ep = np.sqrt(2 * QM_res / r)
    t = dif / ep
    print(f"  tensão {j}: B - A = {dif:+7.3f} | EP = {ep:.3f} | t({gl_res}) = {t:+.4f} | "
          f"p-valor = {2*stats.t.sf(abs(t), gl_res):.4f}")

# 5. Tukey HSD nas seis células
print("\nTukey HSD entre as seis combinações lã x tensão")
rot = [f"{i}{j}" for i in las for j in tensoes]
mc = np.array([y[(i, j)].mean() for i in las for j in tensoes])
q95 = stats.studentized_range.ppf(0.95, len(rot), gl_res)
ep = np.sqrt(QM_res / r)
print(f"q crítico (5%, 6 médias, gl = {gl_res}) = {q95:.4f} | DMS = {q95*ep:.3f} rupturas")
for u, v in itertools.combinations(range(len(rot)), 2):
    dif = mc[v] - mc[u]
    pa = stats.studentized_range.sf(abs(dif) / ep, len(rot), gl_res)
    if pa < 0.05:
        print(f"  {rot[v]} - {rot[u]}: {dif:+7.3f} | p ajustado = {pa:.4f}  *")

# 6. O modelo só com efeitos principais, para comparação
SQ_res_ad = SQ_res + SQ_inter
gl_res_ad = gl_res + gl_inter
QM_res_ad = SQ_res_ad / gl_res_ad
print("\nModelo aditivo (sem interação), para comparação")
for nome, gl, sq in linhas[:2]:
    Fv = (sq / gl) / QM_res_ad
    print(f"  {nome:<10} F({gl}; {gl_res_ad}) = {Fv:6.4f} | p-valor = {stats.f.sf(Fv, gl, gl_res_ad):.5f}")
print(f"  o efeito da lã, que era p = {stats.f.sf((SQ_la/gl_la)/QM_res, gl_la, gl_res):.4f} no modelo completo,")
print(f"  passa a p = {stats.f.sf((SQ_la/gl_la)/QM_res_ad, gl_la, gl_res_ad):.4f} quando a interação é jogada no resíduo.")
