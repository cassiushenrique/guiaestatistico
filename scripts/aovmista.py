# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados fictícios: distância no teste de caminhada de seis minutos (m) de 20
# participantes alocados ao acaso em dois protocolos de treinamento, medidos
# no início, com 4 semanas e com 8 semanas.
# Fator ENTRE participantes: protocolo (A = convencional, B = intervalado).
# Fator DENTRO de participantes: momento (3 medidas na mesma pessoa).
A = np.array([[430, 441, 473], [457, 420, 442], [427, 438, 457], [387, 428, 393],
              [440, 489, 479], [442, 440, 434], [407, 421, 413], [427, 450, 448],
              [429, 447, 456], [421, 452, 466]], dtype=float)
B = np.array([[423, 451, 507], [426, 490, 496], [410, 441, 458], [409, 458, 495],
              [391, 433, 485], [428, 482, 523], [396, 467, 514], [406, 442, 498],
              [400, 453, 469], [390, 453, 483]], dtype=float)
dados = np.vstack([A, B])
protocolo = np.repeat([0, 1], len(A))
rot_prot = ["A (convencional)", "B (intervalado)"]
momentos = ["início", "4 semanas", "8 semanas"]
a, k = 2, 3
n = len(A)                      # participantes por protocolo
N = a * n

print(f"{N} participantes ({n} por protocolo) x {k} momentos")
print("protocolo            início   4 sem   8 sem   média")
cel = np.array([[dados[protocolo == i][:, j].mean() for j in range(k)] for i in range(a)])
for i in range(a):
    print(f"{rot_prot[i]:<18}  {cel[i,0]:7.2f} {cel[i,1]:7.2f} {cel[i,2]:7.2f}  {cel[i].mean():7.2f}")
print(f"{'média do momento':<18}  {dados[:,0].mean():7.2f} {dados[:,1].mean():7.2f} "
      f"{dados[:,2].mean():7.2f}  {dados.mean():7.2f}")
print("\nDesvios-padrão das células")
for i in range(a):
    print(f"{rot_prot[i]:<18}  " + " ".join(f"{dados[protocolo==i][:,j].std(ddof=1):7.2f}" for j in range(k)))
print("\nGanho individual (8 semanas - início):")
for i in range(a):
    gan = dados[protocolo == i][:, 2] - dados[protocolo == i][:, 0]
    print(f"  {rot_prot[i]:<18} média = {gan.mean():+7.2f} m | DP = {gan.std(ddof=1):6.2f} | valores: {gan.astype(int)}")

# 1. Decomposição da soma de quadrados em duas partes
mg = dados.mean()
m_prot = np.array([dados[protocolo == i].mean() for i in range(a)])
m_mom = dados.mean(axis=0)
SQ_total = ((dados - mg)**2).sum()
# parte ENTRE participantes
SQ_prot = k * n * ((m_prot - mg)**2).sum()
SQ_sujdentro = k * sum(((dados[protocolo == i].mean(axis=1) - m_prot[i])**2).sum() for i in range(a))
# parte DENTRO de participantes
SQ_mom = N * ((m_mom - mg)**2).sum()
SQ_inter = n * ((cel - m_prot[:, None] - m_mom[None, :] + mg)**2).sum()
SQ_erro_d = SQ_total - SQ_prot - SQ_sujdentro - SQ_mom - SQ_inter
gl_prot, gl_sd = a - 1, a * (n - 1)
gl_mom, gl_int, gl_ed = k - 1, (a - 1) * (k - 1), a * (n - 1) * (k - 1)
QM_sd, QM_ed = SQ_sujdentro / gl_sd, SQ_erro_d / gl_ed
F_prot = (SQ_prot / gl_prot) / QM_sd
F_mom = (SQ_mom / gl_mom) / QM_ed
F_int = (SQ_inter / gl_int) / QM_ed
p_prot = stats.f.sf(F_prot, gl_prot, gl_sd)
p_mom = stats.f.sf(F_mom, gl_mom, gl_ed)
p_int = stats.f.sf(F_int, gl_int, gl_ed)
print(f"\nSQ total = {SQ_total:.2f}")
print(f"  parte entre participantes: {SQ_prot:.2f} (protocolo) + {SQ_sujdentro:.2f} (participantes no protocolo)")
print(f"  parte dentro:              {SQ_mom:.2f} (momento) + {SQ_inter:.2f} (interação) + {SQ_erro_d:.2f} (erro)")
print("\nFonte                        GL        SQ          QM        F      p-valor")
print("ENTRE participantes")
print(f"  Protocolo                {gl_prot:3d}  {SQ_prot:9.2f}  {SQ_prot/gl_prot:9.2f}  {F_prot:7.4f}  {p_prot:.5f}")
print(f"  Participante(Protocolo)  {gl_sd:3d}  {SQ_sujdentro:9.2f}  {QM_sd:9.2f}   <- erro deste teste")
print("DENTRO de participantes")
print(f"  Momento                  {gl_mom:3d}  {SQ_mom:9.2f}  {SQ_mom/gl_mom:9.2f}  {F_mom:7.4f}  {p_mom:.3e}")
print(f"  Protocolo:Momento        {gl_int:3d}  {SQ_inter:9.2f}  {SQ_inter/gl_int:9.2f}  {F_int:7.4f}  {p_int:.3e}")
print(f"  Erro                     {gl_ed:3d}  {SQ_erro_d:9.2f}  {QM_ed:9.2f}   <- erro dos dois testes acima")
print(f"Total                      {N*k-1:3d}  {SQ_total:9.2f}")
print("\nSão DOIS termos de erro: cada efeito é comparado ao erro do seu nível.")

# 2. Esfericidade (matriz de covariâncias agrupada dentro dos protocolos)
Sp = sum((n - 1) * np.cov(dados[protocolo == i], rowvar=False) for i in range(a)) / (a * (n - 1))
C = np.array([[1, -1, 0], [1, 1, -2]], dtype=float)
C = C / np.sqrt((C**2).sum(axis=1, keepdims=True))
Sc = C @ Sp @ C.T
pp = k - 1
W_m = np.linalg.det(Sc) / (np.trace(Sc) / pp)**pp
f_corr = 1 - (2*pp**2 + pp + 2) / (6 * pp * gl_sd)
chi2_m = -gl_sd * f_corr * np.log(W_m)
p_m = stats.chi2.sf(chi2_m, pp*(pp+1)/2 - 1)
ev = np.linalg.eigvalsh(Sc)
gg = ev.sum()**2 / (pp * (ev**2).sum())
hf = min(1.0, (gl_sd * pp * gg - 2) / (pp * (gl_sd - pp * gg)))
print(f"\nMauchly: W = {W_m:.5f}, qui-quadrado = {chi2_m:.4f}, gl = 2, p-valor = {p_m:.4f}")
print(f"Epsilon de Greenhouse-Geisser = {gg:.4f} | Huynh-Feldt = {hf:.4f}")
print("Valores-p corrigidos (a esfericidade está atendida: as correções quase não alteram nada)")
for nome, Fv, gl1 in [("Momento", F_mom, gl_mom), ("Protocolo:Momento", F_int, gl_int)]:
    print(f"  {nome:<18} sem correção: {stats.f.sf(Fv, gl1, gl_ed):.3e} | "
          f"GG: {stats.f.sf(Fv, gg*gl1, gg*gl_ed):.3e} | HF: {stats.f.sf(Fv, hf*gl1, hf*gl_ed):.3e}")

# 3. Pressupostos restantes
res = np.empty_like(dados)
for i in range(a):
    bl = dados[protocolo == i]
    res[protocolo == i] = bl - bl.mean(axis=1, keepdims=True) - bl.mean(axis=0, keepdims=True) + bl.mean()
print(f"\nShapiro-Wilk dos resíduos: W = {stats.shapiro(res.ravel())[0]:.4f}, p-valor = {stats.shapiro(res.ravel())[1]:.4f}")
md = [dados[protocolo == i].mean(axis=1) for i in range(a)]
print(f"Levene nas médias dos participantes (homogeneidade entre protocolos): "
      f"p-valor = {stats.levene(*md, center='median')[1]:.4f}")
for j, mm in enumerate(momentos):
    gs = [dados[protocolo == i][:, j] for i in range(a)]
    print(f"  Levene em {mm:<10}: p-valor = {stats.levene(*gs, center='median')[1]:.4f}")

# 4. Tamanho de efeito
print(f"\neta² parcial do protocolo         = {SQ_prot/(SQ_prot+SQ_sujdentro):.4f}")
print(f"eta² parcial do momento           = {SQ_mom/(SQ_mom+SQ_erro_d):.4f}")
print(f"eta² parcial da interação         = {SQ_inter/(SQ_inter+SQ_erro_d):.4f}")
print(f"eta² generalizado da interação    = {SQ_inter/(SQ_inter+SQ_sujdentro+SQ_erro_d):.4f}")

# 5. Efeitos simples: a interação significativa obriga a olhar dentro de cada nível
print("\nEfeito do momento DENTRO de cada protocolo")
for i in range(a):
    bl = dados[protocolo == i]
    SQ_s = n * ((bl.mean(axis=0) - bl.mean())**2).sum()
    SQ_e = ((bl - bl.mean(axis=1, keepdims=True) - bl.mean(axis=0, keepdims=True) + bl.mean())**2).sum()
    Fv = (SQ_s / (k-1)) / (SQ_e / ((n-1)*(k-1)))
    print(f"  {rot_prot[i]:<18} F({k-1}; {(n-1)*(k-1)}) = {Fv:7.3f} | "
          f"p-valor = {stats.f.sf(Fv, k-1, (n-1)*(k-1)):.3e}")
print("\nDiferença entre protocolos EM CADA momento (teste t de Welch)")
for j, mm in enumerate(momentos):
    x, yy = dados[protocolo == 0][:, j], dados[protocolo == 1][:, j]
    t, pv = stats.ttest_ind(yy, x, equal_var=False)
    s_p = np.sqrt(((n-1)*x.var(ddof=1) + (n-1)*yy.var(ddof=1)) / (2*n-2))
    print(f"  {mm:<10} B - A = {yy.mean()-x.mean():+7.2f} m | t = {t:+.4f} | "
          f"p-valor = {pv:.4f} | d de Cohen = {(yy.mean()-x.mean())/s_p:+.3f}")

# 6. Comparações pareadas dentro de cada protocolo, com ajuste de Holm
print("\nComparações pareadas dentro de cada protocolo (t pareado, ajuste de Holm)")
for i in range(a):
    bl = dados[protocolo == i]
    pares = list(itertools.combinations(range(k), 2))
    pv = [stats.ttest_rel(bl[:, jj], bl[:, ii])[1] for ii, jj in pares]
    ordem = np.argsort(pv); holm = np.empty(len(pv)); acum = 0.0
    for pos, idx in enumerate(ordem):
        acum = max(acum, (len(pv) - pos) * pv[idx]); holm[idx] = min(1.0, acum)
    print(f"  {rot_prot[i]}")
    for m, (ii, jj) in enumerate(pares):
        d = bl[:, jj] - bl[:, ii]
        print(f"    {momentos[jj]} - {momentos[ii]:<10} = {d.mean():+7.2f} m | "
              f"p bruto = {pv[m]:.3e} | Holm = {holm[m]:.3e}")

# 7. O que aconteceria ignorando a estrutura de medidas repetidas
Fi, pi = stats.f_oneway(*[dados[:, j] for j in range(k)])
print(f"\nIgnorando o pareamento, o efeito do momento daria F = {Fi:.3f} "
      f"(gl = {k-1}; {N*k-k}), p-valor = {pi:.3e}")
print(f"contra F = {F_mom:.3f} (gl = {gl_mom}; {gl_ed}) no modelo misto: a variabilidade")
print(f"entre participantes ({100*SQ_sujdentro/SQ_total:.1f}% do total) deixaria de ser isolada.")
