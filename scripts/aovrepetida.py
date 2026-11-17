# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
import itertools

# Dados fictícios: tempo de reação (ms) de 12 participantes em três condições de ruído.
# É o mesmo conjunto usado na ficha do teste de esfericidade de Mauchly.
dados = np.array([
    [412, 428, 471], [385, 402, 455], [440, 447, 462], [398, 421, 502],
    [455, 460, 489], [372, 395, 468], [420, 431, 444], [401, 426, 517],
    [433, 441, 470], [390, 409, 446], [447, 452, 495], [408, 430, 483]], dtype=float)
condicoes = ["Silêncio", "Ruído moderado", "Ruído intenso"]
n, k = dados.shape
print(f"{n} participantes x {k} condições")
print("condição            média     DP")
for j, cd in enumerate(condicoes):
    print(f"{cd:<18}{dados[:, j].mean():7.2f}  {dados[:, j].std(ddof=1):6.2f}")
print(f"Média geral = {dados.mean():.4f}")
print("Média de cada participante:", np.round(dados.mean(axis=1), 2))

# 1. Decomposição da soma de quadrados
media_geral = dados.mean()
m_cond = dados.mean(axis=0)          # média de cada condição
m_suj = dados.mean(axis=1)           # média de cada participante
SQ_total = ((dados - media_geral)**2).sum()
SQ_suj = k * ((m_suj - media_geral)**2).sum()         # entre participantes
SQ_cond = n * ((m_cond - media_geral)**2).sum()       # efeito da condição
SQ_erro = SQ_total - SQ_suj - SQ_cond                 # resíduo (interação suj x cond)
gl_suj, gl_cond, gl_erro = n - 1, k - 1, (n - 1) * (k - 1)
QM_cond, QM_erro = SQ_cond / gl_cond, SQ_erro / gl_erro
F = QM_cond / QM_erro
p = stats.f.sf(F, gl_cond, gl_erro)
print(f"\nSQ total = {SQ_total:.2f} = SQ participantes ({SQ_suj:.2f}) + SQ condição ({SQ_cond:.2f}) + SQ erro ({SQ_erro:.2f})")
print("\nFonte                 GL        SQ         QM        F      p-valor")
print(f"Participantes       {gl_suj:3d}  {SQ_suj:9.2f}  {SQ_suj/gl_suj:9.2f}")
print(f"Condição            {gl_cond:3d}  {SQ_cond:9.2f}  {QM_cond:9.2f}  {F:7.3f}  {p:.3e}")
print(f"Erro (suj x cond)   {gl_erro:3d}  {SQ_erro:9.2f}  {QM_erro:9.2f}")
print(f"Total               {n*k-1:3d}  {SQ_total:9.2f}")

# 2. O que se perderia ignorando o pareamento (ANOVA de um fator comum)
Fi, pi = stats.f_oneway(*[dados[:, j] for j in range(k)])
SQ_dentro_ind = SQ_suj + SQ_erro
print(f"\nSe os dados fossem tratados como independentes:")
print(f"  F = {Fi:.3f}, gl = ({k-1}; {n*k-k}), p-valor = {pi:.3e}")
print(f"  o resíduo seria {SQ_dentro_ind:.2f} em {n*k-k} gl -> QM = {SQ_dentro_ind/(n*k-k):.2f},")
print(f"  contra QM do erro = {QM_erro:.2f} no modelo de medidas repetidas.")
print(f"  A variabilidade entre participantes ({100*SQ_suj/SQ_total:.1f}% do total) iria para o erro.")

# 3. Esfericidade: Mauchly, Greenhouse-Geisser e Huynh-Feldt
C = np.array([[1, -1, 0], [1, 1, -2]], dtype=float)
C = C / np.sqrt((C**2).sum(axis=1, keepdims=True))      # contrastes ortonormais
S = np.cov(dados, rowvar=False)
Sc = C @ S @ C.T
pp = k - 1
W_m = np.linalg.det(Sc) / (np.trace(Sc) / pp)**pp
f_corr = 1 - (2*pp**2 + pp + 2) / (6 * pp * (n - 1))
chi2_m = -(n - 1) * f_corr * np.log(W_m)
gl_m = pp * (pp + 1) / 2 - 1
p_m = stats.chi2.sf(chi2_m, gl_m)
autov = np.linalg.eigvalsh(Sc)
gg = autov.sum()**2 / (pp * (autov**2).sum())
hf = min(1.0, (n * pp * gg - 2) / (pp * (n - 1 - pp * gg)))
li = 1 / pp
print(f"\nMauchly: W = {W_m:.5f}, qui-quadrado = {chi2_m:.4f}, gl = {int(gl_m)}, p-valor = {p_m:.3e}")
print(f"Epsilon de Greenhouse-Geisser = {gg:.4f} | Huynh-Feldt = {hf:.4f} | limite inferior = {li:.4f}")
print("\nValor-p sob cada correção (a estatística F não muda; mudam os graus de liberdade):")
for nome, e in [("sem correção", 1.0), ("Greenhouse-Geisser", gg), ("Huynh-Feldt", hf), ("limite inferior", li)]:
    print(f"  {nome:<20} gl = ({e*gl_cond:5.3f}; {e*gl_erro:6.3f})  p-valor = {stats.f.sf(F, e*gl_cond, e*gl_erro):.3e}")

# 4. Tamanho de efeito
eta2_p = SQ_cond / (SQ_cond + SQ_erro)
eta2_g = SQ_cond / (SQ_cond + SQ_suj + SQ_erro)
print(f"\neta² parcial = {eta2_p:.4f} | eta² generalizado = {eta2_g:.4f}")

# 5. Comparações múltiplas: testes t pareados com ajuste de Holm e de Bonferroni
print("\nComparações pareadas (teste t pareado):")
brutos = []
pares = list(itertools.combinations(range(k), 2))
for i, j in pares:
    d = dados[:, j] - dados[:, i]
    t, pv = stats.ttest_rel(dados[:, j], dados[:, i])
    dz = d.mean() / d.std(ddof=1)
    brutos.append(pv)
    print(f"  {condicoes[j]} - {condicoes[i]}: diferença = {d.mean():+7.2f} ms | "
          f"t({n-1}) = {t:+.4f} | p bruto = {pv:.3e} | d de Cohen (dz) = {dz:.3f}")
ordem = np.argsort(brutos)
holm = np.empty(len(brutos))
acum = 0.0
for pos, idx in enumerate(ordem):
    acum = max(acum, (len(brutos) - pos) * brutos[idx])
    holm[idx] = min(1.0, acum)
print("\npar                                 p bruto    Bonferroni     Holm")
for m, (i, j) in enumerate(pares):
    print(f"{condicoes[j]} - {condicoes[i]:<18} {brutos[m]:.3e}  {min(1, 3*brutos[m]):.3e}  {holm[m]:.3e}")

# 6. Alternativa não paramétrica
Q, pQ = stats.friedmanchisquare(*[dados[:, j] for j in range(k)])
print(f"\nTeste de Friedman: Q = {Q:.4f}, gl = {k-1}, p-valor = {pQ:.3e}")
print(f"W de Kendall = {Q/(n*(k-1)):.4f}")

# 7. Normalidade dos resíduos do modelo
res = dados - m_suj[:, None] - m_cond[None, :] + media_geral
print(f"\nShapiro-Wilk dos resíduos: W = {stats.shapiro(res.ravel())[0]:.4f}, p-valor = {stats.shapiro(res.ravel())[1]:.4f}")
