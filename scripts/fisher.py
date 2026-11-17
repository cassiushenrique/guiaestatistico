# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats
from math import comb
import itertools

# ===================================================================
# PARTE 1 - o experimento que deu origem ao teste: a prova do chá
# Fisher (1935) descreve uma senhora que afirmava distinguir, pelo sabor,
# se o leite fora posto antes ou depois do chá. Oito xícaras, quatro de
# cada tipo, e ela sabia que eram quatro de cada.
print("A prova do chá (Fisher, 1935): 8 xícaras, 4 com leite primeiro")
print("A senhora acertou as 4. Qual a chance disso, se ela estivesse adivinhando?")
print(f"  C(4,4) x C(4,0) / C(8,4) = {comb(4,4)*comb(4,0)}/{comb(8,4)} = {1/comb(8,4):.6f}")
print("\nDistribuição completa do número de acertos entre as 4 declaradas:")
print(" acertos   combinações   probabilidade")
for a in range(5):
    print(f"    {a}        {comb(4,a)*comb(4,4-a):4d}         {stats.hypergeom.pmf(a,8,4,4):.6f}")
print(f"  soma = {sum(stats.hypergeom.pmf(a,8,4,4) for a in range(5)):.6f}")
cha = np.array([[4, 0], [0, 4]])
print(f"\nscipy, unilateral: {stats.fisher_exact(cha, alternative='greater')[1]:.6f}")
print(f"scipy, bilateral:  {stats.fisher_exact(cha)[1]:.6f}")
print("Com 3 acertos o valor-p unilateral seria "
      f"{stats.fisher_exact(np.array([[3,1],[1,3]]), alternative='greater')[1]:.6f}: não bastaria.")
print("O delineamento foi desenhado para que SÓ o acerto completo fosse significativo.")

# ===================================================================
# PARTE 2 - onde o teste é necessário: contagens pequenas
# Dados fictícios: 23 lotes de sementes tratadas com dois fungicidas,
# registrando se houve ou não emergência de plântulas doentes.
tab = np.array([[2, 10], [8, 3]])      # linhas: fungicida A, B; colunas: com, sem
print("\n" + "=" * 70)
print("Dados fictícios: 23 lotes de sementes, dois fungicidas")
print(f"{'':<14}{'com doença':>12}{'sem doença':>12}{'total':>8}{'taxa':>9}")
rot = ["fungicida A", "fungicida B"]
for i in range(2):
    print(f"{rot[i]:<14}{tab[i,0]:12d}{tab[i,1]:12d}{tab[i].sum():8d}{100*tab[i,0]/tab[i].sum():8.1f}%")
print(f"{'total':<14}{tab[:,0].sum():12d}{tab[:,1].sum():12d}{tab.sum():8d}")

qui, p_qui, gl, esp = stats.chi2_contingency(tab, correction=False)
quiy, p_quiy, _, _ = stats.chi2_contingency(tab, correction=True)
print(f"\nFrequências esperadas:\n{np.round(esp,3)}")
print(f"Menor esperada = {esp.min():.3f}  (abaixo de 5: a aproximação fica duvidosa)")
print(f"qui-quadrado sem correção = {qui:.4f}, p = {p_qui:.5f}")
print(f"qui-quadrado com Yates    = {quiy:.4f}, p = {p_quiy:.5f}")
od, pf = stats.fisher_exact(tab)
print(f"Fisher exato, bilateral: razão de chances condicional = {od:.4f}, p = {pf:.5f}")
print(f"Fisher exato, unilateral (menos doença com A): p = "
      f"{stats.fisher_exact(tab, alternative='less')[1]:.5f}")

# 2b. De onde vem o valor-p exato: enumerar as tabelas com as MESMAS margens
print("\nAs margens são fixas: linhas 12 e 11, colunas 10 e 13.")
print("Só há 11 tabelas possíveis, indexadas pela célula (1,1):")
n1, n2 = tab[0].sum(), tab[1].sum(); m1 = tab[:, 0].sum(); N = tab.sum()
obs_p = stats.hypergeom.pmf(tab[0,0], N, n1, m1)
print(f"{'a':>3}{'tabela':>22}{'probabilidade':>15}{'<= observada?':>15}")
soma = 0.0
for a in range(max(0, m1-n2), min(n1, m1)+1):
    pr = stats.hypergeom.pmf(a, N, n1, m1)
    inc = pr <= obs_p * (1 + 1e-7)
    if inc: soma += pr
    print(f"{a:3d}   [{a:2d} {n1-a:2d} | {m1-a:2d} {N-n1-m1+a:2d}]   {pr:13.8f}   {'sim' if inc else 'não':>10}")
print(f"Soma das probabilidades <= a observada ({obs_p:.8f}) = {soma:.6f} = valor-p bilateral")

# 2c. Razão de chances: a amostral e a condicional de máxima verossimilhança
or_amostral = tab[0,0]*tab[1,1]/(tab[0,1]*tab[1,0])
res = stats.contingency.odds_ratio(tab)          # condicional, por máxima verossimilhança
ic = res.confidence_interval(0.95)
print(f"\nRazão de chances amostral = {or_amostral:.6f}   (a que fisher_exact devolve)")
print(f"Razão de chances condicional = {res.statistic:.6f}   (a que fisher.test do R devolve)")
print(f"IC 95% condicional = [{ic.low:.4f}; {ic.high:.4f}]")
print("As duas NÃO são a mesma coisa. A condicional maximiza a verossimilhança da")
print("hipergeométrica não central, que é a distribuição sobre a qual o teste é")
print("construído; a amostral é o produto cruzado simples. Com contagens pequenas")
print("a diferença é visível, e o R reporta a condicional.")
se = np.sqrt(sum(1/tab[i,j] for i in range(2) for j in range(2)))
print(f"IC 95% de Wald sobre a amostral, para comparação = "
      f"[{np.exp(np.log(or_amostral)-1.96*se):.4f}; {np.exp(np.log(or_amostral)+1.96*se):.4f}]")

# ===================================================================
# PARTE 3 - o preço do exato: conservadorismo
print("\n" + "=" * 70)
print("O teste exato é CONSERVADOR: a taxa de erro real fica abaixo do nominal.")
print("Simulação: duas proporções iguais a 0,3, com n = 15 por grupo.")
rng = np.random.default_rng(2026)
M = 20000
n_g = 15; p0 = 0.3
rej_f = rej_q = rej_y = rej_mid = 0
for _ in range(M):
    x = rng.binomial(n_g, p0); y = rng.binomial(n_g, p0)
    t = np.array([[x, n_g-x], [y, n_g-y]])
    if t.sum(axis=0).min() == 0 or t.sum(axis=1).min() == 0:
        continue
    pf_ = stats.fisher_exact(t)[1]
    if pf_ < 0.05: rej_f += 1
    # mid-p: metade da probabilidade da tabela observada
    N_ = t.sum(); n1_ = t[0].sum(); m1_ = t[:,0].sum()
    po = stats.hypergeom.pmf(t[0,0], N_, n1_, m1_)
    s = sum(stats.hypergeom.pmf(a, N_, n1_, m1_)
            for a in range(max(0,m1_-t[1].sum()), min(n1_,m1_)+1)
            if stats.hypergeom.pmf(a, N_, n1_, m1_) < po*(1-1e-9))
    if s + 0.5*sum(stats.hypergeom.pmf(a, N_, n1_, m1_)
                   for a in range(max(0,m1_-t[1].sum()), min(n1_,m1_)+1)
                   if abs(stats.hypergeom.pmf(a, N_, n1_, m1_) - po) < po*1e-9) < 0.05:
        rej_mid += 1
    try:
        if stats.chi2_contingency(t, correction=False)[1] < 0.05: rej_q += 1
        if stats.chi2_contingency(t, correction=True)[1] < 0.05: rej_y += 1
    except ValueError:
        pass
print(f"  Fisher exato:            {100*rej_f/M:5.2f}%")
print(f"  Fisher com mid-p:        {100*rej_mid/M:5.2f}%")
print(f"  qui-quadrado com Yates:  {100*rej_y/M:5.2f}%")
print(f"  qui-quadrado sem Yates:  {100*rej_q/M:5.2f}%")
print("  (o nominal é 5%; o exato e o Yates ficam bem abaixo, o sem correção fica perto)")
print("\nO conservadorismo vem da discretude: com margens fixas há poucos valores-p")
print("possíveis, e raramente algum cai exatamente em 0,05. A correção mid-p recupera")
print("parte da folga contando metade da probabilidade da tabela observada.")

# ===================================================================
# PARTE 4 - tabelas maiores que 2x2
print("\n" + "=" * 70)
print("Fisher também vale para tabelas r x c. Dados fictícios: 3 cultivares x 3 notas")
tab3 = np.array([[8, 3, 1], [4, 6, 2], [1, 4, 7]])
print(tab3)
q3, p3, g3, e3 = stats.chi2_contingency(tab3)
print(f"qui-quadrado = {q3:.4f}, gl = {g3}, p = {p3:.5f} | menor esperada = {e3.min():.3f}")
# valor-p exato por enumeração via simulação de tabelas com margens fixas
rng = np.random.default_rng(7)
B = 200000
lin = tab3.sum(axis=1); col = tab3.sum(axis=0); n3 = tab3.sum()
def qui_est(t):
    e = np.outer(t.sum(1), t.sum(0))/t.sum()
    return ((t-e)**2/e).sum()
obs3 = qui_est(tab3)
cont = 0
for _ in range(B):
    # tabela aleatória com as mesmas margens (algoritmo de permutação)
    linhas = np.repeat(np.arange(3), lin)
    colunas = rng.permutation(np.repeat(np.arange(3), col))
    t = np.zeros((3,3))
    np.add.at(t, (linhas, colunas), 1)
    if qui_est(t) >= obs3 - 1e-9: cont += 1
print(f"valor-p por permutação ({B} tabelas) = {(1+cont)/(B+1):.6f}")
print("No R: fisher.test(tab3) faz a enumeração exata;")
print("      chisq.test(tab3, simulate.p.value = TRUE) usa reamostragem.")
