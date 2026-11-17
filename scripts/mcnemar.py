# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# PARTE 1 - o exemplo padrão: duas avaliações das MESMAS pessoas
# 1.600 entrevistados opinaram sobre o desempenho de um presidente em duas
# pesquisas sucessivas. Fonte: Agresti (1990), p. 350; é o exemplo de
# mcnemar.test() na documentação do R.
tab = np.array([[794, 150], [86, 570]], dtype=float)
rot_l = ["1ª pesquisa: aprova", "1ª pesquisa: desaprova"]
rot_c = ["2ª: aprova", "2ª: desaprova"]
print("Desempenho presidencial em duas pesquisas sucessivas (Agresti, 1990)")
print(f"{'':<24}{rot_c[0]:>14}{rot_c[1]:>15}{'total':>9}")
for i in range(2):
    print(f"{rot_l[i]:<24}{int(tab[i,0]):14d}{int(tab[i,1]):15d}{int(tab[i].sum()):9d}")
print(f"{'total':<24}{int(tab[:,0].sum()):14d}{int(tab[:,1].sum()):15d}{int(tab.sum()):9d}")
n = tab.sum()
p1 = tab[0].sum()/n; p2 = tab[:,0].sum()/n
print(f"\nAprovação na 1ª pesquisa: {100*p1:.2f}%   na 2ª: {100*p2:.2f}%   "
      f"diferença: {100*(p1-p2):+.2f} pontos")

# 1. Só as células DISCORDANTES carregam informação
b, c = tab[0,1], tab[1,0]
print(f"\nAs duas células concordantes (794 e 570) somam {int(tab[0,0]+tab[1,1])} pessoas")
print("que não mudaram de opinião: elas nada dizem sobre a MUDANÇA.")
print(f"As discordantes: b = {int(b)} mudaram de aprovar para desaprovar,")
print(f"                 c = {int(c)} mudaram de desaprovar para aprovar.")
print(f"A pergunta é se {int(b)} e {int(c)} são compatíveis com uma divisão meio a meio.")

# 2. As três formas do teste
chi_c = (abs(b - c) - 1)**2 / (b + c)
chi_s = (b - c)**2 / (b + c)
print(f"\nMcNemar com correção de continuidade: qui² = {chi_c:.4f}, gl = 1, "
      f"p = {stats.chi2.sf(chi_c,1):.4e}")
print(f"McNemar sem correção:                 qui² = {chi_s:.4f}, gl = 1, "
      f"p = {stats.chi2.sf(chi_s,1):.4e}")
p_exato = min(1.0, 2*stats.binom.cdf(min(b,c), b+c, 0.5))
print(f"Versão exata (binomial com p = 0,5): p = {p_exato:.4e}")
print(f"scipy: {stats.chi2_contingency(np.array([[b,0],[0,c]]))[0]:.4f} não é o caminho;")
print("       use statsmodels.stats.contingency_tables.mcnemar ou a conta acima.")

# 3. Medidas de efeito
print(f"\nRazão de chances de mudança (b/c) = {b/c:.4f}")
se_lor = np.sqrt(1/b + 1/c)
print(f"IC 95% = [{np.exp(np.log(b/c)-1.96*se_lor):.4f}; {np.exp(np.log(b/c)+1.96*se_lor):.4f}]")
dif = (b - c)/n
se_dif = np.sqrt((b + c - (b-c)**2/n))/n
print(f"Diferença de proporções = {dif:+.6f} ({100*dif:+.3f} pontos percentuais)")
print(f"IC 95% = [{100*(dif-1.96*se_dif):+.3f}; {100*(dif+1.96*se_dif):+.3f}] pontos")
print(f"Proporção de mudanças = {100*(b+c)/n:.2f}% dos entrevistados")

# 4. Por que o teste de independência seria a pergunta errada
qui_ind, p_ind, gl_ind, esp_ind = stats.chi2_contingency(tab, correction=False)
print(f"\nSe aplicássemos o qui-quadrado de independência a esta tabela:")
print(f"  qui² = {qui_ind:.2f}, p = {p_ind:.3e}")
print("Esse teste responde a 'a opinião na 2ª pesquisa depende da opinião na 1ª?'")
print("— pergunta trivial, cuja resposta é obviamente sim. O interesse está em saber")
print("se a PROPORÇÃO de aprovação mudou, e é a isso que McNemar responde.")

# ===================================================================
# PARTE 2 - a mesma conta em um delineamento pareado de diagnóstico
# Dados fictícios: 120 amostras de solo analisadas por dois métodos,
# cada amostra pelos dois. Resultado: detecta ou não detecta o patógeno.
print("\n" + "=" * 70)
print("Dados fictícios: 120 amostras de solo, dois métodos de detecção")
tab2 = np.array([[46, 22], [7, 45]], dtype=float)
print(f"{'':<22}{'B detecta':>12}{'B não detecta':>15}{'total':>8}")
for i, r in enumerate(["A detecta", "A não detecta"]):
    print(f"{r:<22}{int(tab2[i,0]):12d}{int(tab2[i,1]):15d}{int(tab2[i].sum()):8d}")
b2, c2 = tab2[0,1], tab2[1,0]; n2 = tab2.sum()
print(f"\nTaxa de detecção: método A = {100*tab2[0].sum()/n2:.2f}%, "
      f"método B = {100*tab2[:,0].sum()/n2:.2f}%")
print(f"Discordâncias: b = {int(b2)} (só A detecta), c = {int(c2)} (só B detecta)")
chi2c = (abs(b2-c2)-1)**2/(b2+c2)
print(f"McNemar com correção: qui² = {chi2c:.4f}, p = {stats.chi2.sf(chi2c,1):.5f}")
print(f"McNemar sem correção: qui² = {(b2-c2)**2/(b2+c2):.4f}, "
      f"p = {stats.chi2.sf((b2-c2)**2/(b2+c2),1):.5f}")
pe2 = min(1.0, 2*stats.binom.cdf(min(b2,c2), b2+c2, 0.5))
print(f"Versão exata: p = {pe2:.6f}   <- com b + c = {int(b2+c2)}, é a versão a usar")
print(f"Razão de chances de discordância = {b2/c2:.4f}")
print(f"Concordância observada = {100*(tab2[0,0]+tab2[1,1])/n2:.2f}%")
pe_ = ((tab2[0].sum()*tab2[:,0].sum() + tab2[1].sum()*tab2[:,1].sum())/n2**2)
kappa = ((tab2[0,0]+tab2[1,1])/n2 - pe_)/(1 - pe_)
print(f"Kappa de Cohen = {kappa:.4f}  (concordância além do acaso)")

# ===================================================================
# PARTE 3 - quando usar a versão exata: simulação
print("\n" + "=" * 70)
print("Correção de continuidade ou versão exata? Simulação com proporções IGUAIS.")
print("(20.000 experimentos; n = 100 pares; probabilidade de discordância variável)")
rng = np.random.default_rng(99)
M = 20000
print(" P(discordar)   b+c médio   sem correção   com correção   exato")
for npares, pd in ((100, 0.05), (100, 0.10), (100, 0.20), (100, 0.40), (400, 0.40), (1000, 0.40)):
    r_s = r_c = r_e = 0; soma_bc = 0
    for _ in range(M):
        d = rng.binomial(npares, pd)           # pares discordantes
        bb = rng.binomial(d, 0.5) if d > 0 else 0
        cc = d - bb; soma_bc += d
        if d == 0: continue
        if (bb-cc)**2/d > stats.chi2.ppf(0.95,1): r_s += 1
        if (abs(bb-cc)-1)**2/d > stats.chi2.ppf(0.95,1): r_c += 1
        if min(1.0, 2*stats.binom.cdf(min(bb,cc), d, 0.5)) < 0.05: r_e += 1
    print(f"   {pd:.2f}        {soma_bc/M:7.2f}      {100*r_s/M:7.2f}%      "
          f"{100*r_c/M:7.2f}%   {100*r_e/M:6.2f}%   (n = {npares})")
print("\nA versão SEM correção fica perto de 5% em todos os cenários. A corrigida e a")
print("exata ficam sistematicamente abaixo, e a convergência é lenta: mesmo com")
print("b + c perto de 400 elas ainda gastam menos de 4,5% dos 5% disponíveis.")
print("A razão é a discretude: b + c é fixo e b só assume valores inteiros, de modo")
print("que poucos valores-p são possíveis e raramente algum cai exatamente em 0,05.")
print("\nNa prática: com b + c pequeno (menos de 25), use a versão exata, que garante")
print("que a taxa nunca EXCEDE o nominal. Com b + c grande, a versão sem correção é")
print("a mais fiel ao nível declarado.")
