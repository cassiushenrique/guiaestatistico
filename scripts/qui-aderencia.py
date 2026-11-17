# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# PARTE 1 - proporções especificadas pela teoria: as ervilhas de Mendel
# F2 do cruzamento dibrido; a teoria prevê a razão 9:3:3:1.
categorias = ["lisa amarela", "lisa verde", "rugosa amarela", "rugosa verde"]
obs = np.array([315, 108, 101, 32])
razao = np.array([9, 3, 3, 1])
n = obs.sum()
esp = n * razao / razao.sum()
print("Ervilhas de Mendel (1866), F2 do cruzamento dibrido")
print(f"{'categoria':<18}{'observado':>10}{'esperado':>11}{'O - E':>9}{'(O-E)²/E':>11}")
for c, o, e in zip(categorias, obs, esp):
    print(f"{c:<18}{o:10d}{e:11.2f}{o-e:9.2f}{(o-e)**2/e:11.5f}")
qui = ((obs - esp)**2 / esp).sum()
gl = len(obs) - 1
print(f"{'total':<18}{n:10d}{esp.sum():11.2f}{0:9.2f}{qui:11.5f}")
print(f"\nqui-quadrado = {qui:.5f} | gl = {gl} | p-valor = {stats.chi2.sf(qui, gl):.5f}")
print(f"scipy: {stats.chisquare(obs, esp)}")
print(f"Menor frequência esperada: {esp.min():.2f} (a regra prática pede pelo menos 5)")
print("Resíduos padronizados (O - E)/raiz(E):", np.round((obs - esp)/np.sqrt(esp), 4))

# 1b. O valor-p grande: o que ele significa e o que não significa
print("\nUm valor-p de 0,925 NÃO prova que a teoria 9:3:3:1 está certa.")
print("Ele diz apenas que os dados são compatíveis com ela. Veja o que aconteceria")
print("se a razão verdadeira fosse ligeiramente diferente:")
for alt, nome in [(np.array([9,3,3,1]), "9:3:3:1 (a hipótese)"),
                  (np.array([8.5,3.2,3.2,1.1]), "8,5:3,2:3,2:1,1"),
                  (np.array([7,3,3,1]), "7:3:3:1"),
                  (np.array([1,1,1,1]), "1:1:1:1")]:
    ea = n*alt/alt.sum(); qa = ((obs-ea)**2/ea).sum()
    print(f"  contra {nome:<24} qui² = {qa:9.3f} | p = {stats.chi2.sf(qa, gl):.3e}")
print("Com n = 556 o teste distingue bem 9:3:3:1 de 7:3:3:1, mas não de 8,5:3,2:3,2:1,1.")

# 1c. A crítica de Fisher: um ajuste bom demais
print("\nA crítica de R. A. Fisher (1936): somando os qui-quadrados de TODOS os")
print("experimentos de Mendel, ele obteve 41,6 com 84 graus de liberdade.")
print(f"P(qui²(84) <= 41,6) = {stats.chi2.cdf(41.6, 84):.5f}")
print("Ou seja, um ajuste tão bom ou melhor ocorreria em cerca de 1 em 25.000 repetições.")
print("O teste de aderência tem, portanto, DOIS lados: valores grandes indicam")
print("incompatibilidade com a teoria; valores muito pequenos indicam dados bons demais.")

# ===================================================================
# PARTE 2 - ajuste de uma DISTRIBUIÇÃO, com parâmetro estimado
print("\n" + "=" * 70)
print("Mortes por coice de cavalo no exército prussiano (Bortkiewicz, 1898)")
print("200 observações corpo-ano; a hipótese é a distribuição de Poisson.")
obs2 = np.array([109, 65, 22, 3, 1])
x = np.arange(5); n2 = obs2.sum()
lam = (obs2 * x).sum() / n2
print(f"\nTotal de mortes = {int((obs2*x).sum())} em {n2} corpos-ano -> lambda estimado = {lam:.4f}")
esp2 = np.array([stats.poisson.pmf(i, lam) * n2 for i in range(4)] + [stats.poisson.sf(3, lam) * n2])
rot = ["0 mortes", "1 morte", "2 mortes", "3 mortes", "4 ou mais"]
print(f"{'categoria':<14}{'observado':>10}{'esperado':>11}{'(O-E)²/E':>11}")
for r, o, e in zip(rot, obs2, esp2):
    print(f"{r:<14}{o:10d}{e:11.3f}{(o-e)**2/e:11.5f}")
q2 = ((obs2 - esp2)**2 / esp2).sum()
print(f"\nqui-quadrado = {q2:.4f}")
print(f"Graus de liberdade = {len(obs2)} - 1 - 1 = {len(obs2)-2}  (um a menos, porque lambda foi ESTIMADO)")
print(f"p-valor = {stats.chi2.sf(q2, len(obs2)-2):.4f}")
print(f"Menor frequência esperada = {esp2.min():.3f}: abaixo de 5, e até abaixo de 1.")

print("\nAgrupando as categorias de cauda, como a regra prática recomenda:")
obs3 = np.array([109, 65, 22, 4]); esp3 = np.array(list(esp2[:3]) + [esp2[3:].sum()])
rot3 = ["0 mortes", "1 morte", "2 mortes", "3 ou mais"]
for r, o, e in zip(rot3, obs3, esp3):
    print(f"  {r:<12}{o:6d}{e:10.3f}{(o-e)**2/e:10.5f}")
q3 = ((obs3 - esp3)**2 / esp3).sum()
print(f"  qui-quadrado = {q3:.4f} | gl = {len(obs3)-2} | p-valor = {stats.chi2.sf(q3, len(obs3)-2):.4f}")
print(f"  menor esperado = {esp3.min():.3f}: agora dentro da regra.")
print("Conclusão: as mortes por coice são compatíveis com um processo de Poisson.")

# ===================================================================
# PARTE 3 - um caso que REJEITA, e o que fazer com esperados pequenos
print("\n" + "=" * 70)
print("Dados fictícios: 180 peças refugadas em uma linha, por tipo de defeito.")
print("A hipótese do setor de qualidade é que os cinco tipos são equiprováveis.")
tipos = ["dimensional", "acabamento", "montagem", "material", "outros"]
obs4 = np.array([58, 41, 35, 32, 14])
n4 = obs4.sum(); esp4 = np.full(5, n4/5)
print(f"{'defeito':<14}{'observado':>10}{'esperado':>10}{'(O-E)²/E':>11}{'resíduo pad.':>14}")
for t, o, e in zip(tipos, obs4, esp4):
    print(f"{t:<14}{o:10d}{e:10.1f}{(o-e)**2/e:11.4f}{(o-e)/np.sqrt(e):14.3f}")
q4 = ((obs4 - esp4)**2 / esp4).sum()
print(f"\nqui-quadrado = {q4:.4f} | gl = 4 | p-valor = {stats.chi2.sf(q4, 4):.5f}")
print(f"scipy: {stats.chisquare(obs4)}")
print("Os resíduos padronizados dizem ONDE está a discrepância: o defeito dimensional")
print(f"contribui com {(obs4[0]-esp4[0])**2/esp4[0]:.3f} dos {q4:.3f}, "
      f"e 'outros' com {(obs4[4]-esp4[4])**2/esp4[4]:.3f}.")
print(f"Resíduos acima de 2 em valor absoluto merecem atenção: "
      f"{[t for t,r in zip(tipos,(obs4-esp4)/np.sqrt(esp4)) if abs(r)>2]}")

# 3b. Tamanho de efeito e poder
w = np.sqrt(q4 / n4)
print(f"\nTamanho de efeito w de Cohen = raiz(qui²/n) = {w:.4f}")
print("  (0,1 pequeno; 0,3 médio; 0,5 grande)")

# 3c. Valor-p por simulação: aqui, só para conferir a aproximação
print("\nO valor-p também pode ser obtido por simulação, sem recorrer à aproximação:")
rng = np.random.default_rng(2026)
B = 100000
sim = rng.multinomial(n4, np.full(5, 0.2), size=B)
qsim = (((sim - esp4)**2) / esp4).sum(axis=1)
print(f"  valor-p simulado ({B} reamostragens) = {(1 + (qsim >= q4).sum()) / (B + 1):.6f}")
print(f"  valor-p pela aproximação qui-quadrado = {stats.chi2.sf(q4, 4):.6f}")
print("  No R: chisq.test(obs, simulate.p.value = TRUE, B = 100000)")

print("Aqui os dois coincidem na prática, porque todas as frequências esperadas são 36.")

# 3d. Quando a aproximação falha: a taxa de erro real com esperados pequenos
print("\nA aproximação qui-quadrado é assintótica. Para ver quando ela falha, basta")
print("simular sob a hipótese nula e contar quantas vezes ela rejeita a 5%:")
print(" n     esperado por célula   taxa de rejeição a 5% (alvo: 5%)")
for n_ in (10, 15, 20, 25, 50, 100, 200):
    e_ = np.full(5, n_/5)
    sim_ = rng.multinomial(n_, np.full(5, 0.2), size=40000)
    q_ = (((sim_ - e_)**2)/e_).sum(axis=1)
    print(f" {n_:4d}        {n_/5:6.1f}            {100*(q_ > stats.chi2.ppf(0.95, 4)).mean():6.2f}%")
print("Com n = 10 (esperado 2 por célula) o teste rejeita menos que o nominal, e com")
print("n = 15 e 20 ele oscila: a distribuição discreta não se ajusta bem à contínua.")
print("A partir de cerca de 25 observações, ou seja 5 por célula, a aproximação fica boa.")
print("É dessa simulação que vem a regra prática, e não de um teorema.")
