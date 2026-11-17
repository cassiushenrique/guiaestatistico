# Bibliotecas: numpy, scipy, matplotlib e scikit-learn (todas já no Google Colab)
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.datasets import load_iris

# ===================================================================
# PARTE 1 - a mecânica, com seis pontos
# Dados fictícios: seis amostras, duas medidas cada.
P = np.array([[1.0, 1.0], [1.5, 1.2], [5.0, 5.0], [5.5, 5.2], [3.0, 4.5], [8.0, 1.0]])
rot = ["a", "b", "c", "d", "e", "f"]
print("Seis amostras, duas medidas")
for r, p in zip(rot, P):
    print(f"  {r}: ({p[0]:.1f}; {p[1]:.1f})")
D = squareform(pdist(P))
print("\nMatriz de distâncias euclidianas")
print("      " + "".join(f"{r:>8}" for r in rot))
for i, r in enumerate(rot):
    print(f"  {r}  " + "".join(f"{D[i,j]:8.3f}" for j in range(6)))

print("\nFusões sucessivas, por três critérios de ligação")
for metodo, nome in [("single", "ligação simples (vizinho mais próximo)"),
                     ("complete", "ligação completa (vizinho mais distante)"),
                     ("average", "ligação média (UPGMA)"),
                     ("ward", "método de Ward (mínima variância)")]:
    Z = hierarchy.linkage(pdist(P), method=metodo)
    print(f"\n{nome}")
    print(f"{'passo':>6}{'funde':>16}{'distância':>12}{'tamanho':>9}")
    nomes = {i: rot[i] for i in range(6)}
    for k_, (i, j, d, t) in enumerate(Z, start=1):
        novo = "(" + nomes[int(i)] + "," + nomes[int(j)] + ")"
        nomes[5 + k_] = novo
        print(f"{k_:6d}{novo:>16}{d:12.4f}{int(t):9d}")
    coph = hierarchy.cophenet(Z, pdist(P))[0]
    print(f"  correlação cofenética = {coph:.4f}")

print("\nA correlação cofenética mede o quanto o dendrograma preserva as distâncias")
print("originais. Valores acima de 0,8 indicam uma representação fiel.")

# ===================================================================
# PARTE 2 - aplicação: iris, sem usar a espécie
print("\n" + "=" * 70)
print("iris: 150 flores, quatro medidas. O agrupamento NÃO usa a espécie;")
print("ela só é revelada no final, para avaliar o resultado.")
d = load_iris(); X = d.data; esp = d.target
especies = ["setosa", "versicolor", "virginica"]
print(f"\nEscalas das quatro variáveis (a padronização importa):")
for j, nm in enumerate(d.feature_names):
    print(f"  {nm:<22} média = {X[:,j].mean():6.3f}  DP = {X[:,j].std(ddof=1):6.3f}  "
          f"amplitude = {np.ptp(X[:,j]):5.2f}")
Z_pad = (X - X.mean(0))/X.std(0, ddof=1)

for titulo, M in [("dados ORIGINAIS", X), ("dados PADRONIZADOS", Z_pad)]:
    dist = pdist(M)
    print(f"\n{titulo}")
    print(f"{'ligação':<12}{'cofenética':>12}{'grupos de 3':>40}")
    for metodo in ("single", "complete", "average", "ward"):
        Z = hierarchy.linkage(dist, method=metodo)
        gr = hierarchy.fcluster(Z, 3, criterion="maxclust")
        coph = hierarchy.cophenet(Z, dist)[0]
        tam = [int(np.sum(gr == g)) for g in np.unique(gr)]
        print(f"{metodo:<12}{coph:12.4f}{str(tam):>40}")

# Avaliação contra a espécie verdadeira
print("\nComparando o agrupamento de Ward (dados padronizados) com a espécie")
Zw = hierarchy.linkage(pdist(Z_pad), method="ward")
gr = hierarchy.fcluster(Zw, 3, criterion="maxclust")
print(f"{'':<14}" + "".join(f"{f'grupo {g}':>10}" for g in np.unique(gr)))
for i, e in enumerate(especies):
    print(f"{e:<14}" + "".join(f"{int(np.sum((gr==g)&(esp==i))):10d}" for g in np.unique(gr)))
# índice de Rand ajustado, calculado sem sklearn.metrics
def rand_ajustado(a, b):
    n = len(a)
    tab = np.zeros((len(np.unique(a)), len(np.unique(b))))
    for i, va in enumerate(np.unique(a)):
        for j, vb in enumerate(np.unique(b)):
            tab[i, j] = np.sum((a == va) & (b == vb))
    c2 = lambda m: m*(m-1)/2
    soma_ij = c2(tab).sum(); soma_i = c2(tab.sum(1)).sum(); soma_j = c2(tab.sum(0)).sum()
    esperado = soma_i*soma_j/c2(n)
    maximo = (soma_i + soma_j)/2
    return (soma_ij - esperado)/(maximo - esperado)
print(f"\nÍndice de Rand ajustado = {rand_ajustado(esp, gr):.4f}")
Zw_bruto = hierarchy.linkage(pdist(X), method="ward")
gr_bruto = hierarchy.fcluster(Zw_bruto, 3, criterion="maxclust")
print(f"Com os dados ORIGINAIS (sem padronizar), o mesmo critério dá "
      f"Rand ajustado = {rand_ajustado(esp, gr_bruto):.4f}")
print("Aqui a padronização PIORA o resultado, porque ela dá o mesmo peso à largura")
print("da sépala, que quase não separa as espécies, e ao comprimento da pétala, que")
print("separa muito bem. Padronizar é o padrão quando as unidades diferem; quando")
print("são as mesmas, como aqui (todas em cm), pode custar informação.")
print(f"Flores classificadas no grupo majoritário da sua espécie: ", end="")
acerto = sum(max(np.sum((gr == g) & (esp == i)) for i in range(3)) for g in np.unique(gr))
print(f"{acerto} de 150 ({100*acerto/150:.1f}%)")
print("A setosa se separa perfeitamente; versicolor e virginica se sobrepõem,")
print("exatamente como nas elipses da ficha da MANOVA.")

# Altura das fusões: onde cortar
print("\nAs dez últimas alturas de fusão (Ward, dados padronizados)")
alturas = Zw[-10:, 2]
print("  " + "  ".join(f"{a:.3f}" for a in alturas))
saltos = np.diff(alturas)
i_salto = int(np.argmax(saltos))
k_salto = 10 - i_salto          # quantos grupos existem logo antes dessa fusão
print(f"  maior salto: de {alturas[i_salto]:.3f} para {alturas[i_salto+1]:.3f} "
      f"(aumento de {saltos.max():.3f})")
print(f"  cortar logo abaixo dele deixa {k_salto} grupos")
print("  O critério do salto é visual e aproximado; compare sempre com o que se sabe")
print("  sobre o problema.")
for k_ in (2, 3, 4, 5):
    g_ = hierarchy.fcluster(Zw, k_, criterion="maxclust")
    print(f"  k = {k_}: tamanhos {[int(np.sum(g_==x)) for x in np.unique(g_)]}, "
          f"Rand ajustado = {rand_ajustado(esp, g_):.4f}")

# ===================================================================
# PARTE 3 - o efeito do encadeamento na ligação simples
print("\n" + "=" * 70)
print("Por que a ligação simples costuma falhar: o efeito de encadeamento")
rng = np.random.default_rng(11)
A = rng.normal([0, 0], 0.35, (30, 2))
B = rng.normal([5, 0], 0.35, (30, 2))
ponte = np.column_stack([np.linspace(0.9, 4.1, 22), rng.normal(0, 0.08, 22)])
Pc = np.vstack([A, B, ponte])
verdade = np.r_[np.zeros(30), np.ones(30), np.full(22, 2)]
print(f"Dois grupos bem separados, com {len(ponte)} pontos formando uma ponte entre eles.")
print(f"{'ligação':<12}{'grupos (k = 2)':>18}{'ponte dividida?':>18}"
      f"{'altura da última':>18}{'penúltima':>12}{'razão':>8}")
for metodo in ("single", "complete", "average", "ward"):
    Zc = hierarchy.linkage(pdist(Pc), method=metodo)
    g = hierarchy.fcluster(Zc, 2, criterion="maxclust")
    tam = sorted(int(np.sum(g == x)) for x in np.unique(g))
    gp = g[60:]                      # os 22 pontos da ponte
    divide = "sim" if len(np.unique(gp)) > 1 else "NÃO"
    h1, h2 = Zc[-1, 2], Zc[-2, 2]
    print(f"{metodo:<12}{str(tam):>18}{divide:>18}{h1:18.4f}{h2:12.4f}{h1/h2:8.2f}")
print("\nA ligação simples anexa a ponte INTEIRA a um dos blocos e deixa o outro")
print("sozinho: ela percorre a cadeia de vizinhos próximos e nunca enxerga o vão")
print("entre os dois grupos. Os demais critérios dividem a ponte e recuperam a")
print("estrutura de dois blocos.")
print("\nA razão entre a última e a penúltima altura de fusão mostra o mesmo: na")
print("ligação simples ela é próxima de 1, ou seja, o dendrograma não tem um corte")
print("natural; em Ward ela passa de 3, e o corte em dois grupos salta aos olhos.")
