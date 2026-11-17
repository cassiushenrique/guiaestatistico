# Bibliotecas: numpy, scipy, matplotlib e scikit-learn (todas já no Google Colab)
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris

# ===================================================================
# PARTE 1 - o algoritmo, passo a passo
# Dados fictícios: doze amostras, duas medidas.
P = np.array([[1.0,1.0],[1.4,0.8],[0.8,1.5],[1.2,1.3],
              [5.0,5.2],[5.4,4.8],[4.8,5.5],[5.2,5.0],
              [1.0,5.0],[1.3,4.6],[0.7,5.3],[1.1,4.9]])
print("Doze amostras, duas medidas. Partimos de centros propositalmente ruins.")
centros = np.array([[1.0,1.0],[1.4,0.8],[0.8,1.5]])   # os três no mesmo canto
print(f"Centros iniciais:\n{centros}")
for it in range(1, 9):
    d = cdist(P, centros)
    rot = d.argmin(axis=1)
    novos = np.array([P[rot==j].mean(axis=0) if np.any(rot==j) else centros[j]
                      for j in range(3)])
    inercia = sum(((P[rot==j]-novos[j])**2).sum() for j in range(3))
    print(f"\niteração {it}")
    print(f"  atribuição: {rot}")
    print(f"  tamanhos:   {[int(np.sum(rot==j)) for j in range(3)]}")
    print(f"  novos centros: {np.round(novos,4).tolist()}")
    print(f"  inércia (soma dos quadrados dentro dos grupos) = {inercia:.4f}")
    if np.allclose(novos, centros, atol=1e-12):
        print("  os centros pararam de se mover: convergiu")
        break
    centros = novos
print("\nO algoritmo alterna dois passos: atribuir cada ponto ao centro mais")
print("próximo e recalcular cada centro como a média do seu grupo. A inércia")
print("nunca aumenta, o que garante convergência — mas apenas para um mínimo LOCAL.")

# 1b. A dependência da inicialização
print("\nA inicialização importa: 2.000 partidas aleatórias")
rng = np.random.default_rng(7)
def kmeans(P, k, centros, iters=300):
    for _ in range(iters):
        rot = cdist(P, centros).argmin(axis=1)
        novos = np.array([P[rot==j].mean(axis=0) if np.any(rot==j) else centros[j]
                          for j in range(k)])
        if np.allclose(novos, centros): break
        centros = novos
    rot = cdist(P, centros).argmin(axis=1)
    return centros, rot, sum(((P[rot==j]-centros[j])**2).sum() for j in range(k))
inercias = []
for _ in range(2000):
    c0 = P[rng.choice(len(P), 3, replace=False)]
    inercias.append(kmeans(P, 3, c0)[2])
inercias = np.array(inercias)
vals, cont = np.unique(np.round(inercias, 6), return_counts=True)
print(f"{'inércia final':>15}{'frequência':>12}{'proporção':>12}")
for v, c in zip(vals[:6], cont[:6]):
    print(f"{v:15.4f}{c:12d}{100*c/2000:11.1f}%")
print(f"  o mínimo global é {vals[0]:.4f}, alcançado em {100*cont[0]/2000:.1f}% das partidas")
print("  É por isso que se roda o algoritmo várias vezes (nstart no R, n_init no Python)")
print("  e se guarda a solução de menor inércia.")

# ===================================================================
# PARTE 2 - escolher k
print("\n" + "=" * 70)
print("iris: 150 flores, quatro medidas, sem usar a espécie")
d = load_iris(); X = d.data; esp = d.target
especies = ["setosa", "versicolor", "virginica"]

def melhor(P, k, tentativas=50, semente=0):
    r = np.random.default_rng(semente)
    best = None
    for _ in range(tentativas):
        c0 = P[r.choice(len(P), k, replace=False)]
        c, rot, inr = kmeans(P, k, c0)
        if best is None or inr < best[2]: best = (c, rot, inr)
    return best

def silhueta(P, rot):
    D = cdist(P, P); s = np.zeros(len(P))
    for i in range(len(P)):
        mesmo = rot == rot[i]; mesmo[i] = False
        if mesmo.sum() == 0: s[i] = 0; continue
        a = D[i, mesmo].mean()
        b = min(D[i, rot == g].mean() for g in np.unique(rot) if g != rot[i])
        s[i] = (b - a)/max(a, b)
    return s

print(f"\n{'k':>3}{'inércia':>12}{'queda':>10}{'silhueta média':>17}{'Calinski-Harabasz':>20}")
inr_ant = None
guardados = {}
for k_ in range(2, 8):
    c, rot, inr = melhor(X, k_, semente=k_)
    guardados[k_] = (c, rot, inr)
    sil = silhueta(X, rot).mean()
    entre = sum(np.sum(rot==j)*((c[j]-X.mean(0))**2).sum() for j in range(k_))
    ch = (entre/(k_-1))/(inr/(len(X)-k_))
    q = "" if inr_ant is None else f"{100*(inr_ant-inr)/inr_ant:9.1f}%"
    print(f"{k_:3d}{inr:12.3f}{q:>10}{sil:17.4f}{ch:20.2f}")
    inr_ant = inr
print("\nA inércia sempre cai quando k aumenta: ela não pode ser usada sozinha.")
print("O 'cotovelo' é o ponto a partir do qual a queda deixa de compensar.")
print("A silhueta e o critério de Calinski-Harabasz tentam formalizar isso.")

# 2b. Comparação com a espécie
print("\nSolução com k = 3, comparada à espécie verdadeira")
c3, rot3, inr3 = guardados[3]
print(f"{'':<14}" + "".join(f"{f'grupo {g+1}':>10}" for g in range(3)))
for i, e in enumerate(especies):
    print(f"{e:<14}" + "".join(f"{int(np.sum((rot3==g)&(esp==i))):10d}" for g in range(3)))
def rand_ajustado(a, b):
    tab = np.zeros((len(np.unique(a)), len(np.unique(b))))
    for i, va in enumerate(np.unique(a)):
        for j, vb in enumerate(np.unique(b)):
            tab[i, j] = np.sum((a == va) & (b == vb))
    c2 = lambda m: m*(m-1)/2
    sij = c2(tab).sum(); si = c2(tab.sum(1)).sum(); sj = c2(tab.sum(0)).sum()
    return (sij - si*sj/c2(len(a)))/((si + sj)/2 - si*sj/c2(len(a)))
acerto = sum(max(np.sum((rot3==g)&(esp==i)) for i in range(3)) for g in range(3))
print(f"\nÍndice de Rand ajustado = {rand_ajustado(esp, rot3):.4f}")
print(f"Flores no grupo majoritário da sua espécie: {acerto} de 150 ({100*acerto/150:.1f}%)")
print(f"Centros encontrados (cm):")
for j in range(3):
    print(f"  grupo {j+1}: " + "  ".join(f"{v:5.2f}" for v in c3[j]))

# 2c. A padronização
Z = (X - X.mean(0))/X.std(0, ddof=1)
cz, rotz, inrz = melhor(Z, 3, semente=3)
print(f"\nCom os dados padronizados: Rand ajustado = {rand_ajustado(esp, rotz):.4f}")
print(f"Com os dados originais:    Rand ajustado = {rand_ajustado(esp, rot3):.4f}")
print("Como as quatro medidas estão na mesma unidade (cm) e a pétala separa muito")
print("melhor que a sépala, padronizar iguala os pesos e PIORA o resultado aqui.")
print("Quando as variáveis têm unidades diferentes, padronizar é obrigatório.")

# ===================================================================
# PARTE 3 - o que o k-means não consegue fazer
print("\n" + "=" * 70)
print("O k-means procura grupos ESFÉRICOS e de tamanhos parecidos.")
rng = np.random.default_rng(3)
# dois anéis concêntricos
ang = rng.uniform(0, 2*np.pi, 150)
r1 = rng.normal(1.0, 0.08, 150); r2 = rng.normal(3.0, 0.08, 150)
ang2 = rng.uniform(0, 2*np.pi, 150)
An = np.vstack([np.column_stack([r1*np.cos(ang), r1*np.sin(ang)]),
                np.column_stack([r2*np.cos(ang2), r2*np.sin(ang2)])])
verdade = np.r_[np.zeros(150), np.ones(150)]
_, rot_an, _ = melhor(An, 2, semente=1)
print(f"\nDois anéis concêntricos (150 pontos cada):")
print(f"  Rand ajustado do k-means = {rand_ajustado(verdade, rot_an):.4f}")
tam = [int(np.sum(rot_an==j)) for j in range(2)]
print(f"  tamanhos dos grupos encontrados: {tam}")
print("  O k-means corta o plano em duas metades e falha por completo.")
# grupos alongados e paralelos, deslocados na direção PERPENDICULAR ao eixo longo
cov = [[9, 8.5], [8.5, 9]]
A1 = rng.multivariate_normal([0, 0], cov, 150)
A2 = rng.multivariate_normal([1.6, -1.6], cov, 150)
Al = np.vstack([A1, A2]); verd2 = np.r_[np.zeros(150), np.ones(150)]
_, rot_al, _ = melhor(Al, 2, semente=2)
print(f"\nDois grupos alongados e paralelos (eixo longo na diagonal, separação")
print(f"perpendicular a ele):")
print(f"  Rand ajustado do k-means = {rand_ajustado(verd2, rot_al):.4f}")
print("  O k-means corta perpendicularmente ao eixo LONGO, que é onde a variação")
print("  é maior, e ignora a separação real.")
# tamanhos muito desiguais e dispersões diferentes
B1 = rng.normal([0,0], 1.2, (285, 2)); B2 = rng.normal([2.5,0], 0.3, (15, 2))
Bd = np.vstack([B1, B2]); verd3 = np.r_[np.zeros(285), np.ones(15)]
_, rot_bd, _ = melhor(Bd, 2, semente=4)
print(f"\nGrupos de tamanhos muito desiguais (285 e 15 pontos) e dispersões")
print(f"diferentes (DP de 1,2 e de 0,3):")
print(f"  Rand ajustado do k-means = {rand_ajustado(verd3, rot_bd):.4f}")
print(f"  tamanhos encontrados: {[int(np.sum(rot_bd==j)) for j in range(2)]}")
print("  Em vez de isolar o grupo pequeno, o k-means parte o grande ao meio:")
print("  dividir 285 pontos dispersos reduz mais a inércia do que separar 15.")
print("\nNos três casos a estrutura existe e salta aos olhos num gráfico, mas não é")
print("do tipo que o k-means procura: ele minimiza a soma de quadrados dentro dos")
print("grupos, o que equivale a procurar blocos esféricos, de tamanhos e dispersões")
print("parecidos. Agrupamento espectral e DBSCAN resolvem os anéis; misturas de")
print("gaussianas resolvem os grupos alongados e as dispersões desiguais; o")
print("agrupamento hierárquico com ligação média ou de Ward é mais flexível que o")
print("k-means nos dois últimos casos.")
