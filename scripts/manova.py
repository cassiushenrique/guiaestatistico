# Bibliotecas: numpy, scipy, matplotlib e scikit-learn (todas já no Google Colab)
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

def manova(X, g, rotulos=None):
    """Devolve as quatro estatísticas multivariadas e suas aproximações F."""
    X = np.asarray(X, float); g = np.asarray(g)
    niveis = np.unique(g); k = len(niveis); n, p = X.shape
    gm = X.mean(axis=0)
    H = np.zeros((p, p)); E = np.zeros((p, p))
    for i in niveis:
        Xi = X[g == i]; mi = Xi.mean(axis=0)
        H += len(Xi) * np.outer(mi - gm, mi - gm)       # entre grupos (hipótese)
        E += (Xi - mi).T @ (Xi - mi)                    # dentro dos grupos (erro)
    autov = np.sort(np.linalg.eigvals(np.linalg.inv(E) @ H).real)[::-1]
    glh, gle = k - 1, n - k
    s = min(glh, p)
    pillai = float(np.sum(autov[:s] / (1 + autov[:s])))
    wilks = float(np.prod(1 / (1 + autov[:s])))
    hl = float(np.sum(autov[:s]))
    roy = float(autov[0])
    m = (abs(glh - p) - 1) / 2; nn = (gle - p - 1) / 2
    saida = {}
    # Pillai
    d1 = s * (2*m + s + 1); d2 = s * (2*nn + s + 1)
    F = (pillai / (s - pillai)) * (d2 / d1)
    saida["Pillai"] = (pillai, F, d1, d2, stats.f.sf(F, d1, d2))
    # Wilks (aproximação de Rao)
    r_ = gle - (p - glh + 1) / 2
    u = (p * glh - 2) / 4
    den = p**2 + glh**2 - 5
    t = np.sqrt((p**2 * glh**2 - 4) / den) if den > 0 else 1.0
    d1 = p * glh; d2 = r_ * t - 2 * u
    F = ((1 - wilks**(1/t)) / wilks**(1/t)) * (d2 / d1)
    saida["Wilks"] = (wilks, F, d1, d2, stats.f.sf(F, d1, d2))
    # Hotelling-Lawley
    b = (gle + glh - p - 1) * (gle - 1) / ((gle - p - 3) * (gle - p))
    d1 = s * (2*m + s + 1); d2 = 4 + (d1 + 2) / (b - 1)
    cte = d1 * (d2 - 2) / (d2 * (gle - p - 1))
    F = hl / cte
    saida["Hotelling-Lawley"] = (hl, F, d1, d2, stats.f.sf(F, d1, d2))
    # Roy
    rr = max(p, glh); d1 = rr; d2 = gle - rr + glh
    F = roy * d2 / d1
    saida["Roy"] = (roy, F, d1, d2, stats.f.sf(F, d1, d2))
    return saida, H, E, autov

def imprime(saida):
    print("estatística           valor      F aprox.     gl1      gl2     p-valor")
    for nome, (v, F, d1, d2, pv) in saida.items():
        print(f"{nome:<20}{v:8.5f}  {F:11.4f}  {d1:6.1f}  {d2:7.2f}   {pv:.4e}")

def boxM(X, g):
    """Teste M de Box para igualdade das matrizes de covariância."""
    X = np.asarray(X, float); g = np.asarray(g)
    niveis = np.unique(g); k = len(niveis); n, p = X.shape
    ns = np.array([np.sum(g == i) for i in niveis], float)
    Ss = [np.cov(X[g == i], rowvar=False) for i in niveis]
    Sp = sum((ns[j] - 1) * Ss[j] for j in range(k)) / (n - k)
    M = (n - k) * np.log(np.linalg.det(Sp)) - sum((ns[j] - 1) * np.log(np.linalg.det(Ss[j])) for j in range(k))
    c = (sum(1/(ns - 1)) - 1/(n - k)) * (2*p**2 + 3*p - 1) / (6*(p + 1)*(k - 1))
    gl = (k - 1) * p * (p + 1) / 2
    qui = M * (1 - c)
    return M, qui, gl, stats.chi2.sf(qui, gl)

# ===================================================================
# PARTE 1 - por que uma MANOVA e não várias ANOVAs
# Dados fictícios: três lotes de um polímero, com duas medidas por corpo de prova:
# resistência à tração (MPa) e alongamento na ruptura (%).
res = np.array([51.62,49.33,48.71,51.01,49.47,50.83,50.19,51.66,49.92,50.79,52.05,50.06,47.94,49.26,49.31,
                52.30,49.77,50.40,48.27,50.21,50.20,52.54,48.56,50.72,50.08,49.83,52.58,49.98,51.20,51.70,
                50.45,51.49,48.84,50.42,50.59,49.51,50.13,49.74,47.87,49.70,50.59,50.74,49.25,49.57,48.08])
alo = np.array([31.10,28.81,29.43,31.04,29.77,30.33,29.88,32.14,30.19,30.52,32.30,30.22,28.15,29.17,29.59,
                30.93,28.42,29.58,27.82,29.63,29.47,31.75,27.62,30.36,29.42,28.83,31.39,29.25,30.31,30.91,
                30.77,31.55,30.04,31.17,31.05,29.75,30.40,29.89,28.50,30.53,32.10,31.07,29.70,30.63,28.82])
Y = np.column_stack([res, alo])
lote = np.repeat([0, 1, 2], 15)
rot = ["A", "B", "C"]
print("lote   n   resistência (MPa)   alongamento (%)")
for i in range(3):
    m = lote == i
    print(f"  {rot[i]}   {m.sum():2d}    {res[m].mean():6.3f} ± {res[m].std(ddof=1):.3f}    "
          f"{alo[m].mean():6.3f} ± {alo[m].std(ddof=1):.3f}")
dentro = np.vstack([Y[lote == i] - Y[lote == i].mean(axis=0) for i in range(3)])
print(f"Correlação entre as duas medidas, DENTRO dos lotes: r = {np.corrcoef(dentro.T)[0,1]:.4f}")

print("\nAs duas ANOVAs separadas não encontram nada:")
for j, nm in enumerate(["resistência", "alongamento"]):
    F, pv = stats.f_oneway(*[Y[lote == i, j] for i in range(3)])
    print(f"  {nm:<12} F(2; 42) = {F:.4f} | p-valor = {pv:.4f}")

print("\nA MANOVA, que usa as duas ao mesmo tempo:")
saida, H, E, autov = manova(Y, lote)
imprime(saida)
print(f"Autovalores de E⁻¹H: {np.round(autov[:2], 5)}")

print("\nPor que a diferença aparece: a combinação resistência - alongamento")
comb = res - alo
F, pv = stats.f_oneway(*[comb[lote == i] for i in range(3)])
print(f"  médias da combinação por lote: {[round(comb[lote==i].mean(), 3) for i in range(3)]}")
print(f"  ANOVA sobre a combinação: F(2; 42) = {F:.4f} | p-valor = {pv:.3e}")
print("  Cada medida sozinha varia muito; a razão entre elas é estável dentro do lote")
print("  e diferente entre lotes. Só a análise conjunta enxerga isso.")

print("\nPressupostos")
M, qui, gl, pM = boxM(Y, lote)
print(f"  Teste M de Box: M = {M:.4f}, qui-quadrado = {qui:.4f}, gl = {int(gl)}, p-valor = {pM:.4f}")
for i in range(3):
    r2 = stats.shapiro(Y[lote == i, 0])[1]; a2 = stats.shapiro(Y[lote == i, 1])[1]
    print(f"  Shapiro-Wilk no lote {rot[i]}: resistência p = {r2:.4f} | alongamento p = {a2:.4f}")

# ===================================================================
# PARTE 2 - o exemplo canônico: iris, com quatro respostas
print("\n" + "=" * 70)
print("iris: quatro medidas florais em três espécies (Anderson 1935; Fisher 1936)")
d = load_iris()
X = d.data; esp = d.target
nomes = ["comprimento da sépala", "largura da sépala", "comprimento da pétala", "largura da pétala"]
especies = ["setosa", "versicolor", "virginica"]
print("\nMédias por espécie (cm)")
print(f"{'espécie':<12}" + "".join(f"{n[:18]:>20}" for n in nomes))
for i in range(3):
    print(f"{especies[i]:<12}" + "".join(f"{X[esp==i, j].mean():20.3f}" for j in range(4)))

saida, H, E, autov = manova(X, esp)
print("\nMANOVA")
imprime(saida)
print(f"Autovalores de E⁻¹H: {np.round(autov[:2], 4)}")
print(f"O primeiro autovalor concentra {100*autov[0]/autov[:2].sum():.2f}% da separação:")
print("uma única direção discriminante quase basta para separar as três espécies.")

print("\nANOVAs univariadas de acompanhamento (com ajuste de Bonferroni para 4 testes)")
print("resposta                     F(2; 147)      p bruto    p ajustado")
for j, nm in enumerate(nomes):
    F, pv = stats.f_oneway(*[X[esp == i, j] for i in range(3)])
    print(f"{nm:<28}{F:10.3f}   {pv:11.3e}  {min(1, 4*pv):11.3e}")

print("\nPressupostos")
M, qui, gl, pM = boxM(X, esp)
print(f"  Teste M de Box: M = {M:.4f}, qui-quadrado = {qui:.4f}, gl = {int(gl)}, p-valor = {pM:.3e}")
print("  O M de Box REJEITA: as covariâncias das três espécies não são iguais.")
print("  Com grupos do mesmo tamanho (50 cada), o traço de Pillai é o mais robusto a isso.")
for i in range(3):
    d2 = np.sum((X[esp == i] - X[esp == i].mean(0)) @ np.linalg.inv(np.cov(X[esp == i], rowvar=False))
                * (X[esp == i] - X[esp == i].mean(0)), axis=1)
    b1 = np.mean([[ (a @ np.linalg.inv(np.cov(X[esp==i], rowvar=False)) @ b)**3
                   for b in (X[esp==i]-X[esp==i].mean(0))] for a in (X[esp==i]-X[esp==i].mean(0))])
    n_i, p_ = len(d2), 4
    A = n_i * b1 / 6
    glm = p_*(p_+1)*(p_+2)/6
    b2 = np.mean(d2**2)
    B = (b2 - p_*(p_+2)) / np.sqrt(8*p_*(p_+2)/n_i)
    print(f"  Mardia na {especies[i]:<11}: assimetria p = {stats.chi2.sf(A, glm):.4f} | "
          f"curtose p = {2*stats.norm.sf(abs(B)):.4f}")

print("\nDireção discriminante (primeiro autovetor de E⁻¹H), dividido pelo maior coeficiente")
ev, evec = np.linalg.eig(np.linalg.inv(E) @ H)
i0 = int(np.argmax(ev.real))
a1 = evec[:, i0].real
a1 = a1 / np.abs(a1).max()
for j, nm in enumerate(nomes):
    print(f"  {nm:<28}{a1[j]:+8.4f}")
print("  A separação é dominada pelas medidas da pétala, com sinal oposto ao das sépalas.")
print("  Confere com lda(Species ~ ., data = iris): os coeficientes de LD1 do R,")
print("  0.8294, 1.5345, -2.2012 e -2.8105, são proporcionais a estes (com sinal trocado).")
