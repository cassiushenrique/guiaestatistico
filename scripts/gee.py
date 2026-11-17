# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

# Dados: 30 operadores executam as mesmas 4 tarefas, de dificuldade crescente (dados fictícios).
# sucesso = 1 se a tarefa foi concluída no tempo previsto | treinamento = 1 se o operador fez o curso
treino_operador = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0,
                            1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0.])
sucesso = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                    1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
                    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1.])
G, T = 30, 4
operador    = np.repeat(np.arange(1, G + 1), T)
treinamento = np.repeat(treino_operador, T)
dificuldade = np.tile(np.arange(1, T + 1), G)
X = np.column_stack([np.ones(G * T), treinamento, dificuldade - 2.5])
nomes = ["(Intercepto)", "treinamento", "dificuldade"]

def imprime(nomes, b, ep, rotulo):
    z = b / ep; p = 2 * stats.norm.sf(np.abs(z)); zc = stats.norm.ppf(0.975)
    print(f"  {rotulo}")
    print(f"  {'Termo':<14}{'Estimativa':>11}{'Erro-padrão':>13}{'z':>8}{'p-valor':>10}{'OR':>8}{'IC 95% da OR':>22}")
    for nm, bi, ei, zi, pi in zip(nomes, b, ep, z, p):
        print(f"  {nm:<14}{bi:>11.4f}{ei:>13.4f}{zi:>8.3f}{pi:>10.4f}{np.exp(bi):>8.3f}   [{np.exp(bi-zc*ei):>6.3f}; {np.exp(bi+zc*ei):>6.3f}]")

def logistica(X, y):
    """Regressão logística comum, que supõe todas as observações independentes."""
    b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        b = b + np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b))
    return b, np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))

def gee(X, y, grupo, estrutura="permutavel"):
    """Equações de estimação generalizadas, resposta binária, com variância robusta (sanduíche)."""
    b, _ = logistica(X, y); grupos = np.unique(grupo); n, p = X.shape
    for _ in range(80):
        mu = 1 / (1 + np.exp(-X @ b))
        r = (y - mu) / np.sqrt(mu * (1 - mu))              # resíduos de Pearson
        phi = np.sum(r**2) / (n - p)
        if estrutura == "permutavel":                       # correlação igual entre todos os pares
            soma = 0.0; pares = 0
            for g in grupos:
                rr = r[grupo == g]
                for a in range(len(rr)):
                    for c in range(a + 1, len(rr)): soma += rr[a] * rr[c]; pares += 1
            alpha = soma / (pares * phi)
        else:
            alpha = 0.0                                     # independência
        U = np.zeros(p); H = np.zeros((p, p)); M = np.zeros((p, p))
        for g in grupos:
            i = np.where(grupo == g)[0]; m = len(i)
            Xi = X[i]; mui = mu[i]; A = mui * (1 - mui)
            R = np.full((m, m), alpha); np.fill_diagonal(R, 1.0)
            V = phi * np.outer(np.sqrt(A), np.sqrt(A)) * R
            Vinv = np.linalg.inv(V); D = Xi * A[:, None]; res = y[i] - mui
            U += D.T @ Vinv @ res; H += D.T @ Vinv @ D
            M += D.T @ Vinv @ np.outer(res, res) @ Vinv @ D
        passo = np.linalg.solve(H, U); b = b + passo
        if np.max(np.abs(passo)) < 1e-10: break
    Hinv = np.linalg.inv(H)
    return b, np.sqrt(np.diag(Hinv @ M @ Hinv)), np.sqrt(np.diag(Hinv)), alpha, phi

# 1. Descrição
print(f"{G} operadores x {T} tarefas = {G*T} observações | taxa geral de sucesso = {sucesso.mean():.3f}")
print(f"Com treinamento: {sucesso[treinamento==1].mean():.3f} | sem treinamento: {sucesso[treinamento==0].mean():.3f}")
print("Taxa por dificuldade:", [round(sucesso[dificuldade == d].mean(), 3) for d in range(1, 5)])

# 2. Análise ingênua: regressão logística que ignora o agrupamento
b_n, ep_n = logistica(X, sucesso)
print(); imprime(nomes, b_n, ep_n, "Logística comum (supõe independência):")

# 3. GEE com correlação permutável (exchangeable)
b_g, ep_rob, ep_mod, alpha, phi = gee(X, sucesso, operador)
print(f"\n  Correlação permutável estimada: alpha = {alpha:.4f} | parâmetro de escala: phi = {phi:.4f}")
imprime(nomes, b_g, ep_rob, "GEE permutável, erro-padrão robusto:")

# 4. O mesmo GEE com erro-padrão baseado no modelo (sem a correção robusta)
print(); imprime(nomes, b_g, ep_mod, "GEE permutável, erro-padrão do modelo:")

# 5. GEE com estrutura de independência e erro-padrão robusto
b_i, ep_i, _, _, _ = gee(X, sucesso, operador, estrutura="independencia")
print(); imprime(nomes, b_i, ep_i, "GEE independência, erro-padrão robusto:")

# 6. Comparação dos erros-padrão
print("\nTermo            Logística comum   GEE robusto    Razão")
for nm, e1, e2 in zip(nomes, ep_n, ep_rob):
    print(f"{nm:<16}{e1:>14.4f}{e2:>14.4f}{e2/e1:>9.2f}")
print("treinamento varia ENTRE operadores: a logística comum subestima o erro-padrão.")
print("dificuldade varia DENTRO de cada operador: a logística comum o superestima.")

# 7. Correlação dentro do operador, observada diretamente
m = sucesso.reshape(G, T)
print(f"\nCorrelação média entre pares de tarefas do mesmo operador: {np.mean([np.corrcoef(m[:, a], m[:, c])[0,1] for a in range(T) for c in range(a+1, T)]):.4f}")
print(f"Operadores com 0, 1, 2, 3 e 4 sucessos: {np.bincount(m.sum(axis=1).astype(int), minlength=5)}")

# 8. Comparação com o modelo misto (efeito aleatório por operador)
#    O GEE estima o efeito MÉDIO NA POPULAÇÃO; o modelo misto, o efeito PARA UM MESMO operador.
nos, pesos = np.polynomial.hermite_e.hermegauss(31)
def menos_logv_misto(par):
    b = par[:3]; sigma = np.exp(par[3]); total = 0.0
    for g in range(1, G + 1):
        i = operador == g; eta = X[i] @ b; yi = sucesso[i]
        lp = np.array([np.sum(yi * (eta + sigma * v) - np.log1p(np.exp(eta + sigma * v))) for v in nos])
        m = lp.max(); total += m + np.log(np.sum(pesos * np.exp(lp - m)) / np.sqrt(2 * np.pi))
    return -total
res = optimize.minimize(menos_logv_misto, [0.1, 1.0, -0.7, np.log(1.5)], method="Nelder-Mead",
                        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
bm = res.x[:3]; sigma = np.exp(res.x[3])
print(f"\nModelo misto (quadratura de Gauss-Hermite): sigma do operador = {sigma:.4f}")
print(f"{'Termo':<16}{'GEE (populacional)':>20}{'Misto (por operador)':>22}")
for nm, g1, g2 in zip(nomes, b_g, bm):
    print(f"{nm:<16}{g1:>20.4f}{g2:>22.4f}")
print(f"Atenuação esperada: {bm[1]:.4f} / raiz(1 + 0,346 x {sigma:.3f}²) = {bm[1]/np.sqrt(1+0.346*sigma**2):.4f}"
      f" (GEE: {b_g[1]:.4f})")

# 9. Gráfico
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t in range(1, 5):
    ax[0].bar([t - 0.2, t + 0.2], [sucesso[(dificuldade == t) & (treinamento == 0)].mean(),
                                   sucesso[(dificuldade == t) & (treinamento == 1)].mean()],
              width=0.38, color=["gray", "darkgreen"])
ax[0].set_xticks(range(1, 5)); ax[0].set_xlabel("Dificuldade da tarefa"); ax[0].set_ylabel("Proporção de sucesso")
ax[1].bar(np.arange(3) - 0.2, ep_n, width=0.38, label="logística comum")
ax[1].bar(np.arange(3) + 0.2, ep_rob, width=0.38, label="GEE robusto")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(nomes, rotation=15); ax[1].set_ylabel("Erro-padrão"); ax[1].legend()
plt.show()
