import numpy as np
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (cab14)
# ------------------------------------------------------------------
import numpy as np, itertools
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (dados14)
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
def le_csv(nome):
    """Lê o arquivo de dados da ficha: primeiro na pasta atual; se não
    estiver lá (por exemplo, no Google Colab), direto do site."""
    try:
        return pd.read_csv(nome)
    except FileNotFoundError:
        return pd.read_csv('https://cassiushenrique.github.io/ModelosEstatisticos/scripts/' + nome)

# Dados fictícios: 90 amostras de solo (arquivo solo.csv, pasta scripts/ do site)
solo = le_csv('solo.csv')
NOMES=['ph', 'mo', 'pres', 'k', 'ca', 'mg', 'al', 'ctc', 'v', 'argila', 'silte', 'areia']
ROT={'ph':'pH','mo':'M.O.','pres':'P (resina)','k':'K','ca':'Ca','mg':'Mg','al':'Al','ctc':'CTC','v':'V%','argila':'Argila','silte':'Silte','areia':'Areia'}
X=solo[NOMES].to_numpy(float); Y=solo['y'].to_numpy(float)
VERD={"mo":0.80,"pres":0.0250,"v":0.0300,"argila":0.0200}
N,P=X.shape
# ------------------------------------------------------------------ fim do bloco comum

def pad(A, mu=None, sd=None):
    """Padroniza colunas (média 0, desvio 1, com n-1)."""
    if mu is None: mu = A.mean(0)
    if sd is None: sd = A.std(0, ddof=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (A - mu)/sd, mu, sd

def ols(A, y):
    Ad = np.column_stack([np.ones(len(A)), A])
    return np.linalg.lstsq(Ad, y, rcond=None)[0]

def prev(A, b):
    return b[0] + A @ b[1:]

def sqr(obs, p_): return float(np.sum((obs - p_)**2))
def eqm(obs, p_): return float(np.mean((obs - p_)**2))

def ridge(A, y, lam):
    """Ridge em A JÁ padronizado; não penaliza o intercepto."""
    n_, p_ = A.shape
    G = A.T @ A + lam*n_*np.eye(p_)
    b = np.linalg.solve(G, A.T @ (y - y.mean()))
    return np.concatenate([[y.mean()], b])

def lasso(A, y, lam, it=20000, tol=1e-11):
    """Descida por coordenadas em A JÁ padronizado (Friedman et al., 2010)."""
    n_, p_ = A.shape
    b = np.zeros(p_); r = y - y.mean()
    nrm = (A**2).sum(0)/n_
    for _ in range(it):
        mx = 0.0
        for j in range(p_):
            bj = b[j]
            rho = A[:, j] @ (r + A[:, j]*bj)/n_
            nb = np.sign(rho)*max(abs(rho) - lam, 0.0)/nrm[j]
            if nb != bj:
                r = r - A[:, j]*(nb - bj); b[j] = nb
                mx = max(mx, abs(nb - bj))
        if mx < tol: break
    return np.concatenate([[y.mean()], b])

def pcr(A, y, m):
    """Regressão nos m primeiros componentes principais de A padronizado."""
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    T = (u*s)[:, :m]
    g = np.linalg.lstsq(T, y - y.mean(), rcond=None)[0]
    b = vt[:m].T @ g
    return np.concatenate([[y.mean()], b])

def pls(A, y, m):
    """NIPALS para uma resposta; devolve coeficientes na escala de A."""
    Xr = A.copy(); yr = y - y.mean()
    W = np.zeros((A.shape[1], m)); Pm = np.zeros((A.shape[1], m)); q = np.zeros(m)
    for a in range(m):
        w = Xr.T @ yr; w = w/np.linalg.norm(w)
        t = Xr @ w; tt = t @ t
        pj = Xr.T @ t/tt; qa = (yr @ t)/tt
        Xr = Xr - np.outer(t, pj); yr = yr - qa*t
        W[:, a] = w; Pm[:, a] = pj; q[a] = qa
    Rm = W @ np.linalg.inv(Pm.T @ W)
    b = Rm @ q
    return np.concatenate([[y.mean()], b])

def dobras(n_, k, seed):
    rg = np.random.default_rng(seed)
    d = np.tile(np.arange(k), int(np.ceil(n_/k)))[:n_]
    rg.shuffle(d); return d

def cv(ajusta, A, y, k=10, seed=0, reps=1):
    """EQM por validação cruzada; TODA a padronização dentro do laço."""
    tot = []
    for r in range(reps):
        d = dobras(len(y), k, seed + 1000*r); e = np.empty(len(y))
        for j in range(k):
            tr, te = d != j, d == j
            Ap, mu, sd = pad(A[tr])
            b = ajusta(Ap, y[tr])
            e[te] = y[te] - prev((A[te] - mu)/sd, b)
        tot.append(float(np.mean(e**2)))
    return float(np.mean(tot)), float(np.std(tot, ddof=1)) if reps > 1 else 0.0

def crit(A, y, cols, s2):
    """Cp de Mallows, AIC e BIC de um submodelo."""
    Ad = np.column_stack([np.ones(len(A))] + [A[:, j] for j in cols])
    b = np.linalg.lstsq(Ad, y, rcond=None)[0]
    sq = sqr(y, Ad @ b); n_, d = len(y), len(cols) + 1
    return dict(sqr=sq, p=d, r2=1 - sq/np.sum((y - y.mean())**2),
                r2a=1 - (sq/(n_ - d))/(np.sum((y - y.mean())**2)/(n_ - 1)),
                cp=sq/s2 - n_ + 2*d,
                aic=n_*np.log(sq/n_) + 2*d, bic=n_*np.log(sq/n_) + np.log(n_)*d)

def vif(A):
    Ap, _, _ = pad(A); out = []
    for j in range(A.shape[1]):
        o = [i for i in range(A.shape[1]) if i != j]
        b = ols(Ap[:, o], Ap[:, j]); r = Ap[:, j] - prev(Ap[:, o], b)
        out.append(1/(1 - (1 - r @ r/np.sum((Ap[:, j] - Ap[:, j].mean())**2))))
    return np.array(out)

def c(v, d=4):
    return ('%.*f' % (d, v)).replace('.', ',')

def lasso_path(A, y, lams, it=5000, tol=1e-10):
    """Caminho do lasso com partida quente (como o glmnet). lams DECRESCENTE."""
    n_, p_ = A.shape
    b = np.zeros(p_); r = y - y.mean(); nrm = (A**2).sum(0)/n_
    out = []
    for lam in lams:
        for _ in range(it):
            mx = 0.0
            for j in range(p_):
                bj = b[j]
                rho = A[:, j] @ (r + A[:, j]*bj)/n_
                nb = np.sign(rho)*max(abs(rho) - lam, 0.0)/nrm[j]
                if nb != bj:
                    r = r - A[:, j]*(nb - bj); b[j] = nb
                    mx = max(mx, abs(nb - bj))
            if mx < tol: break
        out.append(np.concatenate([[y.mean()], b.copy()]))
    return out

def enet_path(A, y, lams, al, it=5000, tol=1e-10):
    n_, p_ = A.shape
    b = np.zeros(p_); r = y - y.mean(); nrm = (A**2).sum(0)/n_
    out = []
    for lam in lams:
        for _ in range(it):
            mx = 0.0
            for j in range(p_):
                bj = b[j]
                rho = A[:, j] @ (r + A[:, j]*bj)/n_
                nb = np.sign(rho)*max(abs(rho) - lam*al, 0.0)/(nrm[j] + lam*(1 - al))
                if nb != bj:
                    r = r - A[:, j]*(nb - bj); b[j] = nb
                    mx = max(mx, abs(nb - bj))
            if mx < tol: break
        out.append(np.concatenate([[y.mean()], b.copy()]))
    return out

def cv_path(ajusta_path, A, y, lams, k=10, seed=0, reps=1):
    """EQM por VC para TODA a grade de lambda de uma vez, com partida quente."""
    acc = np.zeros((reps, len(lams)))
    for rp in range(reps):
        d = dobras(len(y), k, seed + 1000*rp); E = np.zeros((len(lams), len(y)))
        for j in range(k):
            tr, te = d != j, d == j
            Ap, mu, sd = pad(A[tr])
            bs = ajusta_path(Ap, y[tr], lams)
            Ate = (A[te] - mu)/sd
            for i, b in enumerate(bs): E[i, te] = y[te] - prev(Ate, b)
        acc[rp] = (E**2).mean(1)
    return acc.mean(0), acc.std(0, ddof=1) if reps > 1 else np.zeros(len(lams))


def _cd_gram(G, Ay, lam, al, b, dg, it=10000, tol=1e-11):
    """Descida por coordenadas usando a matriz de Gram: só escalares no laço."""
    p_ = len(b)
    for _ in range(it):
        mx = 0.0
        for j in range(p_):
            bj = b[j]
            rho = Ay[j] - (G[j] @ b - dg[j]*bj)
            nb = np.sign(rho)*max(abs(rho) - lam*al, 0.0)/(dg[j] + lam*(1 - al))
            if nb != bj:
                b[j] = nb; mx = max(mx, abs(nb - bj))
        if mx < tol: break
    return b

def enet_path_g(A, y, lams, al=1.0):
    """Caminho da rede elástica (al=1 é o lasso) com Gram e partida quente."""
    n_, p_ = A.shape
    G = A.T @ A/n_; Ay = A.T @ (y - y.mean())/n_; dg = np.diag(G).copy()
    b = np.zeros(p_); out = []
    for lam in lams:
        b = _cd_gram(G, Ay, lam, al, b, dg)
        out.append(np.concatenate([[y.mean()], b.copy()]))
    return out

def lasso_path_g(A, y, lams): return enet_path_g(A, y, lams, 1.0)


def _cd_ativo(G, Ay, lam, al, b, dg, it=200, tol=1e-11):
    """Descida por coordenadas com CONJUNTO ATIVO (estratégia do glmnet):
    varre tudo, depois itera só nos coeficientes não nulos."""
    p_ = len(b)
    for _ in range(it):
        # varredura completa
        mx = 0.0
        for j in range(p_):
            bj = b[j]
            rho = Ay[j] - (G[j] @ b - dg[j]*bj)
            nb = np.sign(rho)*max(abs(rho) - lam*al, 0.0)/(dg[j] + lam*(1 - al))
            if nb != bj:
                b[j] = nb; mx = max(mx, abs(nb - bj))
        ativo = np.flatnonzero(b)
        if len(ativo) == 0 or mx < tol: break
        Ga = G[np.ix_(ativo, ativo)]; Aya = Ay[ativo]; dga = dg[ativo]
        ba = b[ativo].copy()
        for _ in range(2000):
            mxa = 0.0
            for i in range(len(ativo)):
                bi = ba[i]
                rho = Aya[i] - (Ga[i] @ ba - dga[i]*bi)
                nb = np.sign(rho)*max(abs(rho) - lam*al, 0.0)/(dga[i] + lam*(1 - al))
                if nb != bi:
                    ba[i] = nb; mxa = max(mxa, abs(nb - bi))
            if mxa < tol: break
        b[ativo] = ba
        if mx < tol: break
    return b

def enet_path_a(A, y, lams, al=1.0):
    n_, p_ = A.shape
    G = A.T @ A/n_; Ay = A.T @ (y - y.mean())/n_; dg = np.diag(G).copy()
    b = np.zeros(p_); out = []
    for lam in lams:
        b = _cd_ativo(G, Ay, lam, al, b, dg)
        out.append(np.concatenate([[y.mean()], b.copy()]))
    return out

def lasso_path_a(A, y, lams): return enet_path_a(A, y, lams, 1.0)
# ------------------------------------------------------------------ fim do bloco comum
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (dados14b)
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
def le_csv(nome):
    """Lê o arquivo de dados da ficha: primeiro na pasta atual; se não
    estiver lá (por exemplo, no Google Colab), direto do site."""
    try:
        return pd.read_csv(nome)
    except FileNotFoundError:
        return pd.read_csv('https://cassiushenrique.github.io/ModelosEstatisticos/scripts/' + nome)

# Dados fictícios: 60 espectros NIR de 200 bandas (arquivo espectro.csv)
esp = le_csv('espectro.csv')
LAM=np.round(np.linspace(1100,2300,200),1)          # comprimentos de onda (nm)
XE=esp.drop(columns='proteina').to_numpy(float); YE=esp['proteina'].to_numpy(float)
NE,PE=XE.shape
# ------------------------------------------------------------------ fim do bloco comum

Xp,mu,sd=pad(X); yc=Y-Y.mean()
Ep,emu,esd=pad(XE); yec=YE-YE.mean()

print('Dois conjuntos nesta ficha, porque a PCR se comporta de modo muito')
print('diferente nos dois:')
print('  (a) SOLO:     90 amostras, 12 candidatas colineares')
print('  (b) ESPECTRO: 60 amostras, 200 comprimentos de onda (p > n)')

print('\n'+'='*70)
print('(a) SOLO — os 12 componentes principais')
u,s,vt=np.linalg.svd(Xp,full_matrices=False)
T=u*s; pv=s**2/np.sum(s**2)
print()
print('%3s %10s %12s %12s %14s %10s'%('CP','d_j','% var. X','% acum.','|r(CP, y)|','VIF do CP'))
for j in range(P):
    print('%3d %10s %12s %12s %14s %10s'%(j+1,c(float(s[j]),3),c(100*pv[j],2),
          c(100*pv[:j+1].sum(),2),c(abs(float(np.corrcoef(T[:,j],Y)[0,1]))),c(1.0,2)))
print()
print('Os componentes são ORTOGONAIS entre si: todo VIF é exatamente 1. É a')
print('primeira virtude do método — a colinearidade desaparece por construção.')
print()
ordem=np.argsort(-np.abs([np.corrcoef(T[:,j],Y)[0,1] for j in range(P)]))
print('Ordenados por VARIÂNCIA em X:   %s'%', '.join('CP%d'%(j+1) for j in range(P)))
print('Ordenados por RELEVÂNCIA para y: %s'%', '.join('CP%d'%(j+1) for j in ordem))
print()
print('Nos dados de solo as duas ordens quase coincidem no começo, e a PCR')
print('funciona bem. No espectro elas são MUITO diferentes.')

print('\n'+'='*70)
print('(a) Quantos componentes? Validação cruzada (10 dobras, 20 repetições)')
print()
print('%3s %12s %12s %12s %10s'%('m','EQM resub','EQM 10-fold','% var. X','R² val.'))
best=None
for m in range(1,P+1):
    b=pcr(Xp,Y,m); rs=eqm(Y,prev(Xp,b))
    cvm,_=cv(lambda A,y,m=m: pcr(A,y,m),X,Y,10,0,20)
    if best is None or cvm<best[1]: best=(m,cvm)
    print('%3d %12s %12s %12s %10s'%(m,c(rs),c(cvm),c(100*pv[:m].sum(),2),
          c(1-cvm/float(Y.var(ddof=1)))))
print()
print('  melhor m = %d, EQM validado = %s'%(best[0],c(best[1])))
print('  EQM validado do MQO com as 12 = %s'%c(cv(ols,X,Y,10,0,20)[0]))
bo=ols(Xp,Y); bp=pcr(Xp,Y,P)
print('  PCR com m = p é EXATAMENTE o MQO. Maior diferença entre os coeficientes:')
print('    %s'%c(float(np.abs(bo-bp).max()),10))

print('\n'+'='*70)
print('A REGRA DOS "80% DA VARIÂNCIA" É UM MAU CRITÉRIO')
print()
for alvo in [0.70,0.80,0.90,0.95]:
    m=int(np.searchsorted(np.cumsum(pv),alvo)+1)
    cvm,_=cv(lambda A,y,m=m: pcr(A,y,m),X,Y,10,0,20)
    print('  %d%% da variância de X -> m = %2d -> EQM validado %s'%(100*alvo,m,c(cvm)))
print('  escolha por validação cruzada -> m = %2d -> EQM validado %s  <- melhor'%(best[0],c(best[1])))
print()
print('O critério da variância explicada só olha X e ignora y por completo.')
print('A validação cruzada olha o que importa: o erro de predição.')

print('\n'+'='*70)
print('(b) ESPECTRO — aqui a ordem de variância ENGANA')
eu,es,evt=np.linalg.svd(Ep,full_matrices=False)
TE=eu*es; epv=es**2/np.sum(es**2)
print()
print('%3s %12s %12s %14s'%('CP','% var. X','% acum.','|r(CP, y)|'))
for j in range(10):
    print('%3d %12s %12s %14s'%(j+1,c(100*epv[j],2),c(100*epv[:j+1].sum(),2),
          c(abs(float(np.corrcoef(TE[:,j],YE)[0,1])))))
print()
print('O CP1 explica %s%% da variância do espectro e tem correlação de apenas'%c(100*epv[0],2))
print('%s com a proteína. O CP3 explica %s%% e tem correlação de %s.'%(
      c(abs(float(np.corrcoef(TE[:,0],YE)[0,1]))),c(100*epv[2],2),
      c(abs(float(np.corrcoef(TE[:,2],YE)[0,1])))))
print()
print('A razão é física: a maior fonte de variação de um espectro de')
print('reflectância é a deriva da linha de base (espalhamento, tamanho de')
print('partícula), que nada tem a ver com o teor de proteína. A PCR é obrigada')
print('a gastar o primeiro componente nela.')
print()
print('%3s %12s %12s %10s'%('m','EQM 10-fold','% var. X','R² val.'))
bste=None
for m in range(1,13):
    cvm,_=cv(lambda A,y,m=m: pcr(A,y,m),XE,YE,10,0,5)
    if bste is None or cvm<bste[1]: bste=(m,cvm)
    print('%3d %12s %12s %10s'%(m,c(cvm),c(100*epv[:m].sum(),2),c(1-cvm/float(YE.var(ddof=1)))))
print()
print('  melhor m = %d, EQM validado = %s, R² validado = %s'%(bste[0],c(bste[1]),
      c(1-bste[1]/float(YE.var(ddof=1)))))
print()
print('Quantos componentes a PCR precisa para chegar ao erro que a PLS alcança')
print('com 3? (a comparação completa está na ficha pls)')
_pls = pls
cvpls3,_=cv(lambda A,y: _pls(A,y,3),XE,YE,10,0,5)
print('  PLS com 3 componentes: EQM validado = %s'%c(cvpls3))
mm=[m for m in range(1,21) if cv(lambda A,y,m=m: pcr(A,y,m),XE,YE,10,0,5)[0]<=cvpls3]
print('  PCR precisa de m = %s para igualar'%(mm[0] if mm else 'mais de 20'))

print('\n'+'='*70)
print('PCR É UMA VERSÃO DISCRETA DO RIDGE')
print()
print('O ridge multiplica o componente j pelo fator d_j²/(d_j² + n*lambda), que')
print('é contínuo entre 0 e 1. A PCR multiplica por 1 (se j <= m) ou 0 (se j > m).')
print()
l0=0.03162
print('%3s %12s %16s %12s'%('CP','d_j','ridge (lam=0,032)','PCR (m=5)'))
for j in range(P):
    print('%3s %12s %16s %12s'%(j+1,c(float(s[j]),3),
          c(float(s[j]**2/(s[j]**2+l0*N))),'1,0000' if j<5 else '0,0000'))
print()
print('É por isso que os dois métodos dão resultados parecidos quando a')
print('colinearidade é moderada, e por isso que o ridge é geralmente preferível:')
print('o corte abrupto da PCR descarta informação útil junto com o ruído.')

print('\n'+'='*70)
print('A PCR NÃO SELECIONA VARIÁVEIS')
b=pcr(Xp,Y,best[0])
print()
print('Com m = %d, os coeficientes voltam para a escala das variáveis originais'%best[0])
print('(b = V_m * gamma), e TODAS as 12 recebem coeficiente não nulo:')
for j in np.argsort(-np.abs(b[1:])):
    print('  %-11s %10s   %s'%(ROT[NOMES[j]],c(b[1+j]),'(no modelo verdadeiro)' if NOMES[j] in VERD else ''))
print()
print('  coeficientes exatamente zero: %d de 12'%int(np.sum(np.abs(b[1:])<1e-12)))
print()
print('A PCR reduz a DIMENSÃO, não o número de variáveis medidas. Para análise')
print('de solo isso importa: continua sendo preciso medir as doze.')

print('\n'+'='*70)
print('AS CARGAS: o que cada componente significa')
print()
print('Cargas dos quatro primeiros componentes (solo):')
print('%-11s %9s %9s %9s %9s'%('variável','CP1','CP2','CP3','CP4'))
for j in range(P):
    print('%-11s %9s %9s %9s %9s'%(ROT[NOMES[j]],c(float(vt[0,j]),3),c(float(vt[1,j]),3),
          c(float(vt[2,j]),3),c(float(vt[3,j]),3)))
print()
print('%-11s %9s %9s %9s %9s'%('% var. X',c(100*pv[0],1),c(100*pv[1],1),c(100*pv[2],1),c(100*pv[3],1)))
print('%-11s %9s %9s %9s %9s'%('|r| com y',
      c(abs(float(np.corrcoef(T[:,0],Y)[0,1])),3),c(abs(float(np.corrcoef(T[:,1],Y)[0,1])),3),
      c(abs(float(np.corrcoef(T[:,2],Y)[0,1])),3),c(abs(float(np.corrcoef(T[:,3],Y)[0,1])),3)))
print()
print('O CP1 opõe o complexo de troca (Ca, Mg, K, V%) à areia: é o gradiente de')
print('fertilidade química. O CP2 carrega a textura. A leitura das cargas é')
print('interpretação, não resultado estatístico, e deve ser apresentada como tal.')
