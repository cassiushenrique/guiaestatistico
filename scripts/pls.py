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
vye=float(YE.var(ddof=1)); vy=float(Y.var(ddof=1))

print('A PLS foi criada para o problema do ESPECTRO: muitas variáveis')
print('fortemente correlacionadas, poucas amostras, e a maior fonte de variação')
print('das variáveis sem relação com a resposta. Os dois conjuntos da ficha pcr')
print('reaparecem aqui, e o contraste entre eles é o argumento central.')

print('\n'+'='*70)
print('O ALGORITMO NIPALS, o primeiro componente passo a passo (espectro)')
print()
Xr=Ep.copy(); yr=yec.copy()
w=Xr.T@yr
print("  w1 = X'y (direção de maior COVARIÂNCIA com y): norma %s"%c(float(np.linalg.norm(w)),2))
w=w/np.linalg.norm(w)
t=Xr@w; tt=float(t@t); pj=Xr.T@t/tt; q=float((yr@t)/tt)
print('  ||w1|| = %s, depois de normalizar'%c(float(np.linalg.norm(w))))
print('  t1 = X w1 (o escore): variância = %s'%c(float(t.var(ddof=1))))
print("  p1 = X't1/t1't1 (a carga): norma %s"%c(float(np.linalg.norm(pj))))
print("  q1 = y't1/t1't1 = %s"%c(q))
print('  r(t1, y) = %s'%c(float(np.corrcoef(t,yec)[0,1])))
print('  %% da variância de X capturada por t1 = %s'%c(100*float(tt*(pj@pj))/float(np.sum(Ep**2)),2))
print('  %% da variância de y capturada por t1 = %s'%c(100*q*q*tt/float(yec@yec),2))
print()
print("Em seguida X e y são DEFLACIONADOS (X <- X - t1 p1', y <- y - q1 t1) e o")
print('passo se repete nos resíduos, o que garante escores ortogonais.')
print()
eu,es,evt=np.linalg.svd(Ep,full_matrices=False); t1p=(eu*es)[:,0]
print('Para comparar, o primeiro componente PRINCIPAL do mesmo espectro:')
print('%-34s %12s %12s'%('','PCR (CP1)','PLS (t1)'))
print('%-34s %12s %12s'%('% da variância de X',c(100*float(es[0]**2/np.sum(es**2)),2),
      c(100*float(tt*(pj@pj))/float(np.sum(Ep**2)),2)))
print('%-34s %12s %12s'%('|r| com a proteína',c(abs(float(np.corrcoef(t1p,YE)[0,1]))),
      c(abs(float(np.corrcoef(t,YE)[0,1])))))
print()
print('A PLS abre mão de variância em X para ganhar correlação com y. É a única')
print('diferença entre os dois métodos, e explica tudo o que vem depois.')

print('\n'+'='*70)
print('(a) ESPECTRO: variância explicada componente por componente')
def decomp(A,y,m):
    Xr=A.copy(); yr=y-y.mean(); vx=[];vy_=[]
    for a in range(m):
        w=Xr.T@yr; w=w/np.linalg.norm(w); t=Xr@w; tt=t@t
        pp=Xr.T@t/tt; qa=(yr@t)/tt
        vx.append(float(tt*(pp@pp)/np.sum(A**2))); vy_.append(float(qa*qa*tt/np.sum((y-y.mean())**2)))
        Xr=Xr-np.outer(t,pp); yr=yr-qa*t
    return np.array(vx),np.array(vy_)
vx,vyp=decomp(Ep,YE,10)
epv=es**2/np.sum(es**2)
r2pcr=[1-sqr(YE,prev(Ep,pcr(Ep,YE,m)))/float(np.sum((YE-YE.mean())**2)) for m in range(11)]
print()
print('%4s %26s %26s'%('','PLS','PCR'))
print('%4s %12s %13s %12s %13s'%('m','% var X','% var y','% var X','% var y'))
for m in range(10):
    print('%4d %12s %13s %12s %13s'%(m+1,c(100*vx[m],2),c(100*vyp[m],2),
          c(100*epv[m],2),c(100*(r2pcr[m+1]-r2pcr[m]),2)))
print('%4s %12s %13s %12s %13s'%('acum',c(100*vx.sum(),2),c(100*vyp.sum(),2),
      c(100*epv[:10].sum(),2),c(100*r2pcr[10],2)))
print()
print('A PLS põe %s%% da variância de y nos TRÊS primeiros componentes; a PCR,'%c(100*vyp[:3].sum(),2))
print('nos três primeiros, explica %s%%. Em troca, a PCR captura mais'%c(100*r2pcr[3],2))
print('variância de X (%s%% contra %s%%), que é o que ela otimiza.'%(
      c(100*epv[:3].sum(),2),c(100*vx[:3].sum(),2)))

print('\n'+'='*70)
print('(b) ESPECTRO: a validação cruzada decide (10 dobras, 5 repetições)')
print()
print('%4s %14s %14s %13s %13s'%('m','EQM PLS','EQM PCR','R² val PLS','R² val PCR'))
bp=bq=None; cvp={};cvq={}
for m in range(1,13):
    a_,_=cv(lambda A,y,m=m: pls(A,y,m),XE,YE,10,0,5)
    b_,_=cv(lambda A,y,m=m: pcr(A,y,m),XE,YE,10,0,5)
    cvp[m]=a_; cvq[m]=b_
    if bp is None or a_<bp[1]: bp=(m,a_)
    if bq is None or b_<bq[1]: bq=(m,b_)
    print('%4d %14s %14s %13s %13s'%(m,c(a_),c(b_),c(1-a_/vye),c(1-b_/vye)))
print()
print('  melhor PLS: m = %d, EQM %s, R² val %s'%(bp[0],c(bp[1]),c(1-bp[1]/vye)))
print('  melhor PCR: m = %d, EQM %s, R² val %s'%(bq[0],c(bq[1]),c(1-bq[1]/vye)))
print()
cvq2={m:cv(lambda A,y,m=m: pcr(A,y,m),XE,YE,10,0,5)[0] for m in range(1,21)}
print('Quantos componentes a PCR precisa para igualar a PLS?')
print('%-28s %14s %s'%('PLS','EQM','PCR equivalente'))
for m in [1,2,3,4,5]:
    k=[kk for kk in range(1,21) if cvq2[kk]<=cvp[m]]
    print('%-28s %14s %s'%('m = %d'%m,c(cvp[m]),('m = %d'%k[0]) if k else 'mais de 20'))
print()
print('A leitura exige cuidado, porque o resultado tem DUAS partes. No MELHOR')
print('de cada método os dois praticamente EMPATAM: PLS com m = %d dá %s e'%(bp[0],c(bp[1])))
print('PCR com m = %d dá %s — diferença de %s, dentro do erro de simulação.'%(bq[0],c(bq[1]),c(abs(bp[1]-bq[1]))))
print('A PLS NÃO alcança um erro menor que a PCR.')
print()
print('A vantagem da PLS está nos componentes INICIAIS, e é grande: com m = 3')
print('ela já tem R² validado de %s, enquanto a PCR com m = 3 tem %s e'%(c(1-cvp[3]/vye),c(1-cvq[3]/vye)))
print('precisa de m = 6 para igualar. Com m = 2 a PLS tem %s contra %s.'%(c(1-cvp[2]/vye),c(1-cvq[2]/vye)))
print('Em termos práticos: se o objetivo é um modelo PARCIMONIOSO em')
print('componentes — para interpretar, para calibrar um instrumento, para')
print('transferir entre equipamentos — a PLS chega lá com metade deles. Se o')
print('objetivo é apenas o menor erro possível, os dois servem igualmente.')

print('\n'+'='*70)
print('(c) SOLO: e quando a colinearidade é apenas moderada?')
print()
mq,_=cv(ols,X,Y,10,0,10); mr,_=cv(lambda A,y: ridge(A,y,0.03162),X,Y,10,0,10)
print('%4s %14s %14s'%('m','EQM PLS','EQM PCR'))
bs=bt=None
for m in range(1,P+1):
    a_,_=cv(lambda A,y,m=m: pls(A,y,m),X,Y,10,0,10)
    b_,_=cv(lambda A,y,m=m: pcr(A,y,m),X,Y,10,0,10)
    if bs is None or a_<bs[1]: bs=(m,a_)
    if bt is None or b_<bt[1]: bt=(m,b_)
    print('%4d %14s %14s'%(m,c(a_),c(b_)))
print()
print('  melhor PLS: m = %d, EQM %s'%(bs[0],c(bs[1])))
print('  melhor PCR: m = %d, EQM %s'%(bt[0],c(bt[1])))
print('  MQO com as 12:  EQM %s'%c(mq))
print('  ridge por VC:   EQM %s'%c(mr))
print()
print('Com p = 12 e n = 90 a PLS não tem vantagem: o MQO já funciona, e tanto a')
print('PLS como a PCR precisam de quase todos os componentes para igualá-lo.')
print('A PLS rende quando p é grande e as colunas são muito correlacionadas.')

print('\n'+'='*70)
print('PLS COM m = posto(X) É O MQO')
print()
r_=int(np.sum(np.linalg.svd(Xp,compute_uv=False)>1e-8))
bo=ols(Xp,Y); bl=pls(Xp,Y,r_); bc=pcr(Xp,Y,r_)
print('  posto de X (solo) = %d'%r_)
print('  maior diferença entre PLS(m=%d) e MQO: %s'%(r_,c(float(np.abs(bo-bl).max()),10)))
print('  maior diferença entre PCR(m=%d) e MQO: %s'%(r_,c(float(np.abs(bo-bc).max()),10)))
print()
print('A PLS e a PCR são CAMINHOS entre o modelo nulo (m = 0) e o MQO')
print('(m = posto). Escolher m é escolher o grau de regularização.')

print('\n'+'='*70)
print('OS PESOS DA PLS: quais comprimentos de onda importam')
print()
print('Duas quantidades competem para responder a essa pergunta: os')
print('COEFICIENTES da regressão e o VIP (importância na projeção), que é a')
print('convenção da quimiometria. Neste espectro sabe-se a resposta correta —')
print('as bandas em 1.350 e 1.920 nm determinam a proteína; as em 1.680 e')
print('2.180 nm não — e é possível verificar qual das duas acerta.')
print()
def pls_full(A,y,m):
    Xr=A.copy(); yr=y-y.mean()
    W=np.zeros((A.shape[1],m));Pm=np.zeros((A.shape[1],m));q=np.zeros(m);Tt=np.zeros((len(y),m))
    for a in range(m):
        w=Xr.T@yr; w=w/np.linalg.norm(w); t=Xr@w; tt=t@t
        pp=Xr.T@t/tt; qa=(yr@t)/tt
        Xr=Xr-np.outer(t,pp); yr=yr-qa*t
        W[:,a]=w;Pm[:,a]=pp;q[a]=qa;Tt[:,a]=t
    return W,Pm,q,Tt
M=bp[0]
W,Pm,q,Tt=pls_full(Ep,YE,M)
ss=np.array([q[a]**2*(Tt[:,a]@Tt[:,a]) for a in range(M)])
VIP=np.sqrt(PE*np.sum(ss[None,:]*(W**2),axis=1)/ss.sum())
bfin=pls(Ep,YE,M); ab=np.abs(bfin[1:])
REG=[(1100,1260,'1.100-1.260','não'),(1260,1440,'1.350','SIM'),
     (1440,1560,'1.500 (linha de base)','não'),(1560,1800,'1.680','não'),
     (1800,2040,'1.920','SIM'),(2040,2300,'2.180','não')]
print('Com m = %d, região por região:'%M)
print('%-24s %6s %9s %9s %8s %9s  %s'%('região (nm)','faixas','VIP médio','VIP máx.','VIP > 1','Σ|coef|','determina y?'))
for lo,hi,rot,det in REG:
    mk=(LAM>=lo)&(LAM<hi)
    print('%-24s %6d %9s %9s %8d %9s  %s'%(rot,int(mk.sum()),c(float(VIP[mk].mean()),3),
          c(float(VIP[mk].max()),3),int((VIP[mk]>1).sum()),c(float(ab[mk].sum()),3),det))
print()
print('  faixas com VIP > 1: %d de %d'%(int(np.sum(VIP>1)),PE))
print('  VIP quadrático médio = %s (é sempre 1 por construção)'%c(float(np.mean(VIP**2)),4))
print()
print('O resultado contraria a prática corrente, e por isso merece ser dito sem')
print('rodeios: NESTE conjunto, em que a resposta é conhecida, os COEFICIENTES')
print('acertam e o VIP erra.')
print()
print('Os coeficientes somam %s e %s em valor absoluto nas duas bandas que'%(
      c(float(ab[(LAM>=1260)&(LAM<1440)].sum()),3),c(float(ab[(LAM>=1800)&(LAM<2040)].sum()),3)))
print('DETERMINAM a proteína, contra %s e %s nas duas que não determinam —'%(
      c(float(ab[(LAM>=1560)&(LAM<1800)].sum()),3),c(float(ab[(LAM>=2040)&(LAM<2300)].sum()),3)))
print('uma razão de cerca de três para um, na direção correta.')
print()
print('O VIP, ao contrário, aponta %d faixas acima de 1 na região 1.100-1.260 e'%int(((VIP>1)&(LAM>=1100)&(LAM<1260)).sum()))
print('%d na região 2.180, nenhuma das quais determina a resposta — e aponta'%int(((VIP>1)&(LAM>=2040)&(LAM<2300)).sum()))
print('ZERO faixas na região de 1.920, que determina. O VIP máximo ali é %s,'%c(float(VIP[(LAM>=1800)&(LAM<2040)].max()),3))
print('abaixo do limiar. É um falso negativo completo sobre uma banda causal,')
print('acompanhado de dezenas de falsos positivos.')
print()
print('A razão é estrutural. O VIP é construído a partir dos PESOS w de cada')
print('componente, e os pesos se concentram onde o produto variância-por-')
print('covariância é maior. As colunas vizinhas do espectro têm r = 0,998, de')
print('modo que uma região pode servir de procuração para outra: a banda de')
print('1.350 nm, que tem r = %s com a proteína contra %s da banda de 1.920,'%(
      c(float(np.corrcoef(Ep[:,int(np.argmin(np.abs(LAM-1350)))],YE)[0,1])),
      c(float(np.corrcoef(Ep[:,int(np.argmin(np.abs(LAM-1920)))],YE)[0,1]))))
print('absorve o peso das duas. É o mesmo fenômeno da argila e da areia nos')
print('dados de solo, em outra escala: com colunas quase redundantes, não há')
print('como atribuir a importância a uma e não à outra.')
print()
print('A regra "VIP > 1" é uma CONVENÇÃO, não um teste: não há valor-p')
print('associado, e o limiar 1 vem apenas de o VIP quadrático médio ser 1 por')
print('construção. Aqui ela selecionou %d das 200 faixas — quase metade — o que'%int(np.sum(VIP>1)))
print('por si já indica que não está discriminando.')
print()
print('As dez faixas de maior VIP, para registro:')
print('%12s %10s %14s'%('nm','VIP','coef. PLS'))
for j in np.argsort(-VIP)[:10]:
    print('%12s %10s %14s'%('%.0f'%LAM[j],c(float(VIP[j]),3),c(float(bfin[1+j]),4)))

print('\n'+'='*70)
print('A SÍNTESE DO GRUPO, no conjunto em que p > n')
print('Espectro: %d amostras, %d comprimentos de onda.'%(NE,PE))
print()
LME=float(np.abs(Ep.T@yec).max()/NE)
GL=np.logspace(np.log10(LME),np.log10(LME/500),14)
ML,_=cv_path(lasso_path_a,XE,YE,GL,10,0,2)
jl=int(np.argmin(ML)); nal=int(np.sum(np.abs(lasso_path_a(Ep,YE,GL)[jl][1:])>1e-8))
MR=[]
for l in np.logspace(-4,2,25):
    m_,_=cv(lambda A,y,l=l: ridge(A,y,l),XE,YE,10,0,3); MR.append((l,m_))
MR=np.array(MR); jr=int(np.argmin(MR[:,1]))
m0,_=cv(lambda A,y: np.array([y.mean()]+[0.0]*A.shape[1]),XE,YE,10,0,5)
print('%-34s %13s %10s   %s'%('método','EQM validado','R² val.','o que se ajusta'))
for nome,v,obs in [('Modelo nulo (só a média)',m0,'nenhum parâmetro'),
                   ('Ridge, lambda = %s'%c(MR[jr,0],5),MR[jr,1],'200 coeficientes, nenhum zero'),
                   ('Lasso, lambda = %s'%c(GL[jl],5),ML[jl],'%d faixas ativas de 200'%nal),
                   ('PCR, %d componentes'%bq[0],bq[1],'%d componentes'%bq[0]),
                   ('PLS, %d componentes'%bp[0],bp[1],'%d componentes'%bp[0])]:
    print('%-34s %13s %10s   %s'%(nome,c(v),c(1-v/vye),obs))
print()
print('%-34s %13s %10s   %s'%('MQO com as 200 colunas','—','—',"IMPOSSÍVEL: X'X é singular"))
print('%-34s %13s %10s   %s'%('Seleção exaustiva','—','—','IMPOSSÍVEL: 2^200 modelos'))
print('%-34s %13s %10s   %s'%('Seleção para frente','—','—','roda, mas sem controle do erro'))
print()
print('É este o quadro que justifica o grupo inteiro: quando p > n, os métodos')
print('de seleção clássicos não são apenas piores — são INAPLICÁVEIS — e os')
print('quatro métodos regularizados são as únicas opções disponíveis.')
