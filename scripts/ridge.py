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

Xp,mu,sd=pad(X); SQT=float(np.sum((Y-Y.mean())**2))
V=vif(X)
print('Os mesmos 90 perfis de solo e as mesmas 12 candidatas.')
print('O problema desta ficha é a COLINEARIDADE, e ela é severa:')
print()
u,s,vt=np.linalg.svd(Xp,full_matrices=False)
print('  maior VIF (Areia)                 %s'%c(V.max(),1))
print('  número de condição de X           %s'%c(float(s[0]/s[-1]),2))
R_=np.corrcoef(X.T)
_pr=max(((abs(R_[i,j]),i,j) for i in range(P) for j in range(i+1,P)))
print('  maior |r| entre duas candidatas   %s (%s x %s)'%(c(_pr[0]),ROT[NOMES[_pr[1]]],ROT[NOMES[_pr[2]]]))
print('  r(Argila, Silte + Areia)          %s'%c(float(np.corrcoef(X[:,NOMES.index('argila')],X[:,NOMES.index('silte')]+X[:,NOMES.index('areia')])[0,1])))
print('  r(Areia, Argila + Silte)          %s'%c(float(np.corrcoef(X[:,NOMES.index('areia')],X[:,NOMES.index('argila')]+X[:,NOMES.index('silte')])[0,1])))
print('  valores singulares de X, do maior ao menor:')
print('   ',' '.join(c(float(v),3) for v in s))
print()
print('A razão é física: argila + silte + areia = 100% por construção, de modo')
print('que cada fração é quase o complemento das outras duas (r = -0,99); e')
print('Ca + Mg + K estão ligados à CTC e à saturação por bases, com Ca x Mg')
print('chegando a r = 0,90.')

print('\n'+'='*70)
print('A CONSEQUÊNCIA: os coeficientes de mínimos quadrados são instáveis')
b0=ols(Xp,Y)
rg=np.random.default_rng(31)
B=np.zeros((2000,P))
for r in range(2000):
    i=rg.integers(0,N,N); Ab,_,_=pad(X[i]); B[r]=ols(Ab,Y[i])[1:]
print()
print('%-11s %10s %10s %10s %9s   %s'%('variável','MQO','EP boot','VIF','|EP/est|','sinal muda?'))
for j in range(P):
    ep=B[:,j].std(ddof=1)
    pct=100*min(np.mean(B[:,j]>0),np.mean(B[:,j]<0))
    print('%-11s %10s %10s %10s %9s   %4.1f%%'%(ROT[NOMES[j]],c(b0[1+j]),c(ep),
          c(V[j],1),c(abs(ep/b0[1+j]),2) if abs(b0[1+j])>1e-9 else '-',pct))
print()
print('Os coeficientes das três frações texturais e os do complexo de troca')
print('têm erro-padrão da ordem da própria estimativa, e trocam de sinal entre')
print('reamostras. Não é que o modelo não preveja: é que ele não IDENTIFICA')
print('qual das variáveis colineares é responsável.')

print('\n'+'='*70)
print('A SOLUÇÃO RIDGE: encolher os coeficientes para perto de zero')
print()
LAMS=np.logspace(-4,2,61)
PATH=np.array([ridge(Xp,Y,l)[1:] for l in LAMS])
GL=np.array([float(np.sum(s**2/(s**2+l*N))) for l in LAMS])
print('%10s %8s   %s'%('lambda','g.l. ef.','coeficientes (M.O., V%, Argila, Silte, Areia)'))
for l in [0.0,1e-4,1e-3,1e-2,0.05,0.1,0.5,1.0,10.0]:
    b=ridge(Xp,Y,l) if l>0 else ols(Xp,Y)
    g=float(np.sum(s**2/(s**2+l*N))) if l>0 else P
    print('%10s %8s   %s'%(c(l,4),c(g,3),'  '.join(c(b[1+NOMES.index(k)]) for k in ['mo','v','argila','silte','areia'])))
print()
print('Em lambda = 0 o ridge é o MQO. Quando lambda cresce, os coeficientes')
print('encolhem; os das variáveis colineares encolhem PRIMEIRO e mais rápido.')
print('Os graus de liberdade efetivos caem de 12 (MQO) para perto de zero.')

print('\n'+'='*70)
print('O ENCOLHIMENTO, componente por componente')
print()
print('O ridge não encolhe todas as direções igualmente: encolhe cada direção')
print('principal pelo fator d_j^2/(d_j^2 + n*lambda).')
print()
lsel=0.05
print('componente   valor singular d_j   fator de encolhimento (lambda = %s)'%c(lsel,3))
for j in range(P):
    print('%8d   %18s   %s'%(j+1,c(float(s[j]),3),c(float(s[j]**2/(s[j]**2+lsel*N)))))
print()
print('As direções de MAIOR variância em X quase não são tocadas; as de menor')
print('variância — justamente aquelas em que a colinearidade concentra o ruído —')
print('são praticamente eliminadas. É esse o mecanismo.')

print('\n'+'='*70)
print('A ESCOLHA DE lambda POR VALIDAÇÃO CRUZADA (10 dobras, 20 repetições)')
print()
res=[]
for l in LAMS:
    m,dp=cv(lambda A,y,l=l: ridge(A,y,l),X,Y,10,0,20)
    res.append((l,m,dp))
res=np.array(res)
jm=int(np.argmin(res[:,1])); lmin=res[jm,0]
# regra do erro-padrão único, EP ENTRE DOBRAS no lambda mínimo
d=dobras(N,10,0); ef=[]
for j in range(10):
    tr,te=d!=j,d==j; Ap,m_,s_=pad(X[tr])
    b=ridge(Ap,Y[tr],lmin); ef.append(eqm(Y[te],prev((X[te]-m_)/s_,b)))
ep1=float(np.std(ef,ddof=1)/np.sqrt(10))
cand=[r for r in res if r[1]<=res[jm,1]+ep1]
l1se=max(r[0] for r in cand)
print('%10s %10s %10s %9s'%('lambda','EQM 10-fold','g.l. ef.',''))
for l in [1e-4,1e-3,1e-2,0.02,0.05,0.1,0.2,0.5,1.0,5.0,20.0]:
    k=int(np.argmin(np.abs(LAMS-l)))
    mk=' <- menor' if abs(LAMS[k]-lmin)<1e-12 else ''
    print('%10s %10s %10s %9s'%(c(LAMS[k],4),c(res[k,1]),c(GL[k],2),mk))
print()
print('  lambda do menor EQM            = %s  (EQM %s, g.l. ef. %s)'%(c(lmin,5),c(res[jm,1]),c(GL[jm],2)))
print('  erro-padrão entre as 10 dobras = %s'%c(ep1))
print('  lambda da regra do EP único    = %s  (EQM %s, g.l. ef. %s)'%(c(l1se,5),
      c(res[int(np.argmin(np.abs(LAMS-l1se))),1]),c(GL[int(np.argmin(np.abs(LAMS-l1se)))],2)))
print()
print('  EQM validado do MQO (lambda = 0)       = %s'%c(cv(ols,X,Y,10,0,20)[0]))
print('  EQM validado do ridge no melhor lambda = %s'%c(res[jm,1]))
print('  ganho do ridge sobre o MQO             = %.1f%%'%(100*(1-res[jm,1]/cv(ols,X,Y,10,0,20)[0])))

print('\n'+'='*70)
print('O ARGUMENTO DE HOERL E KENNARD (1970): existe lambda > 0 que reduz o')
print('erro quadrático dos COEFICIENTES, e não apenas o erro de predição.')
print('Simulação com o processo gerador conhecido, 1.000 amostras de n = 90.')
print()
bv=np.zeros(P)
for k,v in VERD.items(): bv[NOMES.index(k)]=v
# coeficientes verdadeiros na escala padronizada
bvp=bv*sd
rg=np.random.default_rng(37)
ls=[0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0]
ac={l:[] for l in ls}
for r in range(1000):
    yy=4.00+X@bv+rg.normal(0,0.60,N)
    Ap,m_,s_=pad(X)
    for l in ls:
        b=ols(Ap,yy) if l==0 else ridge(Ap,yy,l)
        ac[l].append(b[1:])
print('%10s %12s %12s %12s'%('lambda','viés² dos b','variância','EQM dos b'))
for l in ls:
    A_=np.array(ac[l]); vi=float(np.sum((A_.mean(0)-bvp)**2)); va=float(np.sum(A_.var(0,ddof=1)))
    print('%10s %12s %12s %12s'%(c(l,4),c(vi),c(va),c(vi+va)))
print()
mb=min(ls,key=lambda l:(lambda A_:float(np.sum((A_.mean(0)-bvp)**2))+float(np.sum(A_.var(0,ddof=1))))(np.array(ac[l])))
print('O lambda que minimiza o EQM dos coeficientes é %s, e NÃO zero.'%c(mb,4))
print('É o teorema de Hoerl e Kennard: o MQO é o de menor variância entre os')
print('estimadores NÃO VIESADOS, mas abrir mão do não viesamento permite')
print('reduzir o erro quadrático total.')

print('\n'+'='*70)
print('O RIDGE NÃO SELECIONA: nenhum coeficiente chega a zero')
b=ridge(Xp,Y,lmin)
print()
print('No lambda escolhido (%s), os 12 coeficientes:'%c(lmin,5))
for j in np.argsort(-np.abs(b[1:])):
    print('  %-11s %10s   %s'%(ROT[NOMES[j]],c(b[1+j]),'(no modelo verdadeiro)' if NOMES[j] in VERD else ''))
print()
print('  coeficientes exatamente zero: %d de 12'%int(np.sum(np.abs(b[1:])<1e-12))) 
print('  menor |coeficiente|:          %s'%c(float(np.abs(b[1:]).min())))
print()
print('Para zerar coeficientes é preciso a penalidade L1 do lasso.')

print('\n'+'='*70)
print('O REGIME EM QUE O RIDGE É INDISPENSÁVEL: p > n')
print('Espectro de reflectância: %d amostras, %d comprimentos de onda,'%(NE,PE))
print('resposta = teor de proteína (%). Dados fictícios, bandas construídas.')
print()
Ep,emu,esd=pad(XE)
ru,rs,rvt=np.linalg.svd(Ep,full_matrices=False)
print('  posto de X: %d (há %d colunas e %d linhas)'%(int(np.sum(rs>1e-8)),PE,NE))
print('  r médio entre colunas vizinhas: %s'%c(float(np.mean([np.corrcoef(Ep[:,i],Ep[:,i+1])[0,1] for i in range(PE-1)]))))
print()
print("O MQO é IMPOSSÍVEL: X'X é singular, há infinitas soluções com resíduo")
print('zero. O R devolve NA para 141 dos 200 coeficientes e R² = 1.')
bo=np.linalg.lstsq(np.column_stack([np.ones(NE),Ep]),YE,rcond=None)[0]
print('  R² de resubstituição da solução de norma mínima: %s'%c(1-sqr(YE,prev(Ep,bo))/float(np.sum((YE-YE.mean())**2))))
print()
rres=[]
for l in np.logspace(-4,2,31):
    m,_=cv(lambda A,y,l=l: ridge(A,y,l),XE,YE,10,0,5)
    rres.append((l,m))
rres=np.array(rres); jr=int(np.argmin(rres[:,1]))
print('%10s %12s'%('lambda','EQM 10-fold'))
for l,m in rres[::4]:
    print('%10s %12s%s'%(c(l,5),c(m),'  <- menor' if abs(l-rres[jr,0])<1e-12 else ''))
print()
print('  melhor lambda = %s, EQM validado = %s'%(c(rres[jr,0],5),c(rres[jr,1])))
print('  variância de y = %s'%c(float(YE.var(ddof=1))))
print('  R² validado = %s'%c(1-rres[jr,1]/float(YE.var(ddof=1))))
print()
print("Com p > n o ridge continua tendo solução única, porque X'X + n*lambda*I")
print('é sempre invertível para lambda > 0. É a razão algébrica da sua utilidade.')

print('\n'+'='*70)
print('UM CUIDADO QUE NÃO É OPCIONAL: PADRONIZAR')
print()
print('A penalidade soma os QUADRADOS dos coeficientes, de modo que ela pesa')
print('mais sobre as variáveis de escala pequena. Mudar a unidade de UMA delas')
print('muda o ajuste de TODAS as outras — o que é inaceitável.')
print()
jP=NOMES.index('pres'); jM=NOMES.index('mo')
X2=X.copy(); X2[:,jP]=X2[:,jP]/1000.0        # mg/dm3 -> g/dm3
def ridge_cru(A,y,lam):
    Ac=A-A.mean(0); n_=len(y)
    b=np.linalg.solve(Ac.T@Ac+lam*n_*np.eye(A.shape[1]),Ac.T@(y-y.mean()))
    return np.concatenate([[y.mean()-A.mean(0)@b],b])
print('SEM padronizar, lambda = 0,05:')
print('%-34s %12s %12s'%('','P em mg/dm³','P em g/dm³'))
b1=ridge_cru(X,Y,0.05); b2=ridge_cru(X2,Y,0.05)
print('%-34s %12s %12s'%('coeficiente do P (resina)',c(b1[1+jP],6),c(b2[1+jP],6)))
print('%-34s %12s %12s'%('coeficiente da M.O.',c(b1[1+jM]),c(b2[1+jM])))
print('%-34s %12s %12s'%('coeficiente do V%',c(b1[1+NOMES.index("v")]),c(b2[1+NOMES.index("v")])))
print('%-34s %12s %12s'%('EQM de resubstituição',c(eqm(Y,prev(X,b1))),c(eqm(Y,prev(X2,b2)))))
print()
print('Para que o ajuste fosse o MESMO, o coeficiente do P teria de passar de')
print('%s para %s (multiplicado por 1.000). Mas a penalidade proíbe:'%(c(b1[1+jP],6),c(1000*b1[1+jP],4)))
print('um coeficiente mil vezes maior pagaria um milhão de vezes mais, e o')
print('ajuste preferiu deixá-lo em %s — isto é, praticamente EXPULSOU o P'%c(b2[1+jP],6))
print('do modelo. E a M.O., que não mudou de unidade, viu o próprio coeficiente')
print('cair de %s para %s por efeito colateral. O EQM de resubstituição'%(c(b1[1+jM]),c(b2[1+jM])))
print('passou de %s para %s.'%(c(eqm(Y,prev(X,b1))),c(eqm(Y,prev(X2,b2)))))
print()
print('COM padronização dentro do laço:')
c1=cv(lambda A,y: ridge(A,y,0.05),X,Y,10,0,20)[0]
c2=cv(lambda A,y: ridge(A,y,0.05),X2,Y,10,0,20)[0]
print('  EQM validado, P em mg/dm³ = %s'%c(c1))
print('  EQM validado, P em g/dm³  = %s'%c(c2))
print('  diferença = %s  (invariância exata, a menos de arredondamento)'%c(abs(c1-c2),8))
print()
print('O glmnet padroniza por padrão (standardize = TRUE) e devolve os')
print('coeficientes já na escala original, que é o que se quer reportar.')
