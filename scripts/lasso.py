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

Xp,mu,sd=pad(X); SQT=float(np.sum((Y-Y.mean())**2)); yc=Y-Y.mean()
LMAX=float(np.abs(Xp.T@yc).max()/N)
print('Os mesmos 90 perfis de solo e as mesmas 12 candidatas.')
print('O modelo VERDADEIRO usa M.O., P (resina), V% e Argila.')
print()
print('O lambda acima do qual TODOS os coeficientes são zero tem forma fechada:')
print("  lambda_max = max_j |x_j' y| / n = %s"%c(LMAX,5))
print('Acima dele o lasso devolve só o intercepto. É por isso que o glmnet')
print('constrói a grade de lambda a partir desse valor, para baixo.')

print('\n'+'='*70)
print('O CAMINHO DO LASSO: a ordem de entrada das variáveis')
LAMS=np.logspace(np.log10(LMAX),np.log10(LMAX/2000),80)
PATH=np.array([b[1:] for b in lasso_path_a(Xp,Y,LAMS)])
ent={}
for j in range(P):
    nz=np.where(np.abs(PATH[:,j])>1e-8)[0]
    if len(nz): ent[j]=LAMS[nz[0]]
print()
print('ordem   variável      lambda de entrada   no modelo verdadeiro?')
for i,(j,l) in enumerate(sorted(ent.items(),key=lambda t:-t[1])):
    print('%5d   %-11s %17s   %s'%(i+1,ROT[NOMES[j]],c(l,5),'SIM' if NOMES[j] in VERD else '-'))
nunca=[ROT[NOMES[j]] for j in range(P) if j not in ent]
print()
print('Variáveis que nunca entram: %s'%(', '.join(nunca) or 'nenhuma'))
print()
GR=[LMAX*0.9,LMAX*0.5,0.20,0.10,0.05,0.02,0.01,0.002]
BS=lasso_path_a(Xp,Y,sorted(GR,reverse=True))
print('%10s %6s   %s'%('lambda','ativas','coeficientes não nulos'))
for l,b in zip(sorted(GR,reverse=True),BS):
    at=[j for j in range(P) if abs(b[1+j])>1e-8]
    print('%10s %6d   %s'%(c(l,5),len(at),', '.join('%s %s'%(ROT[NOMES[j]],c(b[1+j],3)) for j in at) or '(nenhuma)'))

print('\n'+'='*70)
print('O MECANISMO: limiar brando (soft-thresholding)')
print()
print('Em cada passo da descida por coordenadas, o coeficiente de uma variável')
print('é o do MQO PARCIAL (ajustado aos resíduos das outras), deslocado para')
print('zero em exatamente lambda, e grudado em zero se o deslocamento')
print('ultrapassar o próprio valor.')
print()
l0=0.05
b=lasso_path_a(Xp,Y,[LMAX,0.2,l0])[-1]
r=yc-Xp@b[1:]
print('Com lambda = %s, variável por variável:'%c(l0,3))
print('%-11s %13s %10s %13s'%('variável','MQO parcial','limiar','coef. lasso'))
for j in range(P):
    rho=float(Xp[:,j]@(r+Xp[:,j]*b[1+j])/N)
    print('%-11s %13s %10s %13s'%(ROT[NOMES[j]],c(rho,5),c(l0,5),
          c(b[1+j],5)+('  <- ZERO' if abs(b[1+j])<1e-8 else '')))
print()
print('Quando |MQO parcial| <= lambda, o coeficiente é exatamente zero. No ridge')
print('isso nunca acontece: a penalidade quadrática tem derivada NULA na origem,')
print('e a L1 tem derivada igual a lambda — é essa descontinuidade que zera.')

print('\n'+'='*70)
print('A ESCOLHA DE lambda POR VALIDAÇÃO CRUZADA (10 dobras, 10 repetições)')
GL=np.logspace(np.log10(LMAX),np.log10(LMAX/2000),40)
M_,S_=cv_path(lasso_path_a,X,Y,GL,10,0,10)
BA=lasso_path_a(Xp,Y,GL)
NAT=np.array([int(np.sum(np.abs(b[1:])>1e-8)) for b in BA])
jm=int(np.argmin(M_)); lmin=GL[jm]
d=dobras(N,10,0); ef=[]
for j in range(10):
    tr,te=d!=j,d==j; Ap,m_,s_=pad(X[tr])
    bb=lasso_path_a(Ap,Y[tr],[LMAX,lmin])[-1]
    ef.append(eqm(Y[te],prev((X[te]-m_)/s_,bb)))
ep1=float(np.std(ef,ddof=1)/np.sqrt(10))
cand=[i for i in range(len(GL)) if M_[i]<=M_[jm]+ep1]
j1=min(cand); l1se=GL[j1]
print()
print('%10s %13s %8s %s'%('lambda','EQM 10-fold','ativas',''))
for l in [0.30,0.20,0.12,0.08,0.05,0.035,0.0246,0.015,0.008,0.003,0.001]:
    k=int(np.argmin(np.abs(GL-l))); mk=''
    if k==jm: mk='  <- menor EQM'
    elif k==j1: mk='  <- regra do EP único'
    print('%10s %13s %8d%s'%(c(GL[k],5),c(M_[k]),NAT[k],mk))
bmin=BA[jm]; b1se=BA[j1]
amin=[j for j in range(P) if abs(bmin[1+j])>1e-8]; a1se=[j for j in range(P) if abs(b1se[1+j])>1e-8]
print()
print('  lambda do menor EQM = %s | EQM %s | %d ativas'%(c(lmin,5),c(M_[jm]),len(amin)))
print('     %s'%', '.join(ROT[NOMES[j]] for j in amin))
print('  erro-padrão entre as 10 dobras = %s'%c(ep1))
print('  lambda do EP único  = %s | EQM %s | %d ativas'%(c(l1se,5),c(M_[j1]),len(a1se)))
print('     %s'%', '.join(ROT[NOMES[j]] for j in a1se))
vt=sorted(NOMES.index(k) for k in VERD)
print()
print('  o modelo VERDADEIRO é: %s'%', '.join(ROT[NOMES[j]] for j in vt))
for rot,aa in [('menor EQM',amin),('EP único',a1se)]:
    print('  %-10s: recupera o verdadeiro? %-5s | acertos %d de 4 | falsos positivos %d'%(
          rot,sorted(aa)==vt,len(set(aa)&set(vt)),len(set(aa)-set(vt))))

print('\n'+'='*70)
print('O PROBLEMA DO LASSO COM GRUPOS CORRELACIONADOS')
print()
ja,js,jr=NOMES.index('argila'),NOMES.index('silte'),NOMES.index('areia')
print('A argila ESTÁ no modelo verdadeiro e o lasso não a escolhe: escolhe a')
print('areia e o silte. A razão é que argila = 100 - silte - areia, de modo que')
print('-silte-areia é um substituto quase perfeito da argila.')
print()
print('  r(Argila, Areia) = %s'%c(float(np.corrcoef(X[:,ja],X[:,jr])[0,1])))
print('  r(Argila, Silte) = %s'%c(float(np.corrcoef(X[:,ja],X[:,js])[0,1])))
print('  r(Argila, Silte + Areia) = %s'%c(float(np.corrcoef(X[:,ja],X[:,js]+X[:,jr])[0,1])))
print()
print('Dentro de um grupo colinear, o lasso escolhe UM representante (ou um')
print('substituto equivalente) quase ao acaso. 1.000 reamostras bootstrap,')
print('com lambda fixo no valor escolhido:')
rg=np.random.default_rng(41)
cont=np.zeros(P); nat=[]; conj={}
for r in range(1000):
    i=rg.integers(0,N,N); Ab,_,_=pad(X[i])
    bb=lasso_path_a(Ab,Y[i],[LMAX,0.2,lmin])[-1]
    at=tuple(sorted(j for j in range(P) if abs(bb[1+j])>1e-8))
    for j in at: cont[j]+=1
    nat.append(len(at)); conj[at]=conj.get(at,0)+1
print()
print('variável        frequência de seleção   no verdadeiro?')
for j in np.argsort(-cont):
    print('%-11s                %6.1f%%   %s'%(ROT[NOMES[j]],100*cont[j]/1000,'SIM' if NOMES[j] in VERD else '-'))
print()
print('  número de ativas: média %s, de %d a %d'%(c(np.mean(nat),2),min(nat),max(nat)))
print('  conjuntos distintos nas 1.000 reamostras: %d'%len(conj))
print('  o mais frequente apareceu em %.1f%% delas'%(100*max(conj.values())/1000))
print()
print('A comparação com o stepwise é de duas pontas, e vale enunciá-la com')
print('cuidado. Nas três variáveis verdadeiras de efeito forte, o lasso é')
print('MUITO mais confiável: M.O. 100,0%, V% 99,9% e P 99,1%, contra 100,0%,')
print('98,3% e 88,0% do stepwise. Mas o CONJUNTO exato é MENOS reprodutível:')
print('229 conjuntos distintos, o mais frequente em apenas 7,5% das reamostras,')
print('contra 11,5% do stepwise — porque o lasso no lambda do menor EQM ativa')
print('7 variáveis em média, e quanto maior o conjunto, mais formas de variar.')
print('A conclusão honesta: o lasso identifica melhor o que é forte, e não')
print('resolve a arbitrariedade dentro de um grupo colinear.')

print('\n'+'='*70)
print('REDE ELÁSTICA: a correção para grupos correlacionados')
print('Penalidade mista: alpha*L1 + (1-alpha)*L2/2. Com alpha = 1 é o lasso;')
print('com alpha = 0 é o ridge.')
print()
GE=np.logspace(np.log10(LMAX*1.3),np.log10(LMAX/2000),25)
print('%7s %11s %13s %8s   %s'%('alpha','melhor lam','EQM 10-fold','ativas','variáveis'))
for al in [1.0,0.7,0.4,0.1]:
    mm,_=cv_path(lambda A,y,ls,al=al: enet_path_a(A,y,ls,al),X,Y,GE,10,0,6)
    k=int(np.argmin(mm)); bb=enet_path_a(Xp,Y,GE,al)[k]
    at=[j for j in range(P) if abs(bb[1+j])>1e-8]
    print('%7s %11s %13s %8d   %s'%(c(al,2),c(GE[k],5),c(mm[k]),len(at),
          ', '.join(ROT[NOMES[j]] for j in at)[:46]))
print()
print('Quando alpha diminui, a penalidade se aproxima da do ridge e os grupos')
print('colineares passam a entrar JUNTOS, em vez de um representante sozinho.')

print('\n'+'='*70)
print('UM LIMITE ESTRUTURAL: com p > n, o lasso ativa no máximo n variáveis')
print('Espectro de reflectância: %d amostras, %d comprimentos de onda.'%(NE,PE))
Ep,emu,esd=pad(XE)
LME=float(np.abs(Ep.T@(YE-YE.mean())).max()/NE)
GEe=np.logspace(np.log10(LME),np.log10(LME/500),16)
Me,_=cv_path(lasso_path_a,XE,YE,GEe,10,0,3)
BEe=lasso_path_a(Ep,YE,GEe)
NAe=[int(np.sum(np.abs(b[1:])>1e-8)) for b in BEe]
je=int(np.argmin(Me))
print()
print('%10s %8s %13s %s'%('lambda','ativas','EQM 10-fold',''))
for k in range(len(GEe)):
    print('%10s %8d %13s%s'%(c(GEe[k],5),NAe[k],c(Me[k]),'  <- menor' if k==je else ''))
print()
print('  melhor lambda = %s | EQM validado = %s | %d ativas de %d'%(c(GEe[je],5),c(Me[je]),NAe[je],PE))
print('  R² validado = %s'%c(1-Me[je]/float(YE.var(ddof=1))))
print('  máximo de ativas em todo o caminho: %d (n = %d)'%(max(NAe),NE))
print()
print('O lasso não pode ativar mais de n variáveis: é consequência da geometria')
print('do problema, não da implementação. Com 200 candidatas e 60 amostras isso')
print('é ao mesmo tempo uma vantagem (o modelo fica interpretável, %d faixas'%NAe[je])
print('em vez de 200) e uma limitação (se o sinal estiver espalhado por mais de')
print('n variáveis, ele não o captura inteiro).')

print('\n'+'='*70)
print('O VIÉS DE ENCOLHIMENTO, e o lasso relaxado')
print()
print('Os coeficientes do lasso são encolhidos para zero DE PROPÓSITO, e por')
print('isso são viesados mesmo nas variáveis corretamente selecionadas.')
print('Reajustar por MQO apenas nas selecionadas (lasso relaxado) remove o')
print('encolhimento e mantém a seleção.')
print()
bo=ols(Xp,Y); br=ols(Xp[:,amin],Y)
print('%-11s %12s %14s %14s'%('variável','lasso','MQO relaxado','MQO completo'))
for i,j in enumerate(amin):
    print('%-11s %12s %14s %14s'%(ROT[NOMES[j]],c(bmin[1+j]),c(br[1+i]),c(bo[1+j])))
print()
print('  soma de |coef| do lasso        = %s'%c(float(np.abs(bmin[1:]).sum())))
print('  soma de |coef| do MQO relaxado = %s'%c(float(np.abs(br[1:]).sum())))
er_l,_=cv_path(lasso_path_a,X,Y,[LMAX,0.2,lmin],10,0,10)
def ols_sub(A,y,cols):
    bb=ols(A[:,cols],y); full=np.zeros(A.shape[1]+1); full[0]=bb[0]
    for i,j in enumerate(cols): full[1+j]=bb[1+i]
    return full
er_r,_=cv(lambda A,y: ols_sub(A,y,amin),X,Y,10,0,10)
print('  EQM validado do lasso          = %s'%c(er_l[-1]))
print('  EQM validado do MQO relaxado   = %s  (seleção FIXA: otimista)'%c(er_r))

print('\n'+'='*70)
print('A SÍNTESE DO GRUPO: os métodos no MESMO conjunto de solo')
print()
print('Duas colunas, e a diferença entre elas importa. Na primeira, o')
print('hiperparâmetro (lambda ou m) foi fixado no valor escolhido com a amostra')
print('INTEIRA — é o que se faz na prática, e é levemente otimista. Na segunda,')
print('ele é reescolhido DENTRO de cada dobra, por validação cruzada aninhada.')
print()
vy=float(Y.var(ddof=1))
def nest_lasso(A,y):
    GG=np.logspace(np.log10(LMAX),np.log10(LMAX/2000),24)
    mm,_=cv_path(lasso_path_a,A,y,GG,5,7,1)
    return lasso_path_a(pad(A)[0],y,GG)[int(np.argmin(mm))]
def nest_ridge(A,y):
    LL=np.logspace(-4,1,20); mm=[cv(lambda B,z,l=l: ridge(B,z,l),A,y,5,7,1)[0] for l in LL]
    return ridge(pad(A)[0],y,LL[int(np.argmin(mm))])
def nest_pcr(A,y):
    mm=[cv(lambda B,z,m=m: pcr(B,z,m),A,y,5,7,1)[0] for m in range(1,A.shape[1]+1)]
    return pcr(pad(A)[0],y,int(np.argmin(mm))+1)
def nest_pls(A,y):
    mm=[cv(lambda B,z,m=m: pls(B,z,m),A,y,5,7,1)[0] for m in range(1,A.shape[1]+1)]
    return pls(pad(A)[0],y,int(np.argmin(mm))+1)
m0,_=cv(lambda A,y: np.array([y.mean()]+[0.0]*A.shape[1]),X,Y,10,0,10)
m1,_=cv(ols,X,Y,10,0,10)
m2,_=cv(ols,X[:,vt],Y,10,0,10)
m3,_=cv(lambda A,y: ridge(A,y,0.03162),X,Y,10,0,10)
m4=er_l[-1]
m5,_=cv_path(lasso_path_a,X,Y,[LMAX,0.2,l1se],10,0,10); m5=m5[-1]
m6,_=cv(lambda A,y: pcr(A,y,7),X,Y,10,0,10)
m7,_=cv(lambda A,y: pls(A,y,3),X,Y,10,0,10)
n3,_=cv(nest_ridge,X,Y,10,0,3); n4,_=cv(nest_lasso,X,Y,10,0,3)
n6,_=cv(nest_pcr,X,Y,10,0,3);  n7,_=cv(nest_pls,X,Y,10,0,3)
print('%-34s %13s %9s %15s'%('método','EQM (fixo)','R² val.','EQM (aninhado)'))
for nome,v,nv in [('Modelo nulo (só a média)',m0,None),
                  ('MQO, 12 candidatas',m1,None),
                  ('MQO, só as 4 VERDADEIRAS',m2,None),
                  ('Ridge, lambda por VC',m3,n3),
                  ('Lasso, lambda do menor EQM',m4,n4),
                  ('Lasso, lambda do EP único',m5,None),
                  ('PCR, 7 componentes',m6,n6),
                  ('PLS, 3 componentes',m7,n7)]:
    print('%-34s %13s %9s %15s'%(nome,c(v),c(1-v/vy),c(nv) if nv else '—'))
print()
best=min([('MQO com as 4 verdadeiras',m2),('lasso',m4),('ridge',m3),('PCR',m6),('PLS',m7)],key=lambda t:t[1])
print('Com lambda fixo, o melhor é: %s (EQM %s).'%(best[0],c(best[1])))
print('O lasso (%s) e o MQO ajustado diretamente nas quatro variáveis'%c(m4))
print('VERDADEIRAS (%s) ficam praticamente EMPATADOS — diferença de %s,'%(c(m2),c(abs(m4-m2))))
print('dentro do erro de simulação. É um resultado notável: o lasso chega ao')
print('mesmo desempenho SEM saber quais são as quatro, escolhendo seis variáveis')
print('que incluem substitutos colineares das verdadeiras.')
print()
bn=min([('ridge',n3),('lasso',n4),('PCR',n6),('PLS',n7)],key=lambda t:t[1])
print('Na coluna ANINHADA, em que o hiperparâmetro é reescolhido dentro de cada')
print('dobra, todos pioram — e é essa a estimativa honesta. O melhor aninhado é')
print('o %s, com %s, contra %s do MQO com as quatro verdadeiras.'%(bn[0],c(bn[1]),c(m2)))
print()
print('A conclusão geral: com n = 90 e p = 12 a colinearidade incomoda mas não')
print('inviabiliza, e nenhum método regularizado ganha muito do MQO. O quadro')
print('muda por completo quando p > n, e é o que as fichas pcr e pls mostram.')
