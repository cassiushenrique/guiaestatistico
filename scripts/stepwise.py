import numpy as np, itertools, sys
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
from scipy import stats

Xp,mu,sd=pad(X); SQT=float(np.sum((Y-Y.mean())**2))

def ajuste(A,y,cols):
    Ad=np.column_stack([np.ones(len(y))]+[A[:,j] for j in cols])
    b=np.linalg.lstsq(Ad,y,rcond=None)[0]; sq=sqr(y,Ad@b)
    return b,sq,Ad
def aic(sq,n_,d): return n_*np.log(sq/n_)+2*d
def bic_(sq,n_,d): return n_*np.log(sq/n_)+np.log(n_)*d

def passo(A,y,modo,crit_=aic,verb=False,p_ent=0.05,p_sai=0.10,por_p=False):
    n_=len(y); pp=A.shape[1]
    cur=[] if modo=='frente' else list(range(pp))
    hist=[]
    while True:
        b,sq,_=ajuste(A,y,cur); cc=crit_(sq,n_,len(cur)+1)
        cands=[]
        if modo in ('frente','ambos'):
            for j in range(pp):
                if j in cur: continue
                b2,sq2,_=ajuste(A,y,cur+[j])
                cands.append(('+',j,crit_(sq2,n_,len(cur)+2),sq2))
        if modo in ('tras','ambos'):
            for j in cur:
                o=[k for k in cur if k!=j]
                b2,sq2,_=ajuste(A,y,o)
                cands.append(('-',j,crit_(sq2,n_,len(o)+1),sq2))
        if not cands: break
        cands.sort(key=lambda t:t[2])
        op,j,novo,_=cands[0]
        if novo>=cc-1e-10: break
        cur=cur+[j] if op=='+' else [k for k in cur if k!=j]
        hist.append((op,j,cc,novo))
        if verb: print('   %s%-10s AIC %8s -> %8s'%(op,ROT[NOMES[j]],c(cc,2),c(novo,2)))
    return sorted(cur),hist

print('Os mesmos 90 perfis de solo e as mesmas 12 candidatas da ficha subset.')
print('O modelo VERDADEIRO usa quatro: M.O., P (resina), V% e Argila.')
print('A busca exaustiva (4.096 modelos) achou, no tamanho 4: M.O., P (resina),')
print('V% e AREIA — e a areia é quase o complemento da argila.')
print()
print('='*70)
print('AS TRÊS DIREÇÕES, por AIC')
res={}
for modo,rot in [('frente','PARA FRENTE (forward)'),('tras','PARA TRÁS (backward)'),('ambos','NOS DOIS SENTIDOS (both)')]:
    print('\n'+rot)
    cur,hist=passo(Xp,Y,modo,verb=True)
    res[modo]=cur
    b,sq,_=ajuste(Xp,Y,cur)
    print('   SELECIONADO (%d vars): %s'%(len(cur),', '.join(ROT[NOMES[j]] for j in cur)))
    print('   R² = %s | AIC = %s | EQM resub = %s'%(c(1-sq/SQT),c(aic(sq,N,len(cur)+1),2),c(sq/N)))
print()
print('Coincidem?','SIM' if res['frente']==res['tras']==res['ambos'] else 'NÃO')
for a,b_ in [('frente','tras'),('frente','ambos'),('tras','ambos')]:
    d=set(res[a])^set(res[b_])
    print('  %-7s x %-7s: %s'%(a,b_,'idênticos' if not d else 'diferem em '+', '.join(ROT[NOMES[j]] for j in sorted(d))))
print()
print('Com BIC, que penaliza mais:')
for modo in ['frente','tras','ambos']:
    cur,_=passo(Xp,Y,modo,crit_=bic_)
    print('  %-7s (%d vars): %s'%(modo,len(cur),', '.join(ROT[NOMES[j]] for j in cur)))

print('\n'+'='*70)
print('STEPWISE contra BUSCA EXAUSTIVA: quantos modelos cada um examina?')
print()
print('   p   exaustiva   para frente (p(p+1)/2)   razão')
for p_ in [12,20,30,50,100]:
    print('%4d   %9s   %20d   %s'%(p_,'2^%d'%p_,p_*(p_+1)//2,'%.3g'%(2**p_/(p_*(p_+1)/2))))
print()
cur,_=passo(Xp,Y,'frente')
print('Com p = 12, a exaustiva examina 4.096 modelos e a seleção para frente')
print('examina no máximo 78. E chegou ao mesmo lugar? ',end='')
# melhor de mesmo tamanho pela exaustiva
tam=len(cur); bb=None
for cols in itertools.combinations(range(P),tam):
    _,sq,_=ajuste(Xp,Y,list(cols))
    if bb is None or sq<bb[0]: bb=(sq,cols)
print('o melhor subconjunto de tamanho %d é'%tam)
print('   %s'%', '.join(ROT[NOMES[j]] for j in bb[1]))
print('   e o da seleção para frente é')
print('   %s'%', '.join(ROT[NOMES[j]] for j in cur))
print('   iguais?',sorted(bb[1])==cur)

print('\n'+'='*70)
print('OS VALORES-p APÓS A SELEÇÃO NÃO SÃO VÁLIDOS')
print()
b,sq,Ad=ajuste(Xp,Y,cur)
s2=sq/(N-len(cur)-1); C=np.linalg.inv(Ad.T@Ad)*s2
ep=np.sqrt(np.diag(C)); t=b/ep
pv=2*(1-stats.t.cdf(np.abs(t),N-len(cur)-1))
print('A saída que o R imprime para o modelo selecionado:')
print('%-12s %10s %10s %9s %12s'%('termo','estimativa','erro-padrão','t','p (NOMINAL)'))
print('%-12s %10s %10s %9s %12s'%('(intercepto)',c(b[0]),c(ep[0]),c(t[0],2),'%.2e'%pv[0]))
for i,j in enumerate(cur):
    print('%-12s %10s %10s %9s %12s'%(ROT[NOMES[j]],c(b[i+1]),c(ep[i+1]),c(t[i+1],2),'%.2e'%pv[i+1]))
print()
print('Todos "altamente significativos". Mas esses p supõem que o modelo foi')
print('escolhido ANTES de ver os dados, e ele não foi.')
print()
print('A DEMONSTRAÇÃO: y PERMUTADO, sem nenhuma relação com X, 2.000 vezes.')
print('Em cada permutação, roda-se a seleção para frente e registra-se o menor')
print('valor-p do modelo selecionado.')
rg=np.random.default_rng(13)
menor=[];nsel=[];qsig=0;r2s=[]
for r in range(2000):
    yp=rg.permutation(Y)
    cu,_=passo(Xp,yp,'frente')
    nsel.append(len(cu))
    if not cu: menor.append(1.0); r2s.append(0.0); continue
    b2,sq2,Ad2=ajuste(Xp,yp,cu)
    s22=sq2/(N-len(cu)-1); ep2=np.sqrt(np.diag(np.linalg.inv(Ad2.T@Ad2)*s22))
    t2=b2[1:]/ep2[1:]
    pv2=2*(1-stats.t.cdf(np.abs(t2),N-len(cu)-1))
    menor.append(pv2.min()); r2s.append(1-sq2/np.sum((yp-yp.mean())**2))
    if pv2.min()<0.05: qsig+=1
menor=np.array(menor); nsel=np.array(nsel); r2s=np.array(r2s)
print()
print('  variáveis selecionadas (média)                 %s'%c(nsel.mean(),3))
print('  permutações em que selecionou >= 1 variável     %.1f%%'%(100*np.mean(nsel>0)))
print('  permutações com algum p < 0,05                  %.1f%%  <- deveria ser ~5%%'%(100*qsig/2000))
print('  permutações com algum p < 0,01                  %.1f%%  <- deveria ser ~1%%'%(100*np.mean(menor<0.01)))
print('  permutações com algum p < 0,001                 %.1f%%  <- deveria ser ~0,1%%'%(100*np.mean(menor<0.001)))
print('  R² médio do modelo selecionado                  %s'%c(r2s.mean()))
print('  R² máximo observado                             %s'%c(r2s.max()))
print()
print('Em dados sem NENHUM sinal, a seleção para frente por AIC devolve em média')
print('%s variáveis, e em %.1f%% das vezes ao menos uma delas tem p < 0,05.'%(c(nsel.mean(),2),100*qsig/2000))

print('\n'+'='*70)
print('INSTABILIDADE: 1.000 reamostras bootstrap, seleção refeita em cada uma')
rg=np.random.default_rng(17)
cont=np.zeros(P); tam=[]; mesmo=0; cur0=set(cur)
for r in range(1000):
    i=rg.integers(0,N,N); Xb,_,_=pad(X[i])
    cu,_=passo(Xb,Y[i],'frente')
    for j in cu: cont[j]+=1
    tam.append(len(cu))
    if set(cu)==cur0: mesmo+=1
tam=np.array(tam)
print()
print('variável        frequência de seleção   no verdadeiro?   no selecionado?')
for j in np.argsort(-cont):
    print('%-10s                %6.1f%%   %-14s   %s'%(ROT[NOMES[j]],100*cont[j]/1000,
          'SIM' if NOMES[j] in VERD else '-','SIM' if j in cur0 else '-'))
print()
print('tamanho do modelo selecionado: média %s, de %d a %d'%(c(tam.mean(),2),tam.min(),tam.max()))
print('o conjunto exato da amostra original foi reencontrado em %.1f%% das reamostras'%(100*mesmo/1000))

print('\n'+'='*70)
print('O ERRO DE PREDIÇÃO, honestamente estimado')
d=dobras(N,10,3); er=np.empty(N); escolhas=[]
for j in range(10):
    tr,te=d!=j,d==j
    Ap,m_,s_=pad(X[tr]); ytr=Y[tr]
    cu,_=passo(Ap,ytr,'frente')
    b2,_,_=ajuste(Ap,ytr,cu)
    Xte=(X[te]-m_)/s_
    er[te]=Y[te]-(b2[0]+Xte[:,cu]@b2[1:])
    escolhas.append(cu)
_,sqf,_=ajuste(Xp,Y,cur)
print()
print('EQM de resubstituição do modelo selecionado          = %s'%c(sqf/N))
dd=dobras(N,10,3); efx=np.empty(N)
for j in range(10):
    tr,te=dd!=j,dd==j
    Ap,m_,s_=pad(X[tr]); b2,_,_=ajuste(Ap,Y[tr],cur)
    Xte=(X[te]-m_)/s_; efx[te]=Y[te]-(b2[0]+Xte[:,cur]@b2[1:])
print('EQM validado, modelo FIXO (selecionado na amostra)   = %s  <- OTIMISTA'%c(eqm(np.zeros(N),efx)))
print('EQM validado, seleção REFEITA em cada dobra          = %s'%c(eqm(np.zeros(N),er)))
print('EQM validado do modelo VERDADEIRO                    = %s'%c(cv(ols,X[:,[NOMES.index(k) for k in VERD]],Y,10,3)[0]))
print('EQM validado do modelo COMPLETO                      = %s'%c(cv(ols,X,Y,10,3)[0]))
print()
print('Os modelos escolhidos nas 10 dobras:')
for j,cu in enumerate(escolhas):
    print('  dobra %2d (%d vars): %s'%(j+1,len(cu),', '.join(ROT[NOMES[k]] for k in cu)))
tod=set().union(*[set(c_) for c_ in escolhas]); com=set(escolhas[0]).intersection(*[set(c_) for c_ in escolhas])
print()
print('variáveis que apareceram em ALGUMA dobra: %d de 12'%len(tod))
print('variáveis que apareceram em TODAS as dobras: %s'%', '.join(ROT[NOMES[j]] for j in sorted(com)))

print('\n'+'='*70)
print('O REGIME EM QUE O STEPWISE FRACASSA POR COMPLETO: p próximo de n')
print('Simulação: n = 50, p = 45 candidatas, NENHUMA com efeito real.')
rg=np.random.default_rng(23)
r2=[];ns=[]
for r in range(200):
    Z=rg.normal(0,1,(50,45)); yz=rg.normal(0,1,50)
    Zp,_,_=pad(Z)
    cu,_=passo(Zp,yz,'frente')
    if cu:
        _,sq2,_=ajuste(Zp,yz,cu); r2.append(1-sq2/np.sum((yz-yz.mean())**2))
    else: r2.append(0.0)
    ns.append(len(cu))
print()
print('  variáveis selecionadas: média %s, máximo %d'%(c(np.mean(ns),2),max(ns)))
print('  R² médio do modelo selecionado: %s'%c(np.mean(r2)))
print('  R² máximo: %s'%c(max(r2)))
print('  proporção com R² > 0,50: %.1f%%'%(100*np.mean(np.array(r2)>0.5)))
print()
print('Nenhuma das 45 candidatas tem relação com y, e ainda assim o R² médio do')
print('modelo selecionado é %s. É a busca, e não os dados, que produz o ajuste.'%c(np.mean(r2)))
