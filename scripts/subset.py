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
np.set_printoptions(suppress=True)

print('ANÁLISE DE SOLO: 90 amostras, 12 candidatas, resposta = produtividade')
print('O modelo VERDADEIRO (dados fictícios, de construção conhecida) usa')
print('apenas QUATRO das doze: M.O., P resina, V% e argila.')
print()
print('variável      VIF    r com y   no modelo verdadeiro?')
V=vif(X)
for j,nm in enumerate(NOMES):
    print('%-9s %9.1f   %+7.4f   %s'%(ROT[nm],V[j],np.corrcoef(X[:,j],Y)[0,1],
          'SIM' if nm in VERD else '-'))
print()
print('A argila, o silte e a areia somam aproximadamente 100% por construção')
print('física, e por isso os VIF delas são os maiores: 101, 29 e 126.')

Xp,mu,sd=pad(X); SQT=float(np.sum((Y-Y.mean())**2))
s2=sqr(Y,prev(Xp,ols(Xp,Y)))/(N-P-1)
print('\nsigma^2 estimado pelo modelo completo = %s (g.l. = %d)'%(c(s2),N-P-1))

print('\n'+'='*70)
print('ENUMERAÇÃO EXAUSTIVA: todos os 2^12 = 4.096 subconjuntos')
melhor={}; todos=[]
for tam in range(0,P+1):
    bb=None
    for cols in itertools.combinations(range(P),tam):
        r=crit(Xp,Y,list(cols),s2); r['cols']=cols; todos.append(r)
        if bb is None or r['sqr']<bb['sqr']: bb=r
    melhor[tam]=bb
print('%d subconjuntos avaliados'%len(todos))
print()
print('tam.  variáveis do melhor subconjunto desse tamanho          R²     R²aj      Cp      AIC      BIC')
for tam in range(0,P+1):
    b=melhor[tam]; nm=', '.join(ROT[NOMES[j]] for j in b['cols']) or '(só intercepto)'
    print('%3d   %-50s %6s %7s %7s %8s %8s'%(tam,nm[:50],c(b['r2'],4),c(b['r2a'],4),
          c(b['cp'],2),c(b['aic'],2),c(b['bic'],2)))
print()
for nome,chave,mn in [('R² ajustado','r2a',False),('Cp de Mallows','cp',None),
                      ('AIC','aic',True),('BIC','bic',True)]:
    if chave=='cp':
        cand=[k for k in range(P+1) if melhor[k]['cp']<=melhor[k]['p']]
        t=min(cand); t2=min(range(P+1),key=lambda k:melhor[k]['cp'])
        print('%-15s escolhe tamanho %d (MENOR modelo com Cp <= p)'%(nome,t))
        print('%-15s  menor Cp absoluto: tamanho %d (Cp = %s)'%('',t2,c(melhor[t2]['cp'],2)))
        print('%-15s  ATENÇÃO: o Cp do modelo COMPLETO é sempre exatamente p.'%'')
        print('%-15s  Aqui: Cp = %s e p = %d. A regra "Cp mais próximo de p"'%('',c(melhor[P]['cp'],2),melhor[P]['p']))
        print('%-15s  escolheria trivialmente o modelo completo, e está errada.'%'')
        print('%-15s  A regra correta é o MENOR modelo com Cp <= p: tamanho %d,'%('',t))
        print('%-15s  que é %s'%('',', '.join(ROT[NOMES[j]] for j in melhor[t]['cols'])))
    else:
        f=max if not mn else min
        t=f(range(P+1),key=lambda k:melhor[k][chave])
        print('%-15s escolhe tamanho %d: %s'%(nome,t,', '.join(ROT[NOMES[j]] for j in melhor[t]['cols'])))

print('\nO modelo VERDADEIRO tem tamanho 4: M.O., P (resina), V%, Argila')
vt=tuple(sorted(NOMES.index(k) for k in VERD))
print('O melhor subconjunto de tamanho 4 é:',', '.join(ROT[NOMES[j]] for j in melhor[4]['cols']))
print('É o verdadeiro?',melhor[4]['cols']==vt)
rv=crit(Xp,Y,list(vt),s2)
print('SQR do verdadeiro = %s | SQR do melhor de tamanho 4 = %s'%(c(rv['sqr']),c(melhor[4]['sqr'])))
print('R² do verdadeiro = %s | do melhor de tamanho 4 = %s'%(c(rv['r2']),c(melhor[4]['r2'])))
pos=sorted(range(len(todos)),key=lambda i:todos[i]['sqr'])
rk=[i for i,q in enumerate(pos) if todos[q]['cols']==vt][0]
print('Posição do modelo verdadeiro no ranking dos 4.096 por SQR: %dº'%(rk+1))

print('\n'+'='*70)
print('A MALDIÇÃO DO VENCEDOR: o R² do melhor subconjunto é otimista por')
print('construção, porque é o MÁXIMO de muitas tentativas.')
print()
print('Simulação: y PERMUTADO (nenhuma relação com X), 400 repetições.')
rg=np.random.default_rng(5)
mr=np.zeros((400,6))
for r in range(400):
    yp=rg.permutation(Y)
    for ti,tam in enumerate([1,2,3,4,5,6]):
        mx=0
        for cols in itertools.combinations(range(P),tam):
            Ad=np.column_stack([np.ones(N)]+[Xp[:,j] for j in cols])
            b=np.linalg.lstsq(Ad,yp,rcond=None)[0]
            mx=max(mx,1-sqr(yp,Ad@b)/np.sum((yp-yp.mean())**2))
        mr[r,ti]=mx
print('tam.   R² esperado de UM subconjunto    R² do MELHOR subconjunto')
for ti,tam in enumerate([1,2,3,4,5,6]):
    print('%3d                      %s                     %s'%(tam,c(tam/(N-1),4),c(mr[:,ti].mean(),4)))
print()
print('Com y permutado, o R² esperado de um subconjunto fixo de tamanho k é')
print('k/(n-1). O R² do MELHOR de todos os subconjuntos desse tamanho é muito')
print('maior: no tamanho 4, %s contra %s esperados. É inflação pura,'%(c(mr[:,3].mean(),4),c(4/(N-1),4)))
print('vinda da própria busca, sem nenhum sinal nos dados.')

print('\n'+'='*70)
print('O que a busca exaustiva CUSTA, e o que ela NÃO resolve')
print()
print('   p   subconjuntos        viável?')
for p_ in [10,12,15,20,25,30,40,50]:
    print('%4d   %18s   %s'%(p_,('%d'%2**p_),'sim' if p_<=25 else 'não'))
print()
print('O leaps::regsubsets usa ramificação e poda (branch and bound) e chega a')
print('p da ordem de 30 a 40; acima disso, nem ele. E o custo não é o pior')
print('problema: o pior é a INSTABILIDADE da escolha.')

print('\n'+'='*70)
print('INSTABILIDADE: 500 reamostras bootstrap, busca exaustiva em cada uma.')
print('Com que frequência cada variável entra no melhor subconjunto de 4?')
rg=np.random.default_rng(11)
cont=np.zeros(P); mesmo=0
for r in range(500):
    i=rg.integers(0,N,N); Xb,_,_=pad(X[i]); yb=Y[i]
    bb=None
    for cols in itertools.combinations(range(P),4):
        Ad=np.column_stack([np.ones(N)]+[Xb[:,j] for j in cols])
        b=np.linalg.lstsq(Ad,yb,rcond=None)[0]; sq=sqr(yb,Ad@b)
        if bb is None or sq<bb[0]: bb=(sq,cols)
    for j in bb[1]: cont[j]+=1
    if bb[1]==melhor[4]['cols']: mesmo+=1
print()
print('variável        frequência de inclusão   no verdadeiro?')
for j in np.argsort(-cont):
    print('%-10s                 %6.1f%%   %s'%(ROT[NOMES[j]],100*cont[j]/500,
          'SIM' if NOMES[j] in VERD else '-'))
print()
print('O melhor subconjunto de 4 da amostra original foi reencontrado em')
print('apenas %.1f%% das 500 reamostras.'%(100*mesmo/500))

print('\n'+'='*70)
print('A COMPARAÇÃO QUE IMPORTA: erro de PREDIÇÃO validado por k-fold')
print('(validação cruzada ANINHADA: a busca é refeita dentro de cada dobra)')
print()
d=dobras(N,10,3)
er_an=np.empty(N); er_fix=np.empty(N)
for j in range(10):
    tr,te=d!=j,d==j
    Ap,m_,s_=pad(X[tr]); ytr=Y[tr]
    s2t=sqr(ytr,prev(Ap,ols(Ap,ytr)))/(tr.sum()-P-1)
    bb=None
    for tam in range(1,P+1):
        for cols in itertools.combinations(range(P),tam):
            r=crit(Ap,ytr,list(cols),s2t)
            if bb is None or r['bic']<bb['bic']: bb=r; bb['cols']=cols
    Ad=np.column_stack([np.ones(tr.sum())]+[Ap[:,k] for k in bb['cols']])
    b=np.linalg.lstsq(Ad,ytr,rcond=None)[0]
    Xte=(X[te]-m_)/s_
    er_an[te]=Y[te]-(b[0]+Xte[:,list(bb['cols'])]@b[1:])
    # mesmo modelo, mas escolhido NA AMOSTRA INTEIRA (errado de propósito)
    Ad2=np.column_stack([np.ones(tr.sum())]+[Ap[:,k] for k in melhor[4]['cols']])
    b2=np.linalg.lstsq(Ad2,ytr,rcond=None)[0]
    er_fix[te]=Y[te]-(b2[0]+Xte[:,list(melhor[4]['cols'])]@b2[1:])
print('EQM de resubstituição do melhor subconjunto de 4 = %s'%c(melhor[4]['sqr']/N))
print('EQM validado, subconjunto escolhido na amostra INTEIRA = %s  <- OTIMISTA'%c(eqm(np.zeros(N),er_fix)))
print('EQM validado, busca REFEITA dentro de cada dobra     = %s'%c(eqm(np.zeros(N),er_an)))
print('EQM validado do modelo VERDADEIRO (4 variáveis)      = %s'%c(cv(ols,X[:,list(vt)],Y,10,3)[0]))
print('EQM validado do modelo COMPLETO (12 variáveis)       = %s'%c(cv(ols,X,Y,10,3)[0]))
print()
print('Os subconjuntos escolhidos dentro de cada uma das 10 dobras:')
d=dobras(N,10,3)
for j in range(10):
    tr=d!=j; Ap,_,_=pad(X[tr]); ytr=Y[tr]
    s2t=sqr(ytr,prev(Ap,ols(Ap,ytr)))/(tr.sum()-P-1); bb=None
    for tam in range(1,P+1):
        for cols in itertools.combinations(range(P),tam):
            r=crit(Ap,ytr,list(cols),s2t)
            if bb is None or r['bic']<bb['bic']: bb=r; bb['cols']=cols
    print('  dobra %2d (%d vars): %s'%(j+1,len(bb['cols']),', '.join(ROT[NOMES[k]] for k in bb['cols'])))
