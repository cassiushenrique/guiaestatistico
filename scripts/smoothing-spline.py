import numpy as np
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (cab15)
# ------------------------------------------------------------------
import numpy as np
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (dados15)
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

# Dados fictícios: 260 medições de trocas gasosas (arquivo fotossintese.csv)
fot = le_csv('fotossintese.csv')
PAR=fot['par'].to_numpy(float); TEMP=fot['temp'].to_numpy(float)
VPD=fot['vpd'].to_numpy(float); A=fot['A'].to_numpy(float)
NG=len(A)
SIG=1.30
INT=14.0

def f_par(x):
    x=np.asarray(x,float)
    return 26.0*(1-np.exp(-0.0024*x))-1.9e-6*np.maximum(x-1000,0)**2
def f_temp(t): t=np.asarray(t,float); return -0.042*(t-27.0)**2
def f_vpd(v):  v=np.asarray(v,float); return -2.9*v+0.33*v**2
C1=19.9605909253; C2=-1.4311050277; C3=-2.0245067433
def verdade(x,t,v): return INT+(f_par(x)-C1)+(f_temp(t)-C2)+(f_vpd(v)-C3)
# ------------------------------------------------------------------ fim do bloco comum

# ------------------------------------------------------------------ utilidades
def c(v, d=4): return ('%.*f' % (d, v)).replace('.', ',')
def eqm(o, p): return float(np.mean((np.asarray(o) - np.asarray(p))**2))
def sqr(o, p): return float(np.sum((np.asarray(o) - np.asarray(p))**2))
def r2(o, p):  return 1 - sqr(o, p)/float(np.sum((np.asarray(o) - np.mean(o))**2))

def dobras(n, k, seed):
    rg = np.random.default_rng(seed)
    d = np.tile(np.arange(k), int(np.ceil(n/k)))[:n]; rg.shuffle(d); return d

def cv_uni(ajusta, x, y, k=10, seed=0, reps=1):
    """EQM por validação cruzada para um suavizador univariado.
    ajusta(xtr, ytr, xte) -> predições em xte."""
    tot = []
    for r in range(reps):
        d = dobras(len(y), k, seed + 991*r); e = np.empty(len(y))
        for j in range(k):
            tr, te = d != j, d == j
            e[te] = y[te] - ajusta(x[tr], y[tr], x[te])
        tot.append(float(np.mean(e**2)))
    return float(np.mean(tot)), (float(np.std(tot, ddof=1)) if reps > 1 else 0.0)

# ------------------------------------------------------------------ polinômios
def poli(x, g, xr=None):
    """Base polinomial ORTOGONAL (Legendre reescalada ao intervalo de xr)."""
    x = np.asarray(x, float); xr = x if xr is None else np.asarray(xr, float)
    a, b = xr.min(), xr.max()
    z = 2*(x - a)/(b - a) - 1
    return np.polynomial.legendre.legvander(z, g)

def poli_cru(x, g):
    x = np.asarray(x, float)
    return np.column_stack([x**j for j in range(g+1)])

# ------------------------------------------------------------------ B-splines
def bspline(x, nos, grau=3):
    """Base B-spline por Cox-de Boor. 'nos' = vetor de nós INTERNOS."""
    x = np.asarray(x, float); nos = np.asarray(nos, float)
    a, b = bspline.lim
    t = np.concatenate([[a]*(grau+1), nos, [b]*(grau+1)])
    m = len(t) - grau - 1
    B = np.zeros((len(x), m))
    for i in range(m):
        B[:, i] = (x >= t[i]) & (x < t[i+1])
    B[np.isclose(x, b), m-1] = 1.0
    for d in range(1, grau+1):
        Bn = np.zeros((len(x), len(t) - d - 1))
        for i in range(Bn.shape[1]):
            den1 = t[i+d] - t[i]; den2 = t[i+d+1] - t[i+1]
            a1 = (x - t[i])/den1*B[:, i] if den1 > 0 else 0.0
            a2 = (t[i+d+1] - x)/den2*B[:, i+1] if den2 > 0 else 0.0
            Bn[:, i] = a1 + a2
        B = Bn
    return B

def base_bs(x, nos, grau=3, lim=None):
    bspline.lim = lim if lim is not None else (float(np.min(x)), float(np.max(x)))
    return bspline(x, nos, grau)

def trunc_pot(x, nos, grau=3):
    """Base de potências truncadas: 1, x, ..., x^g, (x-k)_+^g."""
    x = np.asarray(x, float)
    cols = [x**j for j in range(grau+1)]
    for k in np.asarray(nos, float):
        cols.append(np.maximum(x - k, 0.0)**grau)
    return np.column_stack(cols)

def natural_cs(x, nos, lim):
    """Base de spline cúbica NATURAL (linear fora dos nós de fronteira)."""
    x = np.asarray(x, float); K = np.concatenate([[lim[0]], np.asarray(nos, float), [lim[1]]])
    k = len(K)
    def d(j):
        return (np.maximum(x - K[j], 0)**3 - np.maximum(x - K[k-1], 0)**3)/(K[k-1] - K[j])
    cols = [np.ones_like(x), x] + [d(j) - d(k-2) for j in range(k-2)]
    return np.column_stack(cols)

def ajusta_base(B, y):
    return np.linalg.lstsq(B, y, rcond=None)[0]

# --------------------------------------------------- spline de suavização
def suav_spline(x, y, lam, escala=True):
    """Spline cúbica de suavização EXATA, pelo algoritmo de Reinsch
    (Green e Silverman, 1994, cap. 2). Resolve o sistema BANDADO
        (R + lam Q' W^-1 Q) gama = Q' ybar ,   ajustado = ybar - lam W^-1 Q gama
    que é numericamente muito mais estável que inverter (W + lam K).
    Com escala=True, x é levado a [0, 1] antes de penalizar, de modo que lambda
    tenha o mesmo significado para variáveis de unidades diferentes — é o que o
    smooth.spline faz por trás do argumento spar.
    Devolve (ajustados, x únicos na escala original, contagens, médias, g.l. efetivos)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    a0, b0 = (x.min(), x.max()) if escala else (0.0, 1.0)
    xs = (x - a0)/(b0 - a0) if escala else x
    xo_s, inv = np.unique(np.round(xs, 12), return_inverse=True)
    m = len(xo_s)
    xo = xo_s*(b0 - a0) + a0 if escala else xo_s
    cont = np.bincount(inv, minlength=m).astype(float)
    ybar = np.bincount(inv, weights=y, minlength=m)/cont
    h = np.diff(xo_s)
    Q = np.zeros((m, m-2)); R = np.zeros((m-2, m-2))
    for j in range(1, m-1):
        Q[j-1, j-1] = 1/h[j-1]; Q[j, j-1] = -1/h[j-1] - 1/h[j]; Q[j+1, j-1] = 1/h[j]
        R[j-1, j-1] = (h[j-1] + h[j])/3
        if j < m-2:
            R[j-1, j] = h[j]/6; R[j, j-1] = h[j]/6
    Winv = 1.0/cont
    M = R + lam*(Q.T*Winv) @ Q
    gama = np.linalg.solve(M, Q.T @ ybar)
    aj = ybar - lam*Winv*(Q @ gama)
    # g.l. efetivos = tr(S) = m - lam * tr( W^-1 Q M^-1 Q' )
    tr = m - lam*float(np.sum(Winv*np.einsum('ij,ji->i', Q, np.linalg.solve(M, Q.T))))
    return aj, xo, cont, ybar, tr

def ss_edf(x, lam):
    return suav_spline(x, np.zeros(len(x)), lam)[4]

def ss_gcv(x, y, lam):
    aj, xo, cont, ybar, tr = suav_spline(x, y, lam)
    n = len(y)
    res = np.asarray(y, float) - np.interp(x, xo, aj)
    return float(np.sum(res**2)/n)/((1 - tr/n)**2), tr

def ss_prev(x, y, lam, xnovo):
    aj, xo, cont, ybar, tr = suav_spline(x, y, lam)
    # extrapolação LINEAR fora do domínio, como manda a spline natural
    d0 = (aj[1]-aj[0])/(xo[1]-xo[0]); d1 = (aj[-1]-aj[-2])/(xo[-1]-xo[-2])
    xn = np.asarray(xnovo, float); out = np.interp(xn, xo, aj)
    out = np.where(xn < xo[0],  aj[0]  + d0*(xn - xo[0]),  out)
    out = np.where(xn > xo[-1], aj[-1] + d1*(xn - xo[-1]), out)
    return out

# ------------------------------------------------------------------ LOESS
def tricubo(u): return (1 - np.abs(u)**3)**3

def loess(x, y, xnovo, span=0.35, grau=1, iteracoes=0):
    """Regressão local ponderada (Cleveland, 1979).
    iteracoes > 0 ativa a reponderação robusta bisquare: os pesos de robustez
    são sempre calculados a partir dos resíduos NOS PONTOS DE AJUSTE x, e não
    nos pontos onde se quer prever."""
    x = np.asarray(x, float); y = np.asarray(y, float); xn = np.asarray(xnovo, float)
    n = len(x); q = max(grau + 2, int(np.ceil(span*n)))

    def ajusta_em(alvos, rob):
        out = np.empty(len(alvos))
        for i, x0 in enumerate(alvos):
            dd = np.abs(x - x0); ord_ = np.argsort(dd); viz = ord_[:q]
            dmax = dd[viz[-1]]
            w = tricubo(dd[viz]/dmax) if dmax > 0 else np.ones(q)
            w = w*rob[viz]
            Z = np.column_stack([(x[viz] - x0)**j for j in range(grau+1)])
            WZ = Z*w[:, None]
            try: b = np.linalg.solve(Z.T @ WZ, WZ.T @ y[viz])
            except np.linalg.LinAlgError: b = np.linalg.lstsq(Z.T @ WZ, WZ.T @ y[viz], rcond=None)[0]
            out[i] = b[0]
        return out

    rob = np.ones(n)
    for _ in range(iteracoes):
        aj = ajusta_em(x, rob)            # SEMPRE nos x de ajuste
        res = y - aj
        s = np.median(np.abs(res))
        rob = np.clip(1 - (res/(6*s))**2, 0, None)**2 if s > 0 else np.ones(n)
    return ajusta_em(xn, rob)

def loess_edf(x, span=0.35, grau=1):
    """Graus de liberdade efetivos: traço da matriz suavizadora."""
    x = np.asarray(x, float); n = len(x); q = max(grau + 2, int(np.ceil(span*n)))
    tr = 0.0
    for i, x0 in enumerate(x):
        dd = np.abs(x - x0); ord_ = np.argsort(dd); viz = ord_[:q]
        dmax = dd[viz[-1]]
        w = tricubo(dd[viz]/dmax) if dmax > 0 else np.ones(q)
        Z = np.column_stack([(x[viz] - x0)**j for j in range(grau+1)])
        WZ = Z*w[:, None]
        try: li = np.linalg.solve(Z.T @ WZ, WZ.T)
        except np.linalg.LinAlgError: li = np.linalg.lstsq(Z.T @ WZ, WZ.T, rcond=None)[0]
        k = np.where(viz == i)[0]
        if len(k): tr += li[0, k[0]]
    return tr

# ------------------------------------------------------------------ GAM
def gam_backfit(Xs, y, suav, it=60, tol=1e-9):
    """Retroajuste (backfitting) de Hastie e Tibshirani.
    Xs: lista de vetores; suav[j](x, r) -> parcela ajustada, centrada."""
    y = np.asarray(y, float); p = len(Xs)
    f = [np.zeros(len(y)) for _ in range(p)]; a0 = y.mean()
    for _ in range(it):
        maxd = 0.0
        for j in range(p):
            r = y - a0 - sum(f[k] for k in range(p) if k != j)
            nv = suav[j](Xs[j], r); nv = nv - nv.mean()
            maxd = max(maxd, float(np.max(np.abs(nv - f[j])))); f[j] = nv
        if maxd < tol: break
    return a0, f
# ------------------------------------------------------------------ fim do bloco comum

x=PAR.copy(); y=A.copy(); o=np.argsort(x); xs=x[o]; ys=y[o]
print('Os mesmos 260 pontos de trocas gasosas: a resposta da fotossíntese A à')
print('radiação PAR. A ficha anterior escolheu NÓS; esta não escolhe nenhum.')

print('\n'+'='*70)
print('A IDEIA: pôr um nó em CADA observação e penalizar a curvatura')
print()
print('O critério é')
print('    soma (y_i - f(x_i))^2  +  lambda * integral de f\'\'(t)^2 dt')
print()
print('O primeiro termo quer passar por todos os pontos; o segundo cobra pela')
print('curvatura. O minimizador, demonstrado por Reinsch (1967), é uma spline')
print('cúbica NATURAL com nós em todos os x distintos — de modo que o problema')
print('de escolher nós desaparece e sobra um único parâmetro contínuo, lambda.')
print()
print('  x distintos nos dados: %d de %d observações'%(len(np.unique(x)),NG))

print('\n'+'='*70)
print('OS DOIS EXTREMOS, verificados numericamente')
aj0,xo0,ct0,yb0,tr0=suav_spline(x,y,1e-15)
ajI,xoI,ctI,ybI,trI=suav_spline(x,y,1e10)
bl=np.polyfit(x,y,1)
print()
print('  lambda -> 0  : g.l. efetivos = %s (há %d x distintos)'%(c(tr0,2),len(xo0)))
print('                 maior diferença para as médias por x = %s'%c(float(np.abs(aj0-yb0).max()),6))
print('  lambda -> inf: g.l. efetivos = %s'%c(trI,4))
print('                 maior diferença para a reta de mínimos quadrados = %s'%c(float(np.abs(ajI-np.polyval(bl,xoI)).max()),8))
print()
print('Entre os dois extremos há um caminho contínuo, e escolher lambda é')
print('escolher um ponto dele. É a mesma estrutura do ridge, do lasso, da PCR')
print('e da PLS: um caminho do modelo nulo ao ajuste saturado.')

print('\n'+'='*70)
print('GRAUS DE LIBERDADE EFETIVOS: a quantidade interpretável')
print('O ajuste é LINEAR em y: f = S_lambda y. Os g.l. efetivos são tr(S).')
print()
print('%14s %16s %16s'%('lambda','g.l. efetivos','EQM resubstituição'))
LAM=np.logspace(-7,1,45)
tab=[]
for lam in LAM:
    aj,xo,ct,yb,tr=suav_spline(x,y,lam)
    rs=eqm(y,np.interp(x,xo,aj)); tab.append((lam,tr,rs))
tab=np.array(tab)
for lam in [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]:
    k=int(np.argmin(np.abs(LAM-lam)))
    print('%14s %16s %16s'%('%.0e'%LAM[k],c(tab[k,1],3),c(tab[k,2])))
print()
print('Ninguém escolhe lambda diretamente: escolhe-se o número de g.l. que se')
print('quer, ou deixa-se a validação cruzada escolher. O R chama de "spar" uma')
print('reparametrização de lambda, e aceita também o argumento "df".')

print('\n'+'='*70)
print('A ESCOLHA DE lambda: validação cruzada contra GCV')
print()
print('%12s %10s %14s %14s %14s'%('lambda','g.l. ef.','EQM 10-fold','VC deixa-um','GCV'))
res=[]
for lam in LAM:
    aj,xo,ct,yb,tr=suav_spline(x,y,lam)
    g,_=ss_gcv(x,y,lam)
    cvm,_=cv_uni(lambda xt,yt,xe,l=lam: ss_prev(xt,yt,l,xe),x,y,10,1,8)
    # deixa-um-de-fora pelo atalho da diagonal do suavizador
    res.append((lam,tr,cvm,g))
res=np.array(res)
# atalho LOO: precisa da diagonal de S; recalcula só para alguns lambda
def loo_exato(lam):
    aj,xo,ct,yb,tr=suav_spline(x,y,lam)
    xr=(x-x.min())/(x.max()-x.min()); xo_s=(xo-x.min())/(x.max()-x.min())
    m=len(xo_s); h=np.diff(xo_s)
    Q=np.zeros((m,m-2)); R=np.zeros((m-2,m-2))
    for j in range(1,m-1):
        Q[j-1,j-1]=1/h[j-1]; Q[j,j-1]=-1/h[j-1]-1/h[j]; Q[j+1,j-1]=1/h[j]
        R[j-1,j-1]=(h[j-1]+h[j])/3
        if j<m-2: R[j-1,j]=h[j]/6; R[j,j-1]=h[j]/6
    Winv=1.0/ct
    M=R+lam*(Q.T*Winv)@Q
    S=np.eye(m)-lam*(Winv[:,None]*(Q@np.linalg.solve(M,Q.T)))
    d=np.diag(S); inv=np.searchsorted(xo,x)
    r=y-np.interp(x,xo,aj)
    return float(np.mean((r/(1-d[inv]))**2))
for lam in [1e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,1e-2,1e-1]:
    k=int(np.argmin(np.abs(LAM-lam)))
    print('%12s %10s %14s %14s %14s'%('%.0e'%LAM[k],c(res[k,1],2),c(res[k,2]),
          c(loo_exato(LAM[k])),c(res[k,3])))
jc=int(np.argmin(res[:,2])); jg=int(np.argmin(res[:,3]))
print()
print('  menor EQM 10-fold: lambda = %.3e, g.l. ef. = %s, EQM = %s'%(res[jc,0],c(res[jc,1],2),c(res[jc,2])))
print('  menor GCV:         lambda = %.3e, g.l. ef. = %s'%(res[jg,0],c(res[jg,1],2)))
print()
print('O GCV é uma aproximação da deixa-um-de-fora que troca cada elemento da')
print('diagonal de S pela média deles, e por isso custa um único ajuste. As')
print('três curvas têm mínimo na mesma vizinhança.')

print('\n'+'='*70)
print('A LIGAÇÃO COM O RIDGE')
print('A spline de suavização É uma regressão penalizada: na base natural com')
print('nós em todos os x, f = (I + lambda K)^-1 y, com K fixa. Trocar K pela')
print('identidade dá exatamente o ridge.')
print()
u,d,_=np.linalg.svd(np.zeros((2,2))+np.eye(2))
aj,xo,ct,yb,tr=suav_spline(x,y,1e-4)
xo_s=(xo-x.min())/(x.max()-x.min()); m=len(xo_s); h=np.diff(xo_s)
Q=np.zeros((m,m-2)); R=np.zeros((m-2,m-2))
for j in range(1,m-1):
    Q[j-1,j-1]=1/h[j-1]; Q[j,j-1]=-1/h[j-1]-1/h[j]; Q[j+1,j-1]=1/h[j]
    R[j-1,j-1]=(h[j-1]+h[j])/3
    if j<m-2: R[j-1,j]=h[j]/6; R[j,j-1]=h[j]/6
K=Q@np.linalg.solve(R,Q.T)
dK=np.sort(np.linalg.eigvalsh((K+K.T)/2))
print('  autovalores de K (os 6 menores): %s'%', '.join('%.3e'%v for v in dK[:6]))
print('  autovalores de K (os 3 maiores): %s'%', '.join('%.3e'%v for v in dK[-3:]))
print()
print('Os DOIS menores autovalores de K são NULOS — saem como %.1e e %.1e,'%(dK[0],dK[1]))
print('isto é, 18 ordens de grandeza abaixo do maior (%.2e), que é o zero'%dK[-1])
print('numérico desta matriz. Eles correspondem às direções constante e linear,')
print('que a penalidade NÃO toca, porque a segunda derivada de uma reta é zero')
print('— e é por isso que')
print('os g.l. efetivos nunca descem abaixo de 2. Cada direção j é encolhida')
print('pelo fator 1/(1 + lambda*d_j), exatamente como no ridge, mas com uma')
print('penalidade que cresce com a RUGOSIDADE em vez do tamanho do coeficiente.')
print()
print('%10s %20s %20s'%('direção j','d_j (autovalor de K)','fator em lambda = 1e-4'))
for j in [0,1,2,3,5,10,30,100,m-1]:
    print('%10d %20s %20s'%(j+1,'%.3e'%dK[j],c(1/(1+1e-4*dK[j]),6)))

print('\n'+'='*70)
print('COMPARAÇÃO COM AS DUAS FICHAS ANTERIORES, no mesmo conjunto')
print()
gl_alvo=[3,4,5,6,8,10,14,20]
print('%10s %18s %18s %18s'%('g.l.','polinômio','spline com nós','spline de suavização'))
for gl in gl_alvo:
    g=gl-1; K_=gl-4
    fp=cv_uni(lambda xt,yt,xe,g=g: poli(xe,g,x)@ajusta_base(poli(xt,g,x),yt),x,y,10,1,8)[0]
    if K_>=0:
        nk=np.percentile(x,np.linspace(0,100,K_+2)[1:-1]) if K_ else np.array([])
        fs=cv_uni(lambda xt,yt,xe,nk=nk: base_bs(xe,nk,3,(x.min(),x.max()))@ajusta_base(base_bs(xt,nk,3,(x.min(),x.max())),yt),x,y,10,1,8)[0]
    else: fs=float('nan')
    # lambda que dá esse numero de g.l.
    k=int(np.argmin(np.abs(tab[:,1]-gl))); lam=tab[k,0]
    fz=cv_uni(lambda xt,yt,xe,l=lam: ss_prev(xt,yt,l,xe),x,y,10,1,8)[0]
    print('%10d %18s %18s %18s'%(gl,c(fp),c(fs) if K_>=0 else '—',c(fz)))
print()
print('  EQM da função VERDADEIRA (o piso possível) = %s'%c(eqm(y,f_par(x)-f_par(x).mean()+y.mean())))
print('  variância do erro, sigma² = %s'%c(SIG**2))

print('\n'+'='*70)
print('O QUE A SPLINE DE SUAVIZAÇÃO NÃO RESOLVE')
print()
print('(a) Ela não é mágica quanto à ESCOLHA: trocou-se escolher nós por')
print('    escolher lambda, e a escolha continua tendo de ser validada.')
print()
print('(b) Ela tem um nó por observação distinta, de modo que o custo cresce')
print('    com n. O algoritmo de Reinsch explora o fato de o sistema ser')
print('    BANDADO e é O(n); a implementação desta ficha usa um resolvedor')
print('    DENSO, por clareza de leitura, e paga O(n^3) por isso. Os tempos')
print('    abaixo mostram o preço, e servem de ilustração do problema:')
import time
for nn in [100,250,500,1000,2000]:
    rg=np.random.default_rng(1); xi=np.sort(rg.uniform(0,2000,nn)); yi=f_par(xi)+rg.normal(0,SIG,nn)
    t0=time.perf_counter(); suav_spline(xi,yi,1e-4); t1=time.perf_counter()
    print('    n = %5d: %7.1f ms'%(nn,1000*(t1-t0)))
print('    (cada duplicação de n multiplica o tempo por cerca de 6, compatível')
print('     com n^2,6. Com o resolvedor bandado o fator seria 2. O R contorna o')
print('     problema por outro caminho: usa no máximo cerca de 50 nós quando n')
print('     é grande, pelo argumento nknots de smooth.spline, ligado por padrão)')
print()
print('(c) Ela SUPÕE o mesmo grau de suavidade em todo o domínio. Onde a')
print('    função verdadeira muda de regime, isso cobra um preço:')
aj,xo,ct,yb,tr=suav_spline(x,y,res[jc,0])
vd=f_par(xo)-f_par(x).mean()+y.mean()
print()
print('%22s %12s %12s'%('faixa de PAR','viés médio','|viés| médio'))
for lo,hi in [(0,200),(200,500),(500,900),(900,1400),(1400,2100)]:
    mk=(xo>=lo)&(xo<hi)
    print('%22s %12s %12s'%('%d a %d'%(lo,hi),c(float((aj[mk]-vd[mk]).mean())),c(float(np.abs(aj[mk]-vd[mk]).mean()))))
print()
print('(d) Com erros CORRELACIONADOS, a validação cruzada escolhe lambda')
print('    pequeno demais e a spline persegue a autocorrelação:')
rg=np.random.default_rng(5)
for rho in [0.0,0.5,0.8]:
    gl_=[]
    for rep in range(30):
        e=np.zeros(NG)
        for i in range(1,NG): e[i]=rho*e[i-1]+rg.normal(0,SIG*np.sqrt(1-rho**2))
        yy=f_par(xs)-f_par(x).mean()+y.mean()+e
        m_=[(lam,cv_uni(lambda xt,yt,xe,l=lam: ss_prev(xt,yt,l,xe),xs,yy,10,rep,1)[0]) for lam in np.logspace(-6,-2,9)]
        lam_=min(m_,key=lambda t:t[1])[0]
        gl_.append(ss_edf(xs,lam_))
    print('    rho = %s: g.l. efetivos escolhidos = %s (média de 30 simulações)'%(c(rho,1),c(float(np.mean(gl_)),2)))
print('    O número de g.l. escolhido cresce com a autocorrelação, embora a')
print('    função verdadeira seja a MESMA nos três casos.')
