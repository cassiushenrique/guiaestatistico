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

x=PAR.copy(); y=A.copy(); LIM=(float(x.min()),float(x.max()))
print('Os mesmos 260 pontos de trocas gasosas da ficha polinomial: a resposta')
print('da fotossíntese A à radiação PAR. A curva sobe depressa, satura e cai um')
print('pouco por fotoinibição — curvaturas MUITO diferentes em regiões diferentes.')
print()
print('  n = %d | PAR de %s a %s | r(PAR, A) = %s'%(NG,c(x.min(),0),c(x.max(),0),c(float(np.corrcoef(x,y)[0,1]))))

print('\n'+'='*70)
print('A CONTABILIDADE DOS GRAUS DE LIBERDADE')
print('Uma spline de grau g com K nós internos, SEM restrições, teria')
print('(K+1)(g+1) parâmetros. As restrições de continuidade da função e das')
print('g-1 primeiras derivadas em cada nó tiram g*K deles.')
print()
print('%8s %6s %16s %16s %10s %12s'%('grau','K','sem restrição','restrições','sobram','posto de B'))
for g,K in [(1,4),(2,4),(3,4),(3,8),(3,12)]:
    nos=np.percentile(x,np.linspace(0,100,K+2)[1:-1])
    B=base_bs(x,nos,g,LIM)
    print('%8d %6d %16d %16d %10d %12d'%(g,K,(K+1)*(g+1),g*K,(K+1)*(g+1)-g*K,
          int(np.linalg.matrix_rank(B))))
print()
print('A conta fecha: (K+1)(g+1) - gK = K + g + 1, que é exatamente o número de')
print('funções da base B-spline, e o posto confirma.')

print('\n'+'='*70)
print('DUAS BASES PARA O MESMO ESPAÇO: potências truncadas e B-splines')
print()
print('%6s %22s %22s %22s'%('K','cond(X) pot. truncadas','cond(X) B-spline','maior |dif| no ajuste'))
for K in [2,4,6,10,16,24]:
    nos=np.percentile(x,np.linspace(0,100,K+2)[1:-1])
    Bt=trunc_pot(x,nos,3); Bb=base_bs(x,nos,3,LIM)
    at=Bt@ajusta_base(Bt,y); ab=Bb@ajusta_base(Bb,y)
    print('%6d %22s %22s %22s'%(K,'%.2e'%np.linalg.cond(Bt),'%.2e'%np.linalg.cond(Bb),
          '%.2e'%float(np.abs(at-ab).max())))
print()
print('As duas bases geram o MESMO espaço de funções, e aqui os ajustes de fato')
print('COINCIDEM: a maior diferença é da ordem de 10^-7 umol, desprezível. Vale')
print('dizer isso com clareza, porque o contraste com a ficha polinomial é')
print('instrutivo — lá a base crua DEIXOU DE FUNCIONAR a partir do grau 6.')
print()
print('A objeção à base de potências truncadas não é que ela quebre com estes')
print('dados, e sim a margem: o número de condição dela é cerca de 10^9 vezes')
print('maior (10^11 contra cerca de 6), e com 16 dígitos de precisão isso')
print('consome mais da metade deles. Com K maior, x de escala maior ou dados')
print('mais desbalanceados, a margem acaba.')
print()
print('A razão positiva para preferir B-splines é a LOCALIDADE: cada função da')
print('base é não nula em apenas g+2 intervalos consecutivos, ao passo que cada')
print('potência truncada é não nula em TODO o domínio à direita do nó.')
nos=np.percentile(x,np.linspace(0,100,8)[1:-1]); Bb=base_bs(x,nos,3,LIM)
print()
print('  largura do suporte de cada B-spline (K = 6, grau 3):')
for j in range(Bb.shape[1]):
    nz=x[Bb[:,j]>1e-12]
    print('    função %2d: de %s a %s'%(j+1,c(float(nz.min()),0),c(float(nz.max()),0)))

print('\n'+'='*70)
print('QUANTOS NÓS? A escolha por validação cruzada')
print()
print('%6s %8s %14s %16s'%('K','p','EQM resub','EQM 10-fold'))
res=[]
for K in range(0,21):
    nos=np.percentile(x,np.linspace(0,100,K+2)[1:-1]) if K else np.array([])
    def aj(xt,yt,xe,nos=nos):
        return base_bs(xe,nos,3,LIM)@ajusta_base(base_bs(xt,nos,3,LIM),yt)
    B=base_bs(x,nos,3,LIM); rs=eqm(y,B@ajusta_base(B,y))
    cvm,_=cv_uni(aj,x,y,10,1,10)
    res.append((K,K+4,rs,cvm))
    if K<=12 or K%4==0: print('%6d %8d %14s %16s'%(K,K+4,c(rs),c(cvm)))
res=np.array(res); Kb=int(res[np.argmin(res[:,3]),0])
print()
print('  menor EQM validado: K = %d nós (p = %d), EQM = %s'%(Kb,Kb+4,c(res[:,3].min())))
print('  para comparar, o polinômio de grau 5 (p = 6) deu EQM validado 5,6789')
print('  e a função VERDADEIRA daria %s'%c(eqm(y,f_par(x)-f_par(x).mean()+y.mean())))

print('\n'+'='*70)
print('ONDE PÔR OS NÓS: quantis contra espaçamento igual')
print()
print('%6s %22s %22s'%('K','nós nos QUANTIS','nós IGUALMENTE espaçados'))
for K in [2,4,6,8,12]:
    nq=np.percentile(x,np.linspace(0,100,K+2)[1:-1])
    ne=np.linspace(LIM[0],LIM[1],K+2)[1:-1]
    def ajq(xt,yt,xe,nos=nq): return base_bs(xe,nos,3,LIM)@ajusta_base(base_bs(xt,nos,3,LIM),yt)
    def aje(xt,yt,xe,nos=ne): return base_bs(xe,nos,3,LIM)@ajusta_base(base_bs(xt,nos,3,LIM),yt)
    print('%6d %22s %22s'%(K,c(cv_uni(ajq,x,y,10,1,10)[0]),c(cv_uni(aje,x,y,10,1,10)[0])))
print()
print('A leitura exige cuidado, porque a tabela não diz o que se espera. Até')
print('K = 6 as duas estratégias EMPATAM — as diferenças ficam abaixo de 0,5%,')
print('e o espaçamento igual é até ligeiramente melhor. A vantagem dos quantis')
print('só aparece quando há muitos nós: em K = 8 a diferença é de 3%, e em')
print('K = 12 é de 16% (6,2923 contra 7,4619).')
print()
print('O motivo é direto: com poucos nós, qualquer colocação razoável funciona.')
print('Com muitos nós e espaçamento igual, alguns caem em regiões com poucos')
print('dados — aqui, acima de 1.700 umol, onde há só 8% das observações — e o')
print('ajuste fica localmente indeterminado. Os quantis põem mais nós onde há')
print('mais dados, por construção, e por isso degradam melhor.')
print()
print('  distribuição do PAR: quartis em %s, %s e %s'%(c(float(np.percentile(x,25)),0),
      c(float(np.percentile(x,50)),0),c(float(np.percentile(x,75)),0)))
K=6
print('  nós nos quantis (K = 6):  %s'%', '.join(c(float(v),0) for v in np.percentile(x,np.linspace(0,100,K+2)[1:-1])))
print('  nós igualmente espaçados: %s'%', '.join(c(float(v),0) for v in np.linspace(LIM[0],LIM[1],K+2)[1:-1]))

print('\n'+'='*70)
print('A SPLINE NATURAL: a restrição que conserta as bordas')
print('Uma spline cúbica natural é LINEAR fora dos dois nós de fronteira. A')
print('restrição impõe segunda e terceira derivadas nulas nas duas pontas, o')
print('que retira 4 graus de liberdade do espaço cúbico com o mesmo conjunto')
print('de nós, e compra estabilidade exatamente onde ela falta.')
print()
xs=np.sort(x)
rg=np.random.default_rng(11)
def variancia(fab,reps=500):
    P=np.zeros((reps,len(xs)))
    for r in range(reps):
        i=rg.integers(0,NG,NG); P[r]=fab(x[i],y[i],xs)
    return P.var(0,ddof=1)
nos6=np.percentile(x,np.linspace(0,100,8)[1:-1])
f_bs=lambda xt,yt,xe: base_bs(xe,nos6,3,LIM)@ajusta_base(base_bs(xt,nos6,3,LIM),yt)
f_ns=lambda xt,yt,xe: natural_cs(xe,nos6,LIM)@ajusta_base(natural_cs(xt,nos6,LIM),yt)
f_po=lambda xt,yt,xe: poli(xe,9,x)@ajusta_base(poli(xt,9,x),yt)
v_bs=variancia(f_bs); v_ns=variancia(f_ns); v_po=variancia(f_po)
print('Parâmetros: B-spline com 6 nós = %d | natural com 6 nós = %d | polinômio grau 9 = %d'%(
      base_bs(x,nos6,3,LIM).shape[1],natural_cs(x,nos6,LIM).shape[1],10))
print()
print('%22s %14s %14s %14s'%('faixa de PAR','B-spline','natural','polinômio g9'))
cortes=[(0,200),(200,500),(500,1400),(1400,1800),(1800,2100)]
for lo,hi in cortes:
    mk=(xs>=lo)&(xs<hi)
    print('%22s %14s %14s %14s'%('%d a %d'%(lo,hi),c(float(v_bs[mk].mean())),
          c(float(v_ns[mk].mean())),c(float(v_po[mk].mean()))))
mio=(xs>=500)&(xs<1400)
print()
print('  razão borda/miolo — B-spline: %s | natural: %s | polinômio: %s'%(
      c(float(v_bs[xs>=1800].mean())/float(v_bs[mio].mean()),1),
      c(float(v_ns[xs>=1800].mean())/float(v_ns[mio].mean()),1),
      c(float(v_po[xs>=1800].mean())/float(v_po[mio].mean()),1)))
print()
print('  EQM validado — B-spline: %s | natural: %s | polinômio g9: %s'%(
      c(cv_uni(f_bs,x,y,10,1,10)[0]),c(cv_uni(f_ns,x,y,10,1,10)[0]),c(cv_uni(f_po,x,y,10,1,10)[0])))

print('\n'+'='*70)
print('E PARA EXTRAPOLAR?')
print('A spline natural é linear fora dos nós de fronteira, de modo que')
print('extrapola em linha reta — errado, mas de forma controlada.')
print()
xf=np.array([2100.,2300.,2600.,3000.])
print('%10s %12s %12s %12s %12s'%('PAR','B-spline','natural','polinômio g9','VERDADE'))
for xv in xf:
    xa=np.array([xv])
    bb=float((base_bs(xa,nos6,3,(LIM[0],3000.))@ajusta_base(base_bs(x,nos6,3,(LIM[0],3000.)),y))[0])
    nn=float((natural_cs(xa,nos6,LIM)@ajusta_base(natural_cs(x,nos6,LIM),y))[0])
    pp=float((poli(xa,9,x)@ajusta_base(poli(x,9,x),y))[0])
    vv=float((f_par(xa)-f_par(x).mean()+y.mean())[0])
    print('%10s %12s %12s %12s %12s'%(c(xv,0),c(bb,2),c(nn,2),c(pp,2),c(vv,2)))
print()
print('Nenhum dos três acerta, e nenhum poderia: não há informação fora do')
print('domínio. A diferença é o TAMANHO do erro. A %s, a spline natural erra'%c(3000.,0))
xa=np.array([3000.])
nn=float((natural_cs(xa,nos6,LIM)@ajusta_base(natural_cs(x,nos6,LIM),y))[0])
pp=float((poli(xa,9,x)@ajusta_base(poli(x,9,x),y))[0])
vv=float((f_par(xa)-f_par(x).mean()+y.mean())[0])
print('%s umol e o polinômio de grau 9 erra %s.'%(c(abs(nn-vv),1),c(abs(pp-vv),1)))

print('\n'+'='*70)
print('SPLINE CONTRA POLINÔMIO, com a MESMA flexibilidade')
print()
print('%6s %10s %16s %16s %12s'%('p','grau/K','EQM val. polinômio','EQM val. spline','vantagem'))
for p_ in [5,6,8,10,12,15,20]:
    g=p_-1; K=p_-4
    fp=lambda xt,yt,xe,g=g: poli(xe,g,x)@ajusta_base(poli(xt,g,x),yt)
    if K>=0:
        nk=np.percentile(x,np.linspace(0,100,K+2)[1:-1]) if K else np.array([])
        fs=lambda xt,yt,xe,nk=nk: base_bs(xe,nk,3,LIM)@ajusta_base(base_bs(xt,nk,3,LIM),yt)
        a=cv_uni(fp,x,y,10,1,10)[0]; b=cv_uni(fs,x,y,10,1,10)[0]
        print('%6d %10s %16s %16s %12s'%(p_,'g=%d / K=%d'%(g,K),c(a),c(b),c(100*(a-b)/a,1)+'%'))
print()
print('Com poucos parâmetros os dois empatam. A partir de p = 10 a spline passa')
print('à frente, e a distância cresce: em p = 20 o polinômio já está arruinado')
print('pela oscilação e a spline continua estável. É a mesma conclusão da ficha')
print('polinomial, vista agora pelo erro de predição.')
