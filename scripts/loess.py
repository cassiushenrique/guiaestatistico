import numpy as np, sys, time
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
print('Os mesmos 260 pontos de trocas gasosas. As três fichas anteriores')
print('ajustaram uma função GLOBAL a todos os dados de uma vez. Esta não ajusta')
print('função nenhuma: para cada ponto onde se quer uma predição, faz-se uma')
print('regressão PRÓPRIA, usando só os vizinhos e dando mais peso aos mais')
print('próximos. É o que Cleveland (1979) chamou de regressão local ponderada.')

print('\n'+'='*70)
print('O PROCEDIMENTO, passo a passo, em um ponto')
x0=600.0; span=0.35; q=int(np.ceil(span*NG))
dd=np.abs(x-x0); ordem=np.argsort(dd); viz=ordem[:q]
dmax=dd[viz[-1]]; w=tricubo(dd[viz]/dmax)
print()
print('  ponto de interesse: PAR = %s'%c(x0,0))
print('  span = %s, logo q = ceil(%s x %d) = %d vizinhos'%(c(span,2),c(span,2),NG,q))
print('  o %dº vizinho mais próximo está a %s umol: essa é a largura da janela'%(q,c(dmax,1)))
print()
print('%16s %12s %12s %14s'%('vizinho','PAR','distância','peso tricubo'))
for k in [0,1,2,10,30,60,q-10,q-2,q-1]:
    j=viz[k]
    print('%16s %12s %12s %14s'%('%dº'%(k+1),c(x[j],1),c(dd[j],1),c(w[k],4)))
print()
print('  soma dos pesos = %s | peso máximo = %s | peso mínimo = %s'%(c(float(w.sum()),2),c(float(w.max())),c(float(w.min()),6)))
Z=np.column_stack([np.ones(q),(x[viz]-x0)])
WZ=Z*w[:,None]; b=np.linalg.solve(Z.T@WZ,WZ.T@y[viz])
print()
print('  reta local ajustada nos %d vizinhos, com esses pesos:'%q)
print('    intercepto (= a predição em PAR = %s) = %s'%(c(x0,0),c(b[0])))
print('    inclinação                             = %s por umol'%c(b[1],6))
print('  predição do loess em PAR = %s: %s'%(c(x0,0),c(float(loess(x,y,[x0],span,1)[0]))))
print('  valor VERDADEIRO ali:                %s'%c(float(f_par(np.array([x0]))[0]-f_par(x).mean()+y.mean())))
print()
print('O mesmo é repetido em cada ponto onde se quer a curva. Não há uma')
print('fórmula para a função inteira: há n regressões pequenas.')

print('\n'+'='*70)
print('O PESO TRICUBO')
print('  w(u) = (1 - |u|^3)^3 para |u| <= 1, e 0 fora')
print()
print('%10s %14s %18s'%('u','tricubo','gaussiano exp(-u²/2)'))
for u in [0,0.2,0.4,0.6,0.8,0.95,1.0,1.2]:
    print('%10s %14s %18s'%(c(u,2),c(float(tricubo(np.array([min(u,1.0)]))[0]) if u<=1 else 0.0),
          c(float(np.exp(-u**2/2)))))
print()
print('Duas propriedades importam. O peso vai a ZERO exatamente em u = 1, de')
print('modo que a janela é finita e os pontos distantes não entram; e a')
print('derivada também se anula ali, de modo que a curva não dá solavancos')
print('quando um ponto entra ou sai da janela. O núcleo gaussiano não tem')
print('nenhuma das duas propriedades.')

print('\n'+'='*70)
print('O SPAN: o único parâmetro que importa')
print()
print('%10s %10s %14s %14s %16s'%('span','vizinhos','g.l. efetivos','EQM resub','EQM 10-fold'))
res=[]
for sp in [0.08,0.12,0.18,0.25,0.35,0.5,0.7,0.9,1.2]:
    gl=loess_edf(x,sp,1)
    aj=loess(x,y,x,sp,1); rs=eqm(y,aj)
    cvm,_=cv_uni(lambda xt,yt,xe,s_=sp: loess(xt,yt,xe,s_,1),x,y,10,1,4)
    res.append((sp,gl,rs,cvm))
    print('%10s %10d %14s %14s %16s'%(c(sp,2),int(np.ceil(sp*NG)),c(gl,2),c(rs),c(cvm)))
res=np.array(res); jb=int(np.argmin(res[:,3]))
print()
print('  menor EQM validado: span = %s (%s g.l. efetivos), EQM = %s'%(c(res[jb,0],2),c(res[jb,1],2),c(res[jb,3])))
print('  para comparar: spline de suavização com 8 g.l. deu 5,8237')
print('                 e a função VERDADEIRA dá %s'%c(eqm(y,f_par(x)-f_par(x).mean()+y.mean())))
print()
print('Note que span > 1 é permitido e usado: com span = 1,2 todos os pontos')
print('entram em toda janela, mas com pesos tricubo calculados como se a janela')
print('fosse 1,2 vezes maior — o resultado é um ajuste mais suave, não idêntico')
print('ao global.')

print('\n'+'='*70)
print('GRAU 1 CONTRA GRAU 2: o viés nas bordas e nos picos')
print()
print('%10s %22s %22s'%('span','grau 1 (EQM 10-fold)','grau 2 (EQM 10-fold)'))
for sp in [0.15,0.25,0.35,0.5,0.7]:
    a=cv_uni(lambda xt,yt,xe,s_=sp: loess(xt,yt,xe,s_,1),x,y,10,1,4)[0]
    b2=cv_uni(lambda xt,yt,xe,s_=sp: loess(xt,yt,xe,s_,2),x,y,10,1,4)[0]
    print('%10s %22s %22s'%(c(sp,2),c(a),c(b2)))
print()
print('O viés por REGIÃO, com span = 0,35 (comparando à função verdadeira):')
vd=f_par(xs)-f_par(x).mean()+y.mean()
a1=loess(x,y,xs,0.35,1); a2=loess(x,y,xs,0.35,2)
print()
print('%22s %16s %16s'%('faixa de PAR','grau 1: |viés|','grau 2: |viés|'))
for lo,hi in [(0,200),(200,500),(500,900),(900,1400),(1400,2100)]:
    mk=(xs>=lo)&(xs<hi)
    print('%22s %16s %16s'%('%d a %d'%(lo,hi),c(float(np.abs(a1[mk]-vd[mk]).mean())),
          c(float(np.abs(a2[mk]-vd[mk]).mean()))))
print()
print('O padrão não é o que o manual costuma dizer, e vale lê-lo com cuidado.')
print('No MIOLO (200 a 900 umol), onde a curva dobra, o grau 2 é claramente')
print('melhor — 0,1656 contra 0,3136 e 0,1279 contra 0,1780 — porque uma reta')
print('dentro da janela corta a curvatura. Nas duas BORDAS, porém, o grau 2 é')
print('PIOR: 0,8493 contra 0,5782 abaixo de 200, e 0,5064 contra 0,3914 acima')
print('de 1.400. Ali a janela é assimétrica e o termo quadrático, estimado com')
print('dados de um lado só, acrescenta variância sem reduzir viés.')
print()
print('Pelo erro validado, os dois praticamente empatam com span pequeno e o')
print('grau 2 ganha com span grande (5,7728 contra 6,4694 em span = 0,70), o')
print('que é coerente: janela larga exige curvatura dentro dela.')

print('\n'+'='*70)
print('A VERSÃO ROBUSTA: o que o loess faz com observações aberrantes')
print('Depois do primeiro ajuste, os resíduos são usados para reponderar: um')
print('ponto a mais de 6 desvios absolutos medianos recebe peso ZERO, e o')
print('ajuste é refeito. Cleveland chamou isso de iteração bisquare.')
print()
rg=np.random.default_rng(19)
for nout,desl in [(0,0.0),(5,10.0),(10,10.0),(10,18.0)]:
    yc_=y.copy()
    if nout:
        idx=rg.choice(np.where((x>400)&(x<1200))[0],nout,replace=False)
        yc_[idx]=yc_[idx]+desl
    vdx=f_par(xs)-f_par(x).mean()+y.mean()
    s0=loess(x,yc_,xs,0.35,1,0); s2=loess(x,yc_,xs,0.35,1,3)
    print('  %2d pontos deslocados em +%4s umol: |viés| médio  sem robustez %s | com %s'%(
          nout,c(desl,0),c(float(np.abs(s0-vdx).mean())),c(float(np.abs(s2-vdx).mean()))))
print()
print('Aqui a teoria se confirma, e com folga. SEM contaminação, as duas')
print('versões praticamente coincidem — 0,2614 e 0,2652, diferença de 1,5% —')
print('de modo que a robustez não custa quase nada quando não é necessária.')
print()
print('Com contaminação, a diferença aparece e cresce: com 10 pontos deslocados')
print('em 10 umol, o ajuste comum piora 62% (de 0,2614 para 0,4241) e o robusto')
print('piora 6% (de 0,2652 para 0,2814). Com o deslocamento de 18 umol, o comum')
print('vai a 0,7300 — quase o triplo do original — e o robusto fica em 0,2355,')
print('isto é, NÃO SE DEGRADA.')
print()
print('A razão de o robusto chegar a melhorar um pouco com a contaminação é')
print('que a reponderação bisquare também apara observações legitimamente')
print('extremas do ruído comum. É um efeito pequeno e não é a finalidade do')
print('procedimento, mas explica por que o número cai em vez de subir.')
print('\n'+'='*70)
print('O QUE O LOESS NÃO TEM')
print()
print('(a) NÃO tem fórmula. Não há equação para reportar, nem coeficientes com')
print('    interpretação. O resultado é uma curva, e o que se publica é o')
print('    gráfico mais o span e o grau.')
print()
print('(b) NÃO tem como extrapolar. Fora do domínio não existem vizinhos, e')
print('    por isso o R devolve NA para newdata fora do intervalo, a menos que')
print('    se peça surface = "direct". A implementação desta ficha é a direta,')
print('    e o que ela devolve fora do domínio é a reta local da borda,')
print('    estendida — o que fica praticamente CONSTANTE e erra cada vez mais:')
xf=np.array([1990.,2100.,2400.])
for xv in xf:
    vv=float((f_par(np.array([xv]))-f_par(x).mean()+y.mean())[0])
    pr=float(loess(x,y,[xv],0.35,1)[0])
    print('    PAR = %s: loess %s | verdade %s'%(c(xv,0),c(pr,2),c(vv,2)))
print()
print('(c) É CARO. Uma predição exige uma regressão própria; n predições')
print('    exigem n regressões, cada uma com cerca de span*n pontos — de modo')
print('    que o trabalho cresce com n². Na prática, a vetorização esconde')
print('    parte disso, e o tempo medido cresce mais devagar:')
for nn in [100,250,500,1000]:
    rg2=np.random.default_rng(1); xi=np.sort(rg2.uniform(0,2000,nn)); yi=f_par(xi)+rg2.normal(0,SIG,nn)
    t0=time.perf_counter(); loess(xi,yi,xi,0.35,1); t1=time.perf_counter()
    print('    n = %5d: %8.1f ms'%(nn,1000*(t1-t0)))
print('    (de n = 100 a n = 1.000, o tempo cresce cerca de 19 vezes, e não 100.')
print('     Mesmo assim, é o mais caro dos quatro métodos do grupo. O R usa a')
print('     aproximação por interpolação de Cleveland, que avalia o')
print('     ajuste exato em poucos pontos e interpola entre eles: é o argumento')
print('     surface = "interpolate", ligado por padrão)')
print()
print('(d) NÃO há teste nem valor-p padronizado. Há intervalos aproximados')
print('    baseados nos g.l. efetivos, e eles supõem que o span foi fixado')
print('    ANTES de ver os dados — o que quase nunca é o caso.')

print('\n'+'='*70)
print('AS QUATRO FICHAS, lado a lado, com a MESMA flexibilidade')
print()
tabss=[]
for lam in np.logspace(-7,1,60):
    tabss.append((lam,ss_edf(x,lam)))
tabss=np.array(tabss)
print('%10s %14s %14s %14s %14s'%('g.l.','polinômio','spline c/ nós','suavização','loess'))
for gl in [4,5,6,8,10,14]:
    g=gl-1; K_=gl-4
    fp=cv_uni(lambda xt,yt,xe,g=g: poli(xe,g,x)@ajusta_base(poli(xt,g,x),yt),x,y,10,1,6)[0]
    nk=np.percentile(x,np.linspace(0,100,K_+2)[1:-1]) if K_>0 else np.array([])
    fs=cv_uni(lambda xt,yt,xe,nk=nk: base_bs(xe,nk,3,(x.min(),x.max()))@ajusta_base(base_bs(xt,nk,3,(x.min(),x.max())),yt),x,y,10,1,6)[0] if K_>=0 else float('nan')
    lam=tabss[int(np.argmin(np.abs(tabss[:,1]-gl))),0]
    fz=cv_uni(lambda xt,yt,xe,l=lam: ss_prev(xt,yt,l,xe),x,y,10,1,6)[0]
    spn=min([(abs(loess_edf(x,s_,1)-gl),s_) for s_ in np.linspace(0.08,1.5,40)])[1]
    fl=cv_uni(lambda xt,yt,xe,s_=spn: loess(xt,yt,xe,s_,1),x,y,10,1,4)[0]
    print('%10d %14s %14s %14s %14s'%(gl,c(fp),c(fs) if K_>=0 else '—',c(fz),c(fl)))
print()
print('  piso possível (função verdadeira) = %s'%c(eqm(y,f_par(x)-f_par(x).mean()+y.mean())))
print()
print('Com a flexibilidade igualada, os quatro métodos ficam dentro de 5% um do')
print('outro na faixa de 4 a 10 g.l. — mas há um padrão, e ele é o esperado.')
print('Com POUCOS g.l. o polinômio e a spline com nós levam ligeira vantagem;')
print('a partir de 10 g.l. os dois métodos de janela ou penalidade passam à')
print('frente, e em 14 g.l. o polinômio já está arruinado (8,98) enquanto os')
print('outros três continuam entre 5,83 e 6,11.')
print()
print('A conclusão prática: com a flexibilidade bem escolhida, a escolha entre')
print('os quatro não é de desempenho — é de que problema se quer evitar.')
print('Oscilação global e falha de extrapolação afastam o polinômio; a escolha')
print('de nós afasta a spline de regressão; o custo computacional e a ausência')
print('de fórmula afastam o loess; e nenhum dos quatro resolve o caso de')
print('VÁRIAS preditoras, que é o assunto da próxima ficha.')
