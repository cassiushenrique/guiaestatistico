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
from scipy import stats

x=PAR.copy(); y=A.copy()
o=np.argsort(x); xs=x[o]; ys=y[o]
print('TROCAS GASOSAS: 260 medições em folhas de cafeeiro.')
print('Resposta: fotossíntese líquida A (umol CO2 m-2 s-1).')
print('Preditoras: radiação PAR (umol m-2 s-1), temperatura foliar (C) e VPD (kPa).')
print('Dados fictícios, com processo gerador CONHECIDO: A = f1(PAR) + f2(temp)')
print('+ f3(VPD) + erro, com erro de desvio-padrão 1,30.')
print()
print('  n = %d | A de %s a %s | variância de A = %s'%(NG,c(A.min(),2),c(A.max(),2),c(float(A.var(ddof=1)))))
print('  PAR  de %s a %s'%(c(PAR.min(),0),c(PAR.max(),0)))
print('  TEMP de %s a %s'%(c(TEMP.min(),1),c(TEMP.max(),1)))
print('  VPD  de %s a %s'%(c(VPD.min(),2),c(VPD.max(),2)))
Xd=np.column_stack([np.ones(NG),PAR,TEMP,VPD]); b=np.linalg.lstsq(Xd,A,rcond=None)[0]
print()
print('  R² do modelo LINEAR nas três          = %s'%c(r2(A,Xd@b)))
print('  R² máximo possível (função verdadeira) = %s'%c(r2(A,verdade(PAR,TEMP,VPD))))
print('  O modelo linear perde %s pontos de R². É esse buraco que o grupo'%c(r2(A,verdade(PAR,TEMP,VPD))-r2(A,Xd@b),4))
print('  inteiro de fichas tenta preencher.')

print('\n'+'='*70)
print('ESTA FICHA: a resposta à RADIAÇÃO, uma variável de cada vez')
print('A relação é a curva de luz: sobe depressa, satura, e cai um pouco por')
print('fotoinibição acima de 1.000 umol. É nitidamente não linear.')
print()
print('  r de Pearson entre PAR e A = %s'%c(float(np.corrcoef(PAR,A)[0,1])))
bl=np.polyfit(x,y,1)
print('  R² da RETA                  = %s'%c(r2(y,np.polyval(bl,x))))
print('  R² da função verdadeira f1  = %s'%c(r2(y,f_par(x)-f_par(x).mean()+y.mean())))

print('\n'+'='*70)
print('BASE CRUA contra BASE ORTOGONAL: mesmo ajuste, números muito diferentes')
print()
print('%5s %18s %18s %14s %14s'%('grau','cond(X) base crua','cond(X) ortogonal','maior |coef| cru','maior |coef| ort'))
for g in [1,2,3,4,6,8,10,12]:
    Bc=poli_cru(x,g); Bo=poli(x,g)
    bc=ajusta_base(Bc,y); bo=ajusta_base(Bo,y)
    print('%5d %18s %18s %14s %14s'%(g,'%.3g'%np.linalg.cond(Bc),'%.3g'%np.linalg.cond(Bo),
          '%.3g'%np.abs(bc).max(),'%.3g'%np.abs(bo).max()))
print()
print('Em ARITMÉTICA EXATA os dois ajustes seriam idênticos: as duas bases geram')
print('o mesmo espaço de funções. Em aritmética de máquina, não:')
print()
print('%6s %22s %20s'%('grau','maior |diferença| entre','cond(X) da base crua'))
print('%6s %22s %20s'%('','os dois ajustes',''))
for g in [1,2,3,4,5,6,8,10,12]:
    Bc=poli_cru(x,g); Bo=poli(x,g)
    d=float(np.abs(Bc@ajusta_base(Bc,y)-Bo@ajusta_base(Bo,y)).max())
    print('%6d %22s %20s'%(g,('%.2e'%d),'%.2e'%np.linalg.cond(Bc)))
print()
print('Até o grau 4 as duas bases concordam até a 10ª casa decimal. A partir do')
print('grau 6 a base crua simplesmente DEIXA DE FUNCIONAR: no grau 8 os dois')
print('ajustes diferem em até %s umol, que é a ordem de grandeza da própria'%c(float(np.abs(poli_cru(x,8)@ajusta_base(poli_cru(x,8),y)-poli(x,8)@ajusta_base(poli(x,8),y)).max()),2))
print('resposta. Não é questão de estilo: com número de condição de 10^27 e')
print('precisão dupla (cerca de 16 dígitos), não sobra dígito nenhum.')
print()
g=8; Bo=poli(x,g)
print('A consequência prática: os valores-p dos coeficientes da base CRUA não')
print('servem para escolher o grau, porque cada um deles testa algo que depende')
print('de todos os outros. Na base ORTOGONAL cada coeficiente é independente,')
print('e o teste do último termo É o teste sequencial do grau.')
print()
print('%5s %14s %12s %12s   %s'%('termo','coef. ortog.','erro-padrão','t','p'))
bo=ajusta_base(Bo,y); res=y-Bo@bo; s2=float(res@res)/(NG-g-1)
ep=np.sqrt(np.diag(np.linalg.inv(Bo.T@Bo))*s2); t=bo/ep
for j in range(g+1):
    pv=2*(1-stats.t.cdf(abs(t[j]),NG-g-1))
    print('%5s %14s %12s %12s   %s'%('L%d'%j if j else 'const',c(bo[j],4),c(ep[j],4),c(t[j],2),
          '%.2e'%pv if pv>=1e-300 else '< 1e-300'))

print('\n'+'='*70)
print('A ESCOLHA DO GRAU: quatro critérios')
print()
print('%5s %10s %12s %14s %12s %12s'%('grau','p','EQM resub','EQM 10-fold','AIC','BIC'))
res=[]
for g in range(1,13):
    B=poli(x,g); bb=ajusta_base(B,y); sq=sqr(y,B@bb); p_=g+1
    aic=NG*np.log(sq/NG)+2*p_; bic=NG*np.log(sq/NG)+np.log(NG)*p_
    cvm,_=cv_uni(lambda xt,yt,xe,g=g: poli(xe,g,x)@ajusta_base(poli(xt,g,x),yt),x,y,10,1,10)
    res.append((g,sq/NG,cvm,aic,bic))
    print('%5d %10d %12s %14s %12s %12s'%(g,p_,c(sq/NG),c(cvm),c(aic,2),c(bic,2)))
res=np.array(res)
print()
print('  menor EQM validado: grau %d (%s)'%(int(res[np.argmin(res[:,2]),0]),c(res[:,2].min())))
print('  menor AIC:          grau %d'%int(res[np.argmin(res[:,3]),0]))
print('  menor BIC:          grau %d'%int(res[np.argmin(res[:,4]),0]))
print()
print('Testes F sequenciais (cada grau contra o anterior):')
print('%5s %12s %12s %12s'%('grau','SQR','F','p'))
sq_ant=None
for g in range(1,11):
    B=poli(x,g); sq=sqr(y,B@ajusta_base(B,y))
    if sq_ant is not None:
        F=(sq_ant-sq)/(sq/(NG-g-1)); pv=1-stats.f.cdf(F,1,NG-g-1)
        print('%5d %12s %12s %12s'%(g,c(sq,2),c(F,2),'%.4f'%pv))
    else:
        print('%5d %12s %12s %12s'%(g,c(sq,2),'—','—'))
    sq_ant=sq

print('\n'+'='*70)
print('ONDE O POLINÔMIO ERRA: o viés não se distribui por igual')
gsel=int(res[np.argmin(res[:,2]),0])
B=poli(xs,gsel,x); aj=B@ajusta_base(poli(x,gsel),y)
vd=f_par(xs)-f_par(x).mean()+y.mean()
print()
print('Com o grau escolhido (%d), comparando o ajuste à função VERDADEIRA f1:'%gsel)
print('%26s %12s %12s %12s'%('faixa de PAR','n','viés médio','|viés| médio'))
cortes=[(0,200),(200,500),(500,900),(900,1400),(1400,2100)]
for lo,hi in cortes:
    mk=(xs>=lo)&(xs<hi)
    print('%26s %12d %12s %12s'%('%d a %d'%(lo,hi),int(mk.sum()),c(float((aj[mk]-vd[mk]).mean())),
          c(float(np.abs(aj[mk]-vd[mk]).mean()))))
print()
print('A VARIÂNCIA do ajuste, por região (500 reamostras bootstrap):')
rg=np.random.default_rng(7); P_=np.zeros((500,len(xs)))
for r in range(500):
    i=rg.integers(0,NG,NG)
    P_[r]=poli(xs,gsel,x)@ajusta_base(poli(x[i],gsel,x),y[i])
va=P_.var(0,ddof=1)
print('%26s %12s %12s'%('faixa de PAR','variância','razão com o miolo'))
mio=(xs>=500)&(xs<1400); vm=float(va[mio].mean())
for lo,hi in cortes:
    mk=(xs>=lo)&(xs<hi)
    print('%26s %12s %12s'%('%d a %d'%(lo,hi),c(float(va[mk].mean())),c(float(va[mk].mean())/vm,2)))
print()
print('A variância nas BORDAS é %s vezes a do miolo. É o mesmo fenômeno da'%c(float(va[xs>=1400].mean())/vm,1))
print('ficha bias-variance, agora numa aplicação real: nas extremidades há')
print('dados de um lado só para ancorar a curva.')

print('\n'+'='*70)
print('EXTRAPOLAR COM POLINÔMIO: o que acontece fora do domínio')
xf=np.array([2100.,2200.,2400.,2600.,3000.])
print()
print('%10s %12s %12s %12s %12s %12s'%('PAR','grau 2','grau 4','grau 6','grau 8','VERDADE'))
for i,xv in enumerate(xf):
    linha=[]
    for g in [2,4,6,8]:
        bb=ajusta_base(poli(x,g),y)
        linha.append(float((poli(np.array([xv]),g,x)@bb)[0]))
    vdd=float((f_par(np.array([xv]))-f_par(x).mean()+y.mean())[0])
    print('%10s %12s %12s %12s %12s %12s'%(c(xv,0),*[c(v,2) for v in linha],c(vdd,2)))
print()
print('A %s umol — 5%% além do maior PAR observado — o grau 8 já prevê %s'%(c(2100.,0),
      c(float((poli(np.array([2100.]),8,x)@ajusta_base(poli(x,8),y))[0]),2)))
print('contra %s da verdade. A %s, prevê %s. Polinômio não extrapola.'%(
      c(float((f_par(np.array([2100.]))-f_par(x).mean()+y.mean())[0]),2),c(3000.,0),
      c(float((poli(np.array([3000.]),8,x)@ajusta_base(poli(x,8),y))[0]),2)))

print('\n'+'='*70)
print('A LIMITAÇÃO ESTRUTURAL: um polinômio é GLOBAL')
print()
print('A curva de luz tem curvaturas MUITO diferentes em regiões diferentes:')
print('sobe quase reta até 300, dobra entre 300 e 900, e vira para baixo depois.')
print('Um polinômio tem um único conjunto de coeficientes para todo o domínio,')
print('de modo que mexer na curvatura de uma região mexe em todas.')
print()
print("A demonstração: acrescentam-se 25 pontos SÓ na faixa de 1.800 a 2.000,")
print('DESLOCADOS em +4 umol (como se um lote de folhas tivesse sido medido com')
print('o equipamento descalibrado), e mede-se quanto o ajuste muda LONGE dali,')
print('em PAR de 150 a 1.000 — a mais de 800 umol da perturbação.')
print()
print('A comparação é feita com a MESMA flexibilidade nos dois lados: polinômio')
print('de grau g contra spline cúbica com o mesmo número de parâmetros.')
rg=np.random.default_rng(3)
xn=rg.uniform(1800,2000,25); yn=f_par(xn)-f_par(x).mean()+y.mean()+4.0+rg.normal(0,SIG,25)
x2=np.concatenate([x,xn]); y2=np.concatenate([y,yn])
lim=(min(x.min(),x2.min()),max(x.max(),x2.max()))
alvo=np.array([150.,300.,600.,1000.])
print()
print('%5s %8s %16s %16s %10s'%('p','grau','polinômio','spline cúbica','razão'))
print('%5s %8s %16s %16s %10s'%('','','deslocamento médio fora da região perturbada','',''))
guarda={}
for df in [5,8,11,14]:
    g=df-1; nos=np.percentile(x,np.linspace(0,100,df-2)[1:-1])
    dp=[];ds=[]
    for xv in alvo:
        a1=float((poli(np.array([xv]),g,x)@ajusta_base(poli(x,g),y))[0])
        a2=float((poli(np.array([xv]),g,x2)@ajusta_base(poli(x2,g,x2),y2))[0])
        dp.append(abs(a2-a1))
        Bv=base_bs(np.array([xv]),nos,3,lim)
        s1=float((Bv@ajusta_base(base_bs(x,nos,3,lim),y))[0])
        s2=float((Bv@ajusta_base(base_bs(x2,nos,3,lim),y2))[0])
        ds.append(abs(s2-s1))
    guarda[df]=(float(np.mean(dp)),float(np.mean(ds)),dp[0],ds[0])
    print('%5d %8d %16s %16s %10s'%(df,g,c(np.mean(dp),4),c(np.mean(ds),4),c(np.mean(dp)/np.mean(ds),1)))
print()
print('%5s %8s %16s %16s %10s'%('p','grau','polinômio','spline cúbica','razão'))
print('%5s %8s %16s %16s %10s'%('','','deslocamento em PAR = 150, o ponto MAIS DISTANTE','',''))
for df in [5,8,11,14]:
    _,_,a,b=guarda[df]
    print('%5d %8d %16s %16s %10s'%(df,df-1,c(a,4),c(b,4),c(a/max(b,1e-9),1)))
print()
print('O resultado é mais interessante que a formulação usual do manual, e')
print('exige dizê-lo com cuidado: a diferença entre base GLOBAL e base LOCAL')
print('DEPENDE DA FLEXIBILIDADE, e com pouca flexibilidade quase não existe.')
print()
print('Com p = 5 — o modelo que a validação cruzada escolheu — os dois se')
print('deslocam igualmente, e a spline até um pouco mais: a razão é %s.'%c(guarda[5][0]/guarda[5][1],1))
print('Com p = 8 a razão já é %s; com p = 11, %s; com p = 14, no ponto mais'%(
      c(guarda[8][0]/guarda[8][1],1),c(guarda[11][0]/guarda[11][1],1)))
print('distante, o polinômio se desloca %s e a spline %s — razão de %s para 1.'%(
      c(guarda[14][2],4),c(guarda[14][3],4),c(guarda[14][2]/guarda[14][3],0)))
print()
print('A leitura correta: um polinômio tem um único conjunto de coeficientes')
print('para TODO o domínio, de modo que um problema localizado contamina o')
print('ajuste inteiro — e quanto mais termos, mais ele contamina, porque os')
print('termos de ordem alta oscilam ao longo de todo o intervalo. A spline,')
print('definida por partes, absorve a perturbação onde ela está, e melhora')
print('à medida que ganha nós. É a razão técnica para preferir bases LOCAIS')
print('quando se precisa de flexibilidade, e o assunto da próxima ficha.')
