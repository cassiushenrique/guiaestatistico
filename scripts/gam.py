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

X=[PAR.copy(),TEMP.copy(),VPD.copy()]; NOM=['PAR','Temperatura','VPD']
y=A.copy()
print('Agora as TRÊS preditoras ao mesmo tempo. As quatro fichas anteriores')
print('trataram uma variável de cada vez; esta junta as três, mantendo a')
print('flexibilidade de cada uma e supondo que os efeitos se SOMAM.')
print()
print('  A = alfa + f1(PAR) + f2(temperatura) + f3(VPD) + erro')
print()
print('Essa é a hipótese ADITIVA, e é o que se ganha e o que se paga: ganha-se')
print('a possibilidade de estimar cada função separadamente, com n razoável;')
print('paga-se a exclusão das interações.')

print('\n'+'='*70)
print('O PONTO DE PARTIDA: quanto o modelo linear deixa na mesa')
Xd=np.column_stack([np.ones(NG)]+X); b=np.linalg.lstsq(Xd,y,rcond=None)[0]
print()
print('%16s %14s %14s'%('modelo','R² resub','EQM 10-fold'))
def cvm_multi(aj,k=10,seed=1,reps=6):
    tot=[]
    for r in range(reps):
        d=dobras(NG,k,seed+77*r); e=np.empty(NG)
        for j in range(k):
            tr,te=d!=j,d==j
            e[te]=y[te]-aj(tr,te)
        tot.append(float(np.mean(e**2)))
    return float(np.mean(tot))
def aj_lin(tr,te):
    Z=np.column_stack([np.ones(int(tr.sum()))]+[v[tr] for v in X])
    bb=np.linalg.lstsq(Z,y[tr],rcond=None)[0]
    Zt=np.column_stack([np.ones(int(te.sum()))]+[v[te] for v in X])
    return Zt@bb
print('%16s %14s %14s'%('linear',c(r2(y,Xd@b)),c(cvm_multi(aj_lin))))
print('%16s %14s %14s'%('verdadeiro',c(r2(y,verdade(PAR,TEMP,VPD))),c(eqm(y,verdade(PAR,TEMP,VPD)))))
print()
print('  variância de A = %s | sigma² = %s'%(c(float(y.var(ddof=1))),c(SIG**2)))

print('\n'+'='*70)
print('O RETROAJUSTE (backfitting) de Hastie e Tibshirani')
print('O algoritmo é simples: começa com todas as parcelas em zero e, em')
print('rodadas, atualiza cada uma suavizando os RESÍDUOS PARCIAIS — o que')
print('sobra de y depois de tirar o intercepto e as outras parcelas.')
print()
lam=[2e-4,2e-4,2e-4]
suav=[(lambda xx,rr,l=lam[j]: ss_prev(xx,rr,l,xx)) for j in range(3)]
# convergencia, rodada a rodada
f=[np.zeros(NG) for _ in range(3)]; a0=y.mean()
print('%8s %18s %16s'%('rodada','maior mudança','SQR'))
hist=[]
for it in range(1,41):
    md=0.0
    for j in range(3):
        r=y-a0-sum(f[k] for k in range(3) if k!=j)
        nv=suav[j](X[j],r); nv=nv-nv.mean()
        md=max(md,float(np.max(np.abs(nv-f[j])))); f[j]=nv
    sq=sqr(y,a0+sum(f)); hist.append((it,md,sq))
    if it in (1,2,3,5,8,12,20,30,40):
        print('%8d %18s %16s'%(it,'%.3e'%md,c(sq,4)))
print()
print()
print('A convergência tem dois ritmos, e vale distinguir. A SOMA DE QUADRADOS')
print('estabiliza depressa: da rodada 3 em diante ela muda na terceira casa.')
print('As PARCELAS continuam se ajustando por mais tempo, porque a parte que')
print('as preditoras compartilham é redistribuída entre elas lentamente —')
print('a mudança máxima cai de %.1e na primeira rodada para %.1e na 40ª.'%(hist[0][1],hist[-1][1]))
print('É a concurvidade em ação, e o assunto reaparece adiante nesta ficha.')
print()
print('Cada parcela é CENTRADA (média zero) a cada')
print('passo: sem isso o intercepto e as parcelas não seriam identificáveis,')
print('porque somar uma constante a f1 e tirá-la de f2 não mudaria nada.')
print()
for j in range(3):
    print('  média de f%d após convergir = %s'%(j+1,c(float(f[j].mean()),10)))
print('  intercepto = %s | média de A = %s'%(c(a0),c(float(y.mean()))))

print('\n'+'='*70)
print('A ESCOLHA DA SUAVIDADE DE CADA PARCELA')
print('Cada f tem seu próprio lambda, e eles são escolhidos juntos.')
print()
GRADE=[1e-5,5e-5,2e-4,1e-3,5e-3]
melhor=None
print('%10s %10s %10s %14s %22s'%('lam1','lam2','lam3','EQM 10-fold','g.l. ef. (f1, f2, f3)'))
for l1 in GRADE:
    for l2 in GRADE:
        for l3 in GRADE:
            def ajg(tr,te,l1=l1,l2=l2,l3=l3):
                sv=[(lambda xx,rr,l=l: ss_prev(xx,rr,l,xx)) for l in (l1,l2,l3)]
                a,ff=gam_backfit([v[tr] for v in X],y[tr],sv,it=12)
                return a+sum(ss_prev(X[j][tr],ff[j],[l1,l2,l3][j],X[j][te]) for j in range(3))
            m=cvm_multi(ajg,reps=2)
            if melhor is None or m<melhor[0]: melhor=(m,l1,l2,l3)
m,l1,l2,l3=melhor
for lt in [(1e-5,1e-5,1e-5),(2e-4,2e-4,2e-4),(l1,l2,l3),(5e-3,5e-3,5e-3)]:
    def ajg(tr,te,lt=lt):
        sv=[(lambda xx,rr,l=l: ss_prev(xx,rr,l,xx)) for l in lt]
        a,ff=gam_backfit([v[tr] for v in X],y[tr],sv,it=12)
        return a+sum(ss_prev(X[j][tr],ff[j],lt[j],X[j][te]) for j in range(3))
    mm=cvm_multi(ajg,reps=4)
    gls=[ss_edf(X[j],lt[j]) for j in range(3)]
    mk=' <- melhor' if lt==(l1,l2,l3) else ''
    print('%10s %10s %10s %14s %22s%s'%('%.0e'%lt[0],'%.0e'%lt[1],'%.0e'%lt[2],c(mm),
          ', '.join(c(g,2) for g in gls),mk))
print()
print('  melhor combinação: lambda = (%.0e, %.0e, %.0e)'%(l1,l2,l3))
print('  g.l. efetivos: f1 = %s, f2 = %s, f3 = %s'%(c(ss_edf(X[0],l1),2),c(ss_edf(X[1],l2),2),c(ss_edf(X[2],l3),2)))
print('  EQM 10-fold = %s, contra %s do modelo linear'%(c(m),c(cvm_multi(aj_lin))))
print('  redução de %s%% no erro de predição'%c(100*(1-m/cvm_multi(aj_lin)),1))

print('\n'+'='*70)
print('AS TRÊS PARCELAS ESTIMADAS, contra as VERDADEIRAS')
sv=[(lambda xx,rr,l=l: ss_prev(xx,rr,l,xx)) for l in (l1,l2,l3)]
a0,F=gam_backfit(X,y,sv,it=60)
fv=[f_par(PAR),f_temp(TEMP),f_vpd(VPD)]
fv=[v-v.mean() for v in fv]
print()
print('%16s %12s %14s %14s %14s'%('parcela','g.l. ef.','amplitude est.','amplitude verd.','|viés| médio'))
for j in range(3):
    print('%16s %12s %14s %14s %14s'%(NOM[j],c([ss_edf(X[j],l) for l in (l1,l2,l3)][j],2),
          c(float(np.ptp(F[j])),2),c(float(np.ptp(fv[j])),2),c(float(np.abs(F[j]-fv[j]).mean()))))
print()
print('  R² de resubstituição do GAM = %s'%c(r2(y,a0+sum(F))))
print('  R² do modelo verdadeiro     = %s'%c(r2(y,verdade(PAR,TEMP,VPD))))
print()
print('As três parcelas são recuperadas com viés pequeno. A do VPD é a mais')
print('difícil: a amplitude verdadeira é %s e os dados se concentram abaixo'%c(float(np.ptp(fv[2])),2))
print('de 1 kPa, de modo que a parte alta da curva é estimada com poucos pontos.')
print()
print('%16s %10s %12s'%('VPD','n','% do total'))
for lo,hi in [(0,0.5),(0.5,0.8),(0.8,1.2),(1.2,1.8),(1.8,3.0)]:
    mk=(VPD>=lo)&(VPD<hi)
    print('%16s %10d %12s'%('%.1f a %.1f'%(lo,hi),int(mk.sum()),c(100*mk.mean(),1)+'%'))

print('\n'+'='*70)
print('CONCURVIDADE: a colinearidade dos modelos aditivos')
print('Duas preditoras colineares tornam os coeficientes indeterminados. Duas')
print('preditoras ligadas por uma RELAÇÃO SUAVE qualquer tornam as PARCELAS')
print('indeterminadas — e isso não aparece no VIF, que só vê relações lineares.')
print()
print('%28s %14s %14s'%('par de preditoras','r de Pearson','R² de um suavizador'))
for i in range(3):
    for j in range(3):
        if i<j:
            aj_=ss_prev(X[i],X[j],2e-4,X[i])
            print('%28s %14s %14s'%('%s e %s'%(NOM[i],NOM[j]),c(float(np.corrcoef(X[i],X[j])[0,1])),
                  c(r2(X[j],aj_))))
print()
print('A temperatura e o VPD estão ligados por uma relação não linear forte: o')
print('r de Pearson é %s, mas um suavizador explica %s da variação do VPD.'%(
      c(float(np.corrcoef(TEMP,VPD)[0,1])),c(r2(VPD,ss_prev(TEMP,VPD,2e-4,TEMP)))))
print('É a concurvidade, e a consequência é que parte do efeito pode ser')
print('atribuída a qualquer uma das duas.')
print()
print('A demonstração: 300 reamostras bootstrap, GAM reajustado em cada uma.')
rg=np.random.default_rng(23)
PT=np.zeros((300,3))
alvos=[np.array([1500.]),np.array([34.]),np.array([1.8])]
for r in range(300):
    i=rg.integers(0,NG,NG)
    sv2=[(lambda xx,rr,l=l: ss_prev(xx,rr,l,xx)) for l in (l1,l2,l3)]
    a_,F_=gam_backfit([v[i] for v in X],y[i],sv2,it=20)
    for j in range(3):
        PT[r,j]=float(ss_prev(X[j][i],F_[j],[l1,l2,l3][j],alvos[j])[0])
print()
print('%28s %14s %14s %14s'%('parcela em','estimativa','EP bootstrap','|EP/amplitude|'))
for j,alv in enumerate(alvos):
    est=float(ss_prev(X[j],F[j],[l1,l2,l3][j],alv)[0])
    print('%28s %14s %14s %14s'%('%s = %s'%(NOM[j],c(float(alv[0]),1)),c(est),c(float(PT[:,j].std(ddof=1))),
          c(float(PT[:,j].std(ddof=1))/float(np.ptp(F[j]))))) 
print()
print('A leitura honesta: os três erros-padrão são PEQUENOS em relação às')
print('amplitudes — de 2% a 6%. A concurvidade não destruiu a estimação aqui,')
print('e dizer o contrário seria exagero.')
print()
print('O que a tabela mostra é uma ORDEM: a parcela do VPD tem incerteza')
print('relativa três vezes maior que a do PAR (6,1% contra 2,1%), e é')
print('exatamente ela que acumula os dois problemas — concurvidade com a')
print('temperatura (um suavizador explica 45% da variação do VPD a partir')
print('dela) e escassez de dados na parte alta da faixa (1,5% das observações')
print('acima de 1,8 kPa). Com concurvidade mais forte, ou com n menor, essa')
print('ordem viraria um problema de fato.')

print('\n'+'='*70)
print('O QUE A HIPÓTESE ADITIVA EXCLUI: interações')
print('Na fisiologia real, o efeito da temperatura DEPENDE da radiação: no')
print('escuro não há fotossíntese para a temperatura modular. Um GAM aditivo')
print('não pode representar isso.')
print()
print('A verificação: simulam-se dados COM interação de força crescente e')
print('mede-se o que o GAM aditivo deixa escapar.')
print()
print('  %5s %14s %16s %18s %14s'%('força','amplitude','R² do GAM','r(resíduo, PARxT)','ganho de R²'))
print('  %5s %14s %16s %18s %14s'%('','da interação','aditivo','','ao incluir PARxT'))
rg=np.random.default_rng(31)
for forca in [0.0,1.0,2.5,5.0,10.0]:
    fint=forca*0.0016*(PAR-PAR.mean())*(TEMP-TEMP.mean())/10.0
    yy=verdade(PAR,TEMP,VPD)+fint-fint.mean()+rg.normal(0,SIG,NG)
    sv3=[(lambda xx,rr,l=l: ss_prev(xx,rr,l,xx)) for l in (l1,l2,l3)]
    a_,F_=gam_backfit(X,yy,sv3,it=40)
    res=yy-a_-sum(F_)
    # o resíduo ainda guarda estrutura? correlaciona com o produto centrado
    pr=(PAR-PAR.mean())*(TEMP-TEMP.mean())
    amp=float(np.ptp(fint)); rr_=float(np.corrcoef(res,pr)[0,1])
    aj_i=np.column_stack([np.ones(NG),a_+sum(F_),pr]); bi=np.linalg.lstsq(aj_i,yy,rcond=None)[0]
    ganho=r2(yy,aj_i@bi)-r2(yy,a_+sum(F_))
    print('  %5s %14s %16s %18s %14s'%(c(forca,1),c(amp,2),c(r2(yy,a_+sum(F_))),c(rr_),c(ganho)))
print()
print()
print('A leitura é quantitativa, e isso importa. Com interação de amplitude 3,3')
print('umol — um sexto da amplitude da parcela do PAR, que é 21,5 — o GAM')
print('aditivo perde pouco: o R² cai de 0,9653 para 0,9483, e incluir o termo')
print('de produto recupera apenas 0,0018 de R². É só quando a interação chega a')
print('uma amplitude comparável à das parcelas principais que ela se torna')
print('visível no resíduo e o ganho de incluí-la fica relevante.')
print()
print('O diagnóstico prático é exatamente esse: examinar o resíduo contra os')
print('produtos das preditoras, e medir o GANHO de R² ao acrescentar o termo —')
print('não apenas constatar que a correlação não é zero.')

print('\n'+'='*70)
print('A SÍNTESE: o que cada ficha do grupo acrescentou')
print()
print('%44s %16s'%('modelo','EQM 10-fold'))
print('%44s %16s'%('média de A (modelo nulo)',c(float(np.mean((y-y.mean())**2)))))
print('%44s %16s'%('linear nas três preditoras',c(cvm_multi(aj_lin))))
def aj_p1(tr,te):
    B=poli(PAR[tr],5,PAR); bb=ajusta_base(B,y[tr]); return poli(PAR[te],5,PAR)@bb
print('%44s %16s'%('polinômio grau 5 só em PAR',c(cvm_multi(aj_p1))))
def aj_lin_poli(tr,te):
    Z=np.column_stack([poli(PAR[tr],5,PAR),TEMP[tr,None],VPD[tr,None]])
    bb=np.linalg.lstsq(Z,y[tr],rcond=None)[0]
    Zt=np.column_stack([poli(PAR[te],5,PAR),TEMP[te,None],VPD[te,None]])
    return Zt@bb
print('%44s %16s'%('polinômio em PAR + linear nas outras',c(cvm_multi(aj_lin_poli))))
def ajg_fin(tr,te):
    sv4=[(lambda xx,rr,l=l: ss_prev(xx,rr,l,xx)) for l in (l1,l2,l3)]
    a,ff=gam_backfit([v[tr] for v in X],y[tr],sv4,it=20)
    return a+sum(ss_prev(X[j][tr],ff[j],[l1,l2,l3][j],X[j][te]) for j in range(3))
print('%44s %16s'%('GAM com as três parcelas suaves',c(cvm_multi(ajg_fin))))
print('%44s %16s'%('modelo VERDADEIRO (o piso)',c(eqm(y,verdade(PAR,TEMP,VPD)))))
