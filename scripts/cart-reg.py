import numpy as np
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (cab16)
# ------------------------------------------------------------------
import numpy as np
from scipy.stats import norm
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (dados16)
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

# Dados fictícios: 600 lotes de café (arquivo cafe.csv, pasta scripts/ do site)
cafe = le_csv('cafe.csv')
ALT, TEMP, CHUVA = (cafe[k].to_numpy(float) for k in ('alt', 'temp', 'chuva'))
SOMB, NFOL, PH = (cafe[k].to_numpy(float) for k in ('somb', 'nfol', 'ph'))
NOTA = cafe['nota'].to_numpy(float); FINO = cafe['fino'].to_numpy(int)
NOMES=['alt','temp','chuva','somb','nfol','ph']
ROT={'alt':'Altitude','temp':'Temperatura','chuva':'Chuva','somb':'Sombreamento','nfol':'N foliar','ph':'pH'}
X=np.column_stack([ALT,TEMP,CHUVA,SOMB,NFOL,PH])
NC,PC=X.shape; SIGN=1.55
def f_nota(alt,temp,chuva,somb,nfol,ph):
    deg=4.6*(np.asarray(alt)>=1050); rampa=0.0028*np.clip(np.asarray(alt)-700,0,None)
    inter=np.where(np.asarray(temp)>20.5,0.052*np.asarray(somb),-0.030*np.asarray(somb))
    agua=-2.6e-6*(np.asarray(chuva)-1450)**2
    nitro=2.9*np.log(np.asarray(nfol)/1.6); acid=-1.9*(np.asarray(ph)-5.4)**2
    return 74.0+deg+rampa+inter+agua+nitro+acid
def verdade(A): return f_nota(A[:,0],A[:,1],A[:,2],A[:,3],A[:,4],A[:,5])
# ------------------------------------------------------------------ fim do bloco comum

def c(v, d=4): return ('%.*f' % (d, v)).replace('.', ',')
def eqm(o, p): return float(np.mean((np.asarray(o) - np.asarray(p))**2))
def r2(o, p):  return 1 - float(np.sum((np.asarray(o)-np.asarray(p))**2))/float(np.sum((np.asarray(o)-np.mean(o))**2))
def acur(o, p): return float(np.mean(np.asarray(o) == np.asarray(p)))

MU = verdade(X)                                   # média verdadeira da nota
PVER = 1 - norm.cdf((80 - MU)/SIGN)               # P(fino | x) verdadeira
ERRO_BAYES = float(np.mean(np.minimum(PVER, 1 - PVER)))
EQM_PISO = SIGN**2

def dobras(n, k, seed):
    rg = np.random.default_rng(seed)
    d = np.tile(np.arange(k), int(np.ceil(n/k)))[:n]; rg.shuffle(d); return d

def cv_reg(fabrica, A, y, k=10, seed=0, reps=1):
    """EQM por validação cruzada. fabrica() devolve um estimador sklearn novo."""
    tot = []
    for r in range(reps):
        d = dobras(len(y), k, seed + 101*r); e = np.empty(len(y))
        for j in range(k):
            tr, te = d != j, d == j
            m = fabrica().fit(A[tr], y[tr]); e[te] = y[te] - m.predict(A[te])
        tot.append(float(np.mean(e**2)))
    return float(np.mean(tot)), (float(np.std(tot, ddof=1)) if reps > 1 else 0.0)

def cv_clf(fabrica, A, y, k=10, seed=0, reps=1, prob=False):
    """Erro de classificação (e, se prob, log-loss e Brier) por validação cruzada."""
    err = []; ll = []; br = []
    for r in range(reps):
        d = dobras(len(y), k, seed + 101*r); pred = np.empty(len(y)); pp = np.empty(len(y))
        for j in range(k):
            tr, te = d != j, d == j
            m = fabrica().fit(A[tr], y[tr]); pred[te] = m.predict(A[te])
            if prob: pp[te] = m.predict_proba(A[te])[:, 1]
        err.append(float(np.mean(pred != y)))
        if prob:
            q = np.clip(pp, 1e-6, 1 - 1e-6)
            ll.append(float(-np.mean(y*np.log(q) + (1 - y)*np.log(1 - q))))
            br.append(float(np.mean((pp - y)**2)))
    if prob: return float(np.mean(err)), float(np.mean(ll)), float(np.mean(br))
    return float(np.mean(err)), (float(np.std(err, ddof=1)) if reps > 1 else 0.0)

def padroniza(A, ref=None):
    ref = A if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0, ddof=1); return (A - mu)/sd, mu, sd

DADOS_TXT = ('600 lotes de café, nota sensorial (escala SCA, de 70 a 86) e seis variáveis de lavoura: '
             'altitude, temperatura média, chuva anual, sombreamento, N foliar e pH. Lote "fino" = nota >= 80.')
# ------------------------------------------------------------------ fim do bloco comum
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

y=NOTA
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Dados fictícios, com processo gerador CONHECIDO. A nota tem:')
print('  - um DEGRAU de +4,6 pontos acima de 1.050 m de altitude;')
print('  - uma INTERAÇÃO: o sombreamento AJUDA acima de 20,5 C e ATRAPALHA abaixo;')
print('  - partes suaves em chuva, N foliar e pH; erro de desvio-padrão 1,55.')
print()
print('  variância da nota = %s | sigma² = %s (o piso)'%(c(float(y.var(ddof=1))),c(EQM_PISO)))
lin,_=cv_reg(LinearRegression,X,y,10,0,5)
spl,_=cv_reg(lambda: make_pipeline(SplineTransformer(n_knots=6,degree=3),Ridge(alpha=1e-3)),X,y,10,0,5)
print('  EQM validado do modelo LINEAR             = %s'%c(lin))
print('  EQM validado do modelo ADITIVO com splines = %s'%c(spl))
print('  EQM da função VERDADEIRA                   = %s'%c(eqm(y,MU)))
print()
print('O modelo aditivo suave não representa nem o degrau nem a interação.')

print('\n'+'='*70)
print('A PRIMEIRA DIVISÃO, procurada exaustivamente')
print('Para cada variável e cada corte possível, calcula-se quanto a soma de')
print('quadrados cai ao separar os lotes em dois grupos e prever cada um pela')
print('sua média.')
SQT=float(np.sum((y-y.mean())**2))
print()
print('%-14s %12s %16s %14s'%('variável','melhor corte','queda da SQR','% da SQT'))
melh=[]
for j,nm in enumerate(NOMES):
    xs=np.sort(np.unique(X[:,j])); cortes=(xs[:-1]+xs[1:])/2
    best=(0,None)
    for t in cortes:
        e=y[X[:,j]<=t]; d=y[X[:,j]>t]
        if len(e)<5 or len(d)<5: continue
        q=SQT-(np.sum((e-e.mean())**2)+np.sum((d-d.mean())**2))
        if q>best[0]: best=(q,t)
    melh.append((best[0],nm,best[1]))
    print('%-14s %12s %16s %14s'%(ROT[nm],c(best[1],1),c(best[0],2),c(100*best[0]/SQT,2)+'%'))
melh.sort(reverse=True)
q,nm,t=melh[0]
e=y[X[:,NOMES.index(nm)]<=t]; d=y[X[:,NOMES.index(nm)]>t]
print()
print('A melhor divisão é %s <= %s, que separa %d e %d lotes, com médias'%(ROT[nm],c(t,1),len(e),len(d)))
print('%s e %s. O degrau verdadeiro está em 1.050 m.'%(c(e.mean(),3),c(d.mean(),3)))

print('\n'+'='*70)
print('CRESCER E PODAR: a profundidade e o custo-complexidade')
print()
print('%11s %8s %14s %16s'%('profundidade','folhas','EQM treino','EQM 10-fold'))
for dp in [1,2,3,4,5,6,8,10,15,None]:
    m=DecisionTreeRegressor(max_depth=dp,random_state=0).fit(X,y)
    cv,_=cv_reg(lambda dp=dp: DecisionTreeRegressor(max_depth=dp,random_state=0),X,y,10,0,5)
    print('%11s %8d %14s %16s'%(str(dp) if dp else 'sem limite',m.get_n_leaves(),c(eqm(y,m.predict(X))),c(cv)))
print()
print('Poda por custo-complexidade: a árvore máxima é podada minimizando')
print('  SQR(T) + alpha * |T|,   |T| = número de folhas')
cam=DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(X,y)
alphas=np.unique(np.concatenate([[0],np.quantile(cam.ccp_alphas[cam.ccp_alphas>0],np.linspace(0,1,40))]))
res=[]
d=dobras(NC,10,0)
for a in alphas:
    erros=[]; e=np.empty(NC)
    for j in range(10):
        tr,te=d!=j,d==j
        m=DecisionTreeRegressor(ccp_alpha=a,random_state=0).fit(X[tr],y[tr])
        erros.append(eqm(y[te],m.predict(X[te]))); e[te]=y[te]-m.predict(X[te])
    nf=DecisionTreeRegressor(ccp_alpha=a,random_state=0).fit(X,y).get_n_leaves()
    res.append((a,float(np.mean(e**2)),float(np.std(erros,ddof=1)/np.sqrt(10)),nf))
res=np.array(res); jm=int(np.argmin(res[:,1]))
lim=res[jm,1]+res[jm,2]
j1=int(max(i for i in range(len(res)) if res[i,1]<=lim))
print()
print('%12s %8s %14s'%('alpha','folhas','EQM 10-fold'))
for i in sorted(set([0,5,10,15,20,25,30,jm,j1,len(res)-1])):
    mk=' <- menor EQM' if i==jm else (' <- regra do EP único' if i==j1 else '')
    print('%12s %8d %14s%s'%(c(res[i,0],4),int(res[i,3]),c(res[i,1]),mk))
amin,a1=res[jm,0],res[j1,0]
print()
print('  menor EQM: alpha = %s, %d folhas, EQM %s'%(c(amin,4),int(res[jm,3]),c(res[jm,1])))
print('  EP único : alpha = %s, %d folhas, EQM %s'%(c(a1,4),int(res[j1,3]),c(res[j1,1])))

print('\n'+'='*70)
print('A ÁRVORE PODADA (regra do EP único), por extenso')
arv=DecisionTreeRegressor(ccp_alpha=a1,random_state=0).fit(X,y)
print()
print(export_text(arv,feature_names=[ROT[k] for k in NOMES],decimals=1))
print('A árvore ENCONTROU o degrau: o primeiro corte é em altitude e o valor')
print('está perto de 1.050 m. A interação aparece como cortes em sombreamento')
print('DENTRO de ramos definidos por temperatura — que é exatamente a forma de')
print('uma árvore representar "o efeito de uma variável depende de outra".')

print('\n'+'='*70)
print('A PREDIÇÃO É UMA ESCADA')
print('Uma árvore prevê a MÉDIA DA FOLHA: a função ajustada é constante por')
print('pedaços. Variando só a altitude, com as demais no valor mediano:')
med=np.median(X,axis=0)
print()
print('%12s %16s %16s'%('altitude','árvore podada','verdade'))
for a in [650,800,950,1040,1060,1150,1300,1450]:
    xx=med.copy(); xx[0]=a
    print('%12s %16s %16s'%(c(a,0),c(float(arv.predict(xx[None,:])[0]),3),c(float(verdade(xx[None,:])[0]),3)))
print()
print('Acima de 1.050 a verdade é uma rampa suave mais um degrau; a árvore')
print('acerta o degrau e aproxima a rampa por degraus menores.')

print('\n'+'='*70)
print('INSTABILIDADE: a mesma análise em 500 reamostras bootstrap')
rg=np.random.default_rng(3)
pri=[];cortes=[];nfolhas=[]
gx=np.tile(med,(1,1))
for r in range(500):
    i=rg.integers(0,NC,NC)
    m=DecisionTreeRegressor(ccp_alpha=a1,random_state=0).fit(X[i],y[i])
    t=m.tree_; pri.append(NOMES[t.feature[0]]); cortes.append(t.threshold[0]); nfolhas.append(m.get_n_leaves())
pri=np.array(pri); cortes=np.array(cortes)
print()
for nm in NOMES:
    q=np.mean(pri==nm)
    if q>0: print('  primeiro corte em %-12s: %5.1f%% das reamostras'%(ROT[nm],100*q))
ca=cortes[pri=='alt']
print('  quando é altitude, o corte vai de %s a %s (mediana %s)'%(c(ca.min(),1),c(ca.max(),1),c(np.median(ca),1)))
print('  número de folhas: de %d a %d (mediana %d)'%(min(nfolhas),max(nfolhas),int(np.median(nfolhas))))
print()
print('O PRIMEIRO corte é estável, porque o degrau é forte. Mas o número de')
print('folhas varia muito, e os cortes de baixo mudam de reamostra para')
print('reamostra: é essa instabilidade que as fichas bagging e random forest')
print('vão explorar.')

print('\n'+'='*70)
print('A SÍNTESE: árvore contra os modelos suaves')
cv_arv,_=cv_reg(lambda: DecisionTreeRegressor(ccp_alpha=a1,random_state=0),X,y,10,0,5)
cv_amin,_=cv_reg(lambda: DecisionTreeRegressor(ccp_alpha=amin,random_state=0),X,y,10,0,5)
cv_max,_=cv_reg(lambda: DecisionTreeRegressor(random_state=0),X,y,10,0,5)
print()
print('%-38s %14s'%('modelo','EQM 10-fold'))
for nm,v in [('média (modelo nulo)',float(y.var(ddof=1))),('linear nas seis',lin),('aditivo com splines',spl),
             ('árvore sem poda',cv_max),('árvore podada, menor EQM',cv_amin),('árvore podada, EP único',cv_arv),
             ('função VERDADEIRA (piso)',eqm(y,MU))]:
    print('%-38s %14s'%(nm,c(v)))
print()
print('A árvore podada bate o linear e o aditivo suave porque o processo tem')
print('degrau e interação, que é o que árvores representam bem. Mas fica longe')
print('do piso: aproximar as partes SUAVES por degraus custa caro.')
