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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

y=NOTA
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: a nota. Piso irredutível sigma² = %s.'%c(EQM_PISO))

print('\n'+'='*70)
print('BAGGING À MÃO: B árvores, cada uma numa reamostra bootstrap')
print('Separamos 200 lotes de TESTE, que nenhuma árvore vê.')
rg=np.random.default_rng(7)
idx=rg.permutation(NC); te,tr=idx[:200],idx[200:]
Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]; n=len(tr)
B=500
P=np.empty((B,len(te))); OOB=np.zeros(n); NOOB=np.zeros(n)
for b in range(B):
    i=rg.integers(0,n,n)
    m=DecisionTreeRegressor(random_state=b).fit(Xtr[i],ytr[i])   # árvore PROFUNDA, sem poda
    P[b]=m.predict(Xte)
    fora=np.setdiff1d(np.arange(n),i)
    OOB[fora]+=m.predict(Xtr[fora]); NOOB[fora]+=1
print()
print('%8s %18s %18s'%('B','EQM no teste','EQM do bagging'))
print('%8s %18s %18s'%('','(uma árvore)','(média de B)'))
for b in [1,2,5,10,25,50,100,200,500]:
    print('%8d %18s %18s'%(b,c(eqm(yte,P[b-1])),c(eqm(yte,P[:b].mean(0)))))
print()
print('Uma árvore isolada, sem poda: EQM médio no teste = %s'%c(float(np.mean([eqm(yte,P[b]) for b in range(B)]))))
print('Bagging com 500 árvores:                          = %s'%c(eqm(yte,P.mean(0))))
pod=DecisionTreeRegressor(ccp_alpha=0.0443,random_state=0).fit(Xtr,ytr)
print('Uma árvore PODADA (ficha de árvores):             = %s'%c(eqm(yte,pod.predict(Xte))))
print('Função verdadeira:                                = %s'%c(eqm(yte,MU[te])))

print('\n'+'='*70)
print('POR QUE FUNCIONA: a decomposição da variância de uma média')
print('Se cada árvore tem variância s² e duas árvores quaisquer têm correlação rho,')
print('  Var(média de B árvores) = rho*s² + (1-rho)*s²/B')
print('Aqui rho é a correlação entre duas árvores ajustadas ao MESMO conjunto')
print('de treino, considerando a variação de um conjunto de treino para outro.')
print('Como o processo gerador é conhecido, sorteamos 30 conjuntos de treino')
print('novos (mesmos lotes, novo erro aleatório) e 50 árvores em cada um.')
R,Bb=30,50; Q=np.empty((R,Bb,len(te)))
for r in range(R):
    yr=MU[tr]+rg.normal(0,SIGN,n)
    for b in range(Bb):
        i=rg.integers(0,n,n); Q[r,b]=DecisionTreeRegressor(random_state=b).fit(Xtr[i],yr[i]).predict(Xte)
s2=float(np.mean(Q.reshape(R*Bb,-1).var(0,ddof=1)))          # variância de UMA árvore
cov=float(np.mean(Q.mean(1).var(0,ddof=1)-Q.var(1,ddof=1).mean(0)/Bb))  # covariância entre duas do mesmo treino
rho=cov/s2
print()
print('  variância de uma árvore, média nos pontos de teste   s² = %s'%c(s2))
print('  covariância entre duas árvores do mesmo treino          = %s'%c(cov))
print('  correlação média entre duas árvores                 rho = %s'%c(rho))
print()
print('%8s %22s %22s'%('B','variância prevista','rho*s² (o limite)'))
for b in [1,5,25,100,500]:
    print('%8d %22s %22s'%(b,c(rho*s2+(1-rho)*s2/b),c(rho*s2)))
print()
print('Aumentar B elimina a segunda parcela, mas NUNCA a primeira. As árvores')
print('são correlacionadas porque veem quase os mesmos dados e escolhem quase')
print('os mesmos cortes. É essa correlação que a random forest vai atacar.')

print('\n'+'='*70)
print('O ERRO FORA DA SACOLA (out-of-bag, OOB)')
print('Cada reamostra deixa de fora, em média, (1-1/n)^n ~ e^-1 = 36,8% dos lotes.')
ok=NOOB>0
print('  fração média de árvores em que um lote ficou fora: %s'%c(float(np.mean(NOOB/B))))
print('  EQM OOB (no treino, sem usar o teste)            = %s'%c(eqm(ytr[ok],OOB[ok]/NOOB[ok])))
print('  EQM no teste independente                        = %s'%c(eqm(yte,P.mean(0))))
print('O OOB é uma estimativa honesta de graça: dispensa a validação cruzada.')

print('\n'+'='*70)
print('BAGGING SÓ AJUDA PROCEDIMENTOS INSTÁVEIS')
print('Mesmo experimento, 10 dobras x 3 repetições, B = 200:')
print()
print('%-30s %14s %14s %12s'%('procedimento','sozinho','com bagging','ganho'))
for nm,f in [('árvore sem poda',lambda: DecisionTreeRegressor(random_state=0)),
             ('árvore podada',lambda: DecisionTreeRegressor(ccp_alpha=0.0443,random_state=0)),
             ('árvore de profundidade 2',lambda: DecisionTreeRegressor(max_depth=2,random_state=0)),
             ('regressão linear',LinearRegression)]:
    a,_=cv_reg(f,X,y,10,0,3)
    b,_=cv_reg(lambda f=f: BaggingRegressor(f(),n_estimators=200,random_state=0,n_jobs=1),X,y,10,0,3)
    print('%-30s %14s %14s %11s%%'%(nm,c(a),c(b),c(100*(a-b)/a,1)))
print()
print('A regressão linear é estável: reamostrar quase não muda a reta, e a')
print('média de retas quase iguais é a mesma reta. A árvore de profundidade 2')
print('tem pouco a ganhar: seu erro é VIÉS, e a média não remove viés.')

print('\n'+'='*70)
print('CLASSIFICAÇÃO: voto da maioria ou média das probabilidades')
yc=FINO
d=dobras(NC,10,0); voto=np.empty(NC); prob=np.empty(NC); um=np.empty(NC)
for j in range(10):
    trj,tej=d!=j,d==j; n2=trj.sum(); Xa,ya=X[trj],yc[trj]
    V=[];Q=[]
    for b in range(200):
        i=rg.integers(0,n2,n2); m=DecisionTreeClassifier(random_state=b).fit(Xa[i],ya[i])
        V.append(m.predict(X[tej])); Q.append(m.predict_proba(X[tej])[:,-1])
        if b==0: um[tej]=V[0]
    voto[tej]=(np.mean(V,0)>=.5); prob[tej]=np.mean(Q,0)
print()
print('  uma árvore sem poda: erro 10-fold      = %s'%c(acur(yc,um)*-1+1))
print('  bagging, voto da maioria (B = 200)     = %s'%c(1-acur(yc,voto)))
print('  bagging, média das probabilidades      = %s'%c(1-acur(yc,(prob>=.5).astype(int))))
print('  erro de Bayes                          = %s'%c(ERRO_BAYES))
q=np.clip(prob,1e-6,1-1e-6)
print('  log-loss da média das probabilidades   = %s'%c(float(-np.mean(yc*np.log(q)+(1-yc)*np.log(1-q)))))
print('  Brier                                  = %s'%c(float(np.mean((prob-yc)**2))))
