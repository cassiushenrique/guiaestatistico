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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance

y=NOTA
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: a nota. Piso irredutível sigma² = %s. p = 6 preditoras.'%c(EQM_PISO))

print('\n'+'='*70)
print('O ÚNICO PARÂMETRO QUE IMPORTA: mtry, quantas variáveis sorteadas por corte')
print('mtry = 6 é o bagging; mtry = 1 escolhe o corte numa variável ao acaso.')
rg=np.random.default_rng(11)
idx=rg.permutation(NC); te,tr=idx[:200],idx[200:]; n=len(tr)
Xtr,Xte=X[tr],X[te]
R,Bb=20,40
print('Para cada mtry: variância de uma árvore (s²), correlação entre árvores')
print('(rho), ambos sobre 20 conjuntos de treino novos, e o EQM OOB da floresta.')
print()
print('%6s %12s %10s %14s %14s'%('mtry','s² (1 árv.)','rho','rho*s²','EQM OOB'))
for mt in [1,2,3,4,6]:
    Q=np.empty((R,Bb,len(te)))
    for r in range(R):
        yr=MU[tr]+rg.normal(0,SIGN,n)
        f=RandomForestRegressor(n_estimators=Bb,max_features=mt,random_state=r).fit(Xtr,yr)
        Q[r]=np.array([t.predict(Xte) for t in f.estimators_])
    s2=float(np.mean(Q.reshape(R*Bb,-1).var(0,ddof=1)))
    cov=float(np.mean(Q.mean(1).var(0,ddof=1)-Q.var(1,ddof=1).mean(0)/Bb))
    oob=RandomForestRegressor(n_estimators=500,max_features=mt,oob_score=True,random_state=0).fit(X,y)
    print('%6d %12s %10s %14s %14s'%(mt,c(s2),c(cov/s2),c(cov),c(eqm(y,oob.oob_prediction_))))
print()
print('Diminuir mtry torna cada árvore PIOR (s² sobe) mas MENOS PARECIDA com')
print('as outras (rho cai). O erro da floresta é o equilíbrio entre os dois.')

print('\n'+'='*70)
print('A FLORESTA ESCOLHIDA, contra as alternativas (10 dobras x 3 repetições)')
from sklearn.linear_model import LinearRegression
print()
print('%-34s %14s'%('modelo','EQM 10-fold'))
for nm,f in [('regressão linear',LinearRegression),
             ('uma árvore podada',lambda: DecisionTreeRegressor(ccp_alpha=0.0443,random_state=0)),
             ('bagging (mtry = 6), 300 árvores',lambda: RandomForestRegressor(300,max_features=6,random_state=0)),
             ('random forest mtry = 2, 300 árvores',lambda: RandomForestRegressor(300,max_features=2,random_state=0)),
             ('random forest mtry = 3, 300 árvores',lambda: RandomForestRegressor(300,max_features=3,random_state=0)),
             ('rf mtry = 3, folha mínima 5',lambda: RandomForestRegressor(300,max_features=3,min_samples_leaf=5,random_state=0))]:
    v,_=cv_reg(f,X,y,10,0,3); print('%-34s %14s'%(nm,c(v)))
print('%-34s %14s'%('função verdadeira (piso)',c(eqm(y,MU))))

print('\n'+'='*70)
print('QUANTAS ÁRVORES? O EQM OOB em função de B (mtry = 3)')
f=RandomForestRegressor(n_estimators=10,max_features=3,random_state=0,oob_score=True,warm_start=True)
curva={}
for b in (25,50,100,200,500,1000):
    f.set_params(n_estimators=b); f.fit(X,y)        # warm_start: acrescenta árvores
    curva[b]=eqm(y,f.oob_prediction_)
print()
for b,v in curva.items(): print('  B = %5d   EQM OOB = %s'%(b,c(v)))
print('Mais árvores nunca causam sobreajuste: o erro só estabiliza.')

print('\n'+'='*70)
print('IMPORTÂNCIA DAS VARIÁVEIS: duas medidas, e a verdade')
print('Verdade: quanto o EQM da função verdadeira piora se a variável for')
print('embaralhada (permutação aplicada à própria função geradora).')
rf=RandomForestRegressor(500,max_features=3,random_state=0,oob_score=True).fit(X,y)
imp_imp=rf.feature_importances_
pi=permutation_importance(rf,X,y,n_repeats=20,random_state=0)
idx=rg.permutation(NC); te,tr=idx[:200],idx[200:]
rf2=RandomForestRegressor(500,max_features=3,random_state=0).fit(X[tr],y[tr])
pit=permutation_importance(rf2,X[te],y[te],n_repeats=30,random_state=0)
verd=[]
for j in range(6):
    v=[]
    for r in range(30):
        Z=X.copy(); Z[:,j]=rg.permutation(Z[:,j]); v.append(np.mean((verdade(Z)-MU)**2))
    verd.append(np.mean(v))
verd=np.array(verd)
print()
print('%-14s %14s %16s %16s %16s'%('variável','impureza (%)','perm. no treino','perm. no teste','VERDADE'))
for j in range(6):
    print('%-14s %14s %16s %16s %16s'%(ROT[NOMES[j]],c(100*imp_imp[j],1),c(pi.importances_mean[j],3),c(pit.importances_mean[j],3),c(verd[j],3)))
print()
print('A altitude e a temperatura têm correlação %s. Uma floresta pode usar'%c(np.corrcoef(X[:,0],X[:,1])[0,1],3))
print('uma no lugar da outra, e a importância se divide entre elas.')

print('\n'+'='*70)
print('UMA VARIÁVEL DE RUÍDO, para calibrar o olho')
Xr=np.column_stack([X,rg.normal(size=NC),rg.integers(0,500,NC).astype(float)])
rf3=RandomForestRegressor(500,max_features=3,random_state=0).fit(Xr[tr],y[tr])
a=rf3.feature_importances_; p=permutation_importance(rf3,Xr[te],y[te],n_repeats=30,random_state=0).importances_mean
print('Acrescentamos duas colunas sem relação com a nota: ruído normal e um')
print('código inteiro de 0 a 499 (muitos valores distintos, como um ID).')
print()
print('%-22s %14s %16s'%('variável','impureza (%)','perm. no teste'))
for nm,j in [('Sombreamento',3),('pH',5),('ruído normal',6),('código (tipo ID)',7)]:
    print('%-22s %14s %16s'%(nm,c(100*a[j],1),c(p[j],3)))
print()
print('A importância por impureza atribui uma fatia visível a colunas sem')
print('NENHUMA relação com a nota, porque elas oferecem muitos cortes possíveis')
print('para ajustar ruído. A permutação no teste dá a elas praticamente zero.')

print('\n'+'='*70)
print('DEPENDÊNCIA PARCIAL: a forma do efeito da altitude')
print('Média das predições com a altitude fixada em cada valor, as demais nos')
print('valores observados:')
print()
print('%10s %14s %14s'%('altitude','floresta','verdade'))
for a in [650,800,950,1040,1060,1150,1300,1450]:
    Z=X.copy(); Z[:,0]=a
    print('%10d %14s %14s'%(a,c(float(rf.predict(Z).mean()),3),c(float(verdade(Z).mean()),3)))
print()
print('O degrau aparece nítido: o corte em altitude é tão forte que quase todas')
print('as árvores o fazem perto de 1.050 m. Entre os degraus, a média de muitas')
print('escadas com cortes diferentes vira uma rampa suave.')

print('\n'+'='*70)
print('CLASSIFICAÇÃO (fino ou não), 10 dobras x 3 repetições')
print()
print('%-30s %10s %10s %10s'%('modelo','erro','log-loss','Brier'))
for mt in [1,2,3,6]:
    er,ll,br=cv_clf(lambda mt=mt: RandomForestClassifier(300,max_features=mt,random_state=0),X,FINO,10,0,3,prob=True)
    print('%-30s %10s %10s %10s'%('random forest, mtry = %d'%mt,c(er),c(ll),c(br)))
print('%-30s %10s'%('erro de Bayes',c(ERRO_BAYES)))
