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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

y=NOTA
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: a nota. Piso irredutível sigma² = %s.'%c(EQM_PISO))

print('\n'+'='*70)
print('BOOSTING À MÃO: cada árvore pequena ajusta o RESÍDUO das anteriores')
print('F_0 = média; F_m = F_(m-1) + nu * árvore ajustada a (y - F_(m-1))')
print('Árvores de profundidade 1 (tocos) e nu = 0,1:')
nu=0.1; F=np.full(NC,y.mean())
print()
print('%4s %-26s %12s %12s'%('m','corte do toco','passo (nu*h)','EQM treino'))
print('%4d %-26s %12s %12s'%(0,'(média)','',c(eqm(y,F))))
for m in range(1,9):
    r=y-F; h=DecisionTreeRegressor(max_depth=1).fit(X,r); t=h.tree_
    F=F+nu*h.predict(X)
    v=t.value.ravel()
    print('%4d %-26s %12s %12s'%(m,'%s <= %s'%(ROT[NOMES[t.feature[0]]],c(t.threshold[0],1)),
          '%s / %s'%(c(nu*v[1],3),c(nu*v[2],3)),c(eqm(y,F))))
print()
print('Os primeiros tocos insistem na altitude: o degrau é grande, e cada passo')
print('de tamanho 0,1 retira só 10% do que falta explicar ali.')

print('\n'+'='*70)
print('AO CONTRÁRIO DA RANDOM FOREST, O BOOSTING SOBREAJUSTA COM M')
print('Treino em 400 lotes, teste em 200; profundidade 3.')
rg=np.random.default_rng(5); idx=rg.permutation(NC); te,tr=idx[:200],idx[200:]
print()
print('%7s %14s %14s %14s'%('M','nu = 1,0','nu = 0,1','nu = 0,01'))
cur={}
for nu_ in [1.0,0.1,0.01]:
    g=GradientBoostingRegressor(n_estimators=3000,learning_rate=nu_,max_depth=3,random_state=0).fit(X[tr],y[tr])
    cur[nu_]=np.array([eqm(y[te],p) for p in g.staged_predict(X[te])])
for M in [1,10,30,100,300,1000,3000]:
    print('%7d %14s %14s %14s'%(M,c(cur[1.0][M-1]),c(cur[0.1][M-1]),c(cur[0.01][M-1])))
print()
for nu_ in [1.0,0.1,0.01]:
    k=int(np.argmin(cur[nu_])); print('  nu = %-5s melhor M = %5d, EQM teste = %s'%(c(nu_,2),k+1,c(cur[nu_][k])))
print('Com nu = 1 o mínimo é alto e estreito: duas árvores, e depois só piora.')
print('Passar a nu = 0,1 baixa o mínimo e o alarga. Passar a 0,01 já não baixa:')
print('só exige cerca de dez vezes mais árvores para chegar ao mesmo lugar.')

print('\n'+'='*70)
print('A PROFUNDIDADE É A ORDEM DAS INTERAÇÕES')
print('Tocos (profundidade 1) só somam funções de UMA variável: o modelo é')
print('aditivo e não representa a interação sombreamento x temperatura.')
print('M escolhido por validação interna (10% dos dados, parada antecipada).')
print()
print('%13s %14s %14s'%('profundidade','EQM 10-fold','M médio'))
for dp in [1,2,3,4,6]:
    Ms=[]
    def fab(dp=dp):
        return GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=dp,subsample=1.0,
                 validation_fraction=0.1,n_iter_no_change=50,random_state=0)
    d=dobras(NC,10,0); e=np.empty(NC)
    for j in range(10):
        trj,tej=d!=j,d==j; m=fab().fit(X[trj],y[trj]); e[tej]=y[tej]-m.predict(X[tej]); Ms.append(m.n_estimators_)
    print('%13d %14s %14d'%(dp,c(float(np.mean(e**2))),int(np.mean(Ms))))

print('\n'+'='*70)
print('A INTERAÇÃO, vista diretamente')
print('Efeito de passar o sombreamento de 10% a 60%, em dois climas:')
g1=GradientBoostingRegressor(n_estimators=600,learning_rate=0.05,max_depth=1,random_state=0).fit(X,y)
g3=GradientBoostingRegressor(n_estimators=400,learning_rate=0.05,max_depth=3,random_state=0).fit(X,y)
print()
print('%-22s %14s %14s %14s'%('','tocos','profund. 3','verdade'))
for T in [18.5,22.5]:
    Z=X.copy(); Z[:,1]=T; A=Z.copy(); A[:,3]=10; Bm=Z.copy(); Bm[:,3]=60
    r=[float((m.predict(Bm)-m.predict(A)).mean()) for m in (g1,g3)]+[float((verdade(Bm)-verdade(A)).mean())]
    print('%-22s %14s %14s %14s'%('temperatura %s C'%c(T,1),*[c(v,3) for v in r]))
print()
print('Os tocos dão o MESMO efeito do sombreamento nos dois climas, porque')
print('não podem fazer outra coisa. A verdade muda de sinal.')

print('\n'+'='*70)
print('BOOSTING ESTOCÁSTICO: cada árvore numa subamostra sem reposição')
print()
print('%11s %14s'%('subamostra','EQM 10-fold'))
for sub in [1.0,0.8,0.5,0.3]:
    v,_=cv_reg(lambda sub=sub: GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=3,subsample=sub,random_state=0),X,y,10,0,3)
    print('%11s %14s'%(c(sub,1),c(v)))

print('\n'+'='*70)
print('CLASSIFICAÇÃO: o AdaBoost e o boosting com perda logística')
yc=FINO
print()
print('%-40s %10s %10s %10s'%('modelo','erro','log-loss','Brier'))
for nm,f in [('um toco',lambda: DecisionTreeClassifier(max_depth=1)),
             ('AdaBoost, 200 tocos',lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200,random_state=0)),
             ('gradient boosting, logística, prof. 1',lambda: GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,max_depth=1,random_state=0)),
             ('gradient boosting, logística, prof. 3',lambda: GradientBoostingClassifier(n_estimators=150,learning_rate=0.05,max_depth=3,random_state=0))]:
    er,ll,br=cv_clf(f,X,yc,10,0,3,prob=True)
    print('%-40s %10s %10s %10s'%(nm,c(er),c(ll),c(br)))
print('%-40s %10s'%('erro de Bayes',c(ERRO_BAYES)))
