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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

y=FINO
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: fino (1) ou não (0). Erro de Bayes = %s.'%c(ERRO_BAYES))

print('\n'+'='*70)
print('UM LOTE NOVO E SEUS VIZINHOS')
x0=np.array([1080.,20.0,1500.,30.,2.9,5.4])
print('Lote novo: altitude 1080 m, 20,0 C, 1500 mm, 30% de sombra, N 2,9, pH 5,4')
print('P(fino) VERDADEIRA para esse lote = %s'%c(float(1-norm.cdf((80-verdade(x0[None,:])[0])/SIGN))))
Z,mu,sd=padroniza(X); z0=(x0-mu)/sd
dz=np.sqrt(((Z-z0)**2).sum(1)); dr=np.sqrt(((X-x0)**2).sum(1))
print()
print('Os 7 vizinhos mais próximos, nas variáveis PADRONIZADAS:')
print('%6s %9s %7s %8s %7s %6s %6s %10s %6s'%('lote','altitude','temp','chuva','sombra','N','pH','distância','fino'))
for i in np.argsort(dz)[:7]:
    a=X[i]; print('%6d %9s %7s %8s %7s %6s %6s %10s %6d'%(i+1,c(a[0],0),c(a[1],1),c(a[2],0),c(a[3],0),c(a[4],2),c(a[5],2),c(dz[i],3),y[i]))
v7=np.argsort(dz)[:7]
print('  -> %d de 7 são finos: P(fino) estimada = %s'%(y[v7].sum(),c(y[v7].mean())))
print()
print('Os 7 vizinhos mais próximos SEM padronizar (distância em unidades brutas):')
for i in np.argsort(dr)[:7]:
    a=X[i]; print('%6d %9s %7s %8s %7s %6s %6s %10s %6d'%(i+1,c(a[0],0),c(a[1],1),c(a[2],0),c(a[3],0),c(a[4],2),c(a[5],2),c(dr[i],1),y[i]))
v7r=np.argsort(dr)[:7]
print('  -> %d de 7 são finos: P(fino) estimada = %s'%(y[v7r].sum(),c(y[v7r].mean())))
print('Sem padronizar, só a altitude e a chuva (desvios-padrão de ~260) pesam na')
print('distância; temperatura, N e pH (desvios abaixo de 2) são quase ignorados:')
print('veja o pH dos "vizinhos" indo de 4,87 a 6,13.')
print('  desvios-padrão: '+', '.join('%s %s'%(ROT[k],c(s_,2)) for k,s_ in zip(NOMES,sd)))

print('\n'+'='*70)
print('A ESCOLHA DE k: viés contra variância')
print()
print('%5s %14s %16s %16s'%('k','erro treino','erro 10-fold','erro (sem padr.)'))
res=[]
for k in [1,3,5,9,15,25,41,61,101,201]:
    f=lambda k=k: make_pipeline(StandardScaler(),KNeighborsClassifier(k))
    tr=1-acur(y,f().fit(X,y).predict(X))
    er,sdv=cv_clf(f,X,y,10,0,5)
    ebr,_=cv_clf(lambda k=k: KNeighborsClassifier(k),X,y,10,0,5)
    res.append((k,er)); print('%5d %14s %16s %16s'%(k,c(tr),c(er),c(ebr)))
kb=min(res,key=lambda t:t[1])[0]
print()
print('  melhor k = %d. Com k = 1 o erro de treino é zero: cada lote é o seu'%kb)
print('  próprio vizinho. Com k = 201, a vizinhança é um terço dos dados.')

print('\n'+'='*70)
print('PONDERAR PELA DISTÂNCIA, e as probabilidades')
print()
print('%-34s %10s %10s %10s'%('modelo','erro','log-loss','Brier'))
for nm,f in [('k = %d, votos iguais'%kb,lambda: make_pipeline(StandardScaler(),KNeighborsClassifier(kb))),
             ('k = %d, peso 1/distância'%kb,lambda: make_pipeline(StandardScaler(),KNeighborsClassifier(kb,weights='distance'))),
             ('k = 1',lambda: make_pipeline(StandardScaler(),KNeighborsClassifier(1)))]:
    er,ll,br=cv_clf(f,X,y,10,0,5,prob=True)
    print('%-34s %10s %10s %10s'%(nm,c(er),c(ll),c(br)))
print('Com k = 1 a probabilidade estimada é 0 ou 1: o log-loss explode.')

print('\n'+'='*70)
print('A MALDIÇÃO DA DIMENSIONALIDADE: acrescentando colunas de puro ruído')
rg=np.random.default_rng(4)
print()
print('%18s %14s %22s'%('colunas de ruído','erro 10-fold','dist. ao vizinho/média'))
for q in [0,2,6,12,24,48]:
    A=np.column_stack([X,rg.normal(size=(NC,q))]) if q else X
    er,_=cv_clf(lambda: make_pipeline(StandardScaler(),KNeighborsClassifier(kb)),A,y,10,0,3)
    Za=padroniza(A)[0]; D=np.sqrt(((Za[:,None,:]-Za[None,:,:])**2).sum(2)); np.fill_diagonal(D,np.inf)
    rel=float(np.mean(D.min(1))/np.mean(D[np.isfinite(D)]))
    print('%18d %14s %22s'%(q,c(er),c(rel,3)))
print()
print('Em dimensão alta, o vizinho mais próximo fica quase tão longe quanto um')
print('lote qualquer: a ideia de "vizinhança" deixa de existir.')

print('\n'+'='*70)
print('kNN PARA REGRESSÃO (a nota), comparado com o piso')
print()
for k in [1,5,15,25,50]:
    v,_=cv_reg(lambda k=k: make_pipeline(StandardScaler(),KNeighborsRegressor(k)),X,NOTA,10,0,3)
    print('  k = %3d   EQM 10-fold = %s'%(k,c(v)))
print('  piso (sigma²) = %s'%c(EQM_PISO))
