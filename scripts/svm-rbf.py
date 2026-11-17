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
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

y=FINO
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: fino (1) ou não (0). Erro de Bayes = %s.'%c(ERRO_BAYES))
Z,mu,sd=padroniza(X)

print('\n'+'='*70)
print('O NÚCLEO GAUSSIANO: K(x, x\') = exp(-gamma |x - x\'|²)')
print('Mede a semelhança entre dois lotes; vale 1 para lotes iguais e cai a')
print('zero com a distância. gamma controla a velocidade da queda.')
d2=((Z[0]-Z[1:6])**2).sum(1)
print()
print('%8s %12s %12s %12s %12s'%('lote','|x1 - x|²','gamma=0,05','gamma=0,5','gamma=5'))
for i in range(5):
    print('%8d %12s %12s %12s %12s'%(i+2,c(d2[i],3),*[c(np.exp(-g*d2[i])) for g in (0.05,0.5,5)]))
print()
print('Com gamma = 5 todo lote só "vê" a si mesmo: a fronteira contorna cada')
print('ponto. Com gamma = 0,05 todos se parecem: a fronteira quase reta.')

print('\n'+'='*70)
print('A GRADE DE C E gamma: erro 10-fold (5 repetições)')
Cs=[0.1,1,10,100]; gs=[0.01,0.05,0.2,1,5]
E=np.zeros((len(Cs),len(gs)))
for a,C in enumerate(Cs):
    for b,g in enumerate(gs):
        E[a,b],_=cv_clf(lambda C=C,g=g: make_pipeline(StandardScaler(),SVC(C=C,gamma=g)),X,y,10,0,5)
print()
print('%10s'%'C \\ gamma'+''.join('%10s'%c(g,2) for g in gs))
for a,C in enumerate(Cs): print('%10s'%c(C,1)+''.join('%10s'%c(E[a,b]) for b in range(len(gs))))
a,b=np.unravel_index(np.argmin(E),E.shape); Cb,gb=Cs[a],gs[b]
print()
print('  melhor: C = %s, gamma = %s, erro = %s'%(c(Cb,1),c(gb,2),c(E[a,b])))
print('  padrão do sklearn: gamma = 1/(p var(Z)) = 1/6 = %s'%c(1/6))

print('\n'+'='*70)
print('SOBREAJUSTE: erro de treino contra erro validado')
print()
print('%8s %8s %14s %14s %14s'%('C','gamma','erro treino','erro 10-fold','vet. suporte'))
for C,g in [(1,0.05),(Cb,gb),(10,1),(100,5)]:
    m=make_pipeline(StandardScaler(),SVC(C=C,gamma=g)).fit(X,y)
    er,_=cv_clf(lambda C=C,g=g: make_pipeline(StandardScaler(),SVC(C=C,gamma=g)),X,y,10,0,5)
    print('%8s %8s %14s %14s %14d'%(c(C,1),c(g,2),c(1-acur(y,m.predict(X))),c(er),m[-1].n_support_.sum()))

print('\n'+'='*70)
print('A INTERAÇÃO, que a SVM linear não via')
print('Probabilidade de fino (SVM com Platt) numa grade de sombreamento,')
print('com altitude 1150 m e demais variáveis medianas, em dois climas:')
m=make_pipeline(StandardScaler(),SVC(C=Cb,gamma=gb,probability=True,random_state=0)).fit(X,y)
med=np.median(X,0)
print()
print('%12s %16s %16s %16s %16s'%('sombra (%)','SVM, 18,5 C','verdade, 18,5 C','SVM, 22,5 C','verdade, 22,5 C'))
for s_ in [0,20,40,60,80]:
    out=[]
    for T in (18.5,22.5):
        x=med.copy(); x[0]=1150; x[1]=T; x[3]=s_
        out+= [c(m.predict_proba(x[None,:])[0,1],3), c(float(1-norm.cdf((80-verdade(x[None,:])[0])/SIGN)),3)]
    print('%12d %16s %16s %16s %16s'%(s_,*out))

print('\n'+'='*70)
print('COMPARAÇÃO FINAL (10 dobras x 3 repetições)')
print()
print('%-34s %10s %10s %10s'%('modelo','erro','log-loss','Brier'))
for nm,f in [('SVM linear (C = 0,01)',lambda: make_pipeline(StandardScaler(),SVC(kernel='linear',C=0.01,probability=True,random_state=0))),
             ('SVM RBF (C e gamma escolhidos)',lambda: make_pipeline(StandardScaler(),SVC(C=Cb,gamma=gb,probability=True,random_state=0))),
             ('SVM polinomial de grau 2',lambda: make_pipeline(StandardScaler(),SVC(kernel='poly',degree=2,C=1,coef0=1,probability=True,random_state=0))),
             ('random forest (300 árvores)',lambda: RandomForestClassifier(300,max_features=2,random_state=0))]:
    er,ll,br=cv_clf(f,X,y,10,0,3,prob=True)
    print('%-34s %10s %10s %10s'%(nm,c(er),c(ll),c(br)))
print('%-34s %10s'%('erro de Bayes',c(ERRO_BAYES)))

print('\n'+'='*70)
print('REGRESSÃO POR VETORES DE SUPORTE (SVR) na nota')
print('Perda epsilon-insensível: desvios menores que epsilon não custam nada.')
print()
print('%8s %8s %10s %14s %14s'%('C','gamma','epsilon','EQM 10-fold','vet. suporte'))
for C,g,e in [(10,0.2,0.1),(10,0.2,1.0),(10,0.2,2.0),(100,0.05,1.0),(30,0.1,1.0)]:
    v,_=cv_reg(lambda C=C,g=g,e=e: make_pipeline(StandardScaler(),SVR(C=C,gamma=g,epsilon=e)),X,NOTA,10,0,3)
    ns=make_pipeline(StandardScaler(),SVR(C=C,gamma=g,epsilon=e)).fit(X,NOTA)[-1].support_.size
    print('%8s %8s %10s %14s %14d'%(c(C,0),c(g,2),c(e,1),c(v),ns))
print('  piso (sigma²) = %s'%c(EQM_PISO))
