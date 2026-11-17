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
from scipy.stats import norm as N
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

y=FINO
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: fino (1) ou não (0). Erro de Bayes = %s.'%c(ERRO_BAYES))

print('\n'+'='*70)
print('O QUE O MODELO ESTIMA: uma média e um desvio-padrão por variável e classe')
print()
print('%-14s %12s %10s %12s %10s'%('variável','média fino','dp fino','média não','dp não'))
par={}
for j,k in enumerate(NOMES):
    a,b=X[y==1,j],X[y==0,j]; par[k]=(a.mean(),a.std(),b.mean(),b.std())
    print('%-14s %12s %10s %12s %10s'%(ROT[k],c(a.mean(),2),c(a.std(),2),c(b.mean(),2),c(b.std(),2)))
pi1=y.mean()
print('  prioris: P(fino) = %s, P(não) = %s'%(c(pi1),c(1-pi1)))

print('\n'+'='*70)
print('O CÁLCULO PARA UM LOTE NOVO, passo a passo')
x0=np.array([1080.,20.0,1500.,30.,2.9,5.4])
print('Lote: altitude 1080 m, 20,0 C, 1500 mm, 30% de sombra, N 2,9, pH 5,4')
print()
print('%-14s %16s %16s %14s'%('variável','dens. | fino','dens. | não','log da razão'))
L1=np.log(pi1); L0=np.log(1-pi1)
for j,k in enumerate(NOMES):
    m1,s1,m0,s0=par[k]; d1=N.pdf(x0[j],m1,s1); d0=N.pdf(x0[j],m0,s0)
    L1+=np.log(d1); L0+=np.log(d0)
    print('%-14s %16s %16s %14s'%(ROT[k],'%.4e'%d1,'%.4e'%d0,c(np.log(d1/d0),4)))
print()
print('  log P(fino) + soma = %s ;  log P(não) + soma = %s'%(c(L1),c(L0)))
p=1/(1+np.exp(L0-L1))
print('  P(fino | x) = 1 / (1 + exp(%s)) = %s'%(c(L0-L1),c(p)))
gnb=GaussianNB(var_smoothing=0).fit(X,y)
print('  conferência com GaussianNB: %s'%c(gnb.predict_proba(x0[None,:])[0,1]))
print('  P(fino | x) VERDADEIRA = %s'%c(float(1-norm.cdf((80-verdade(x0[None,:])[0])/SIGN))))

print('\n'+'='*70)
print('A HIPÓTESE "INGÊNUA" E O QUE ELA CUSTA')
print('Correlação entre altitude e temperatura, DENTRO de cada classe:')
for cl,nm in [(1,'finos'),(0,'não finos')]:
    print('  %-10s r = %s'%(nm,c(np.corrcoef(X[y==cl,0],X[y==cl,1])[0,1],3)))
print('O modelo trata as duas como evidências independentes: conta duas vezes')
print('a mesma informação. A consequência aparece nas probabilidades.')
d=dobras(NC,10,0)
def oof(f):
    p=np.empty(NC)
    for j in range(10):
        tr,te=d!=j,d==j; p[te]=f().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return p
pn=oof(lambda: GaussianNB()); pl=oof(lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1e4,max_iter=3000)))
Xs=np.delete(X,1,1)
pns=np.empty(NC)
for j in range(10):
    tr,te=d!=j,d==j; pns[te]=GaussianNB().fit(Xs[tr],y[tr]).predict_proba(Xs[te])[:,1]
print()
print('Calibração: média da probabilidade prevista contra a frequência real,')
print('por faixa de probabilidade (validação cruzada de 10 dobras):')
print()
print('%-12s %22s %22s'%('faixa','naive Bayes (6 var.)','naive Bayes sem temp.'))
print('%-12s %7s %7s %6s %7s %7s %6s'%('','prev.','real','n','prev.','real','n'))
for a,b in [(0,.05),(.05,.2),(.2,.4),(.4,.6),(.6,.8),(.8,.95),(.95,1.01)]:
    s1=(pn>=a)&(pn<b); s2=(pns>=a)&(pns<b)
    f=lambda s,pp: (c(pp[s].mean(),3) if s.sum() else '-', c(y[s].mean(),3) if s.sum() else '-', s.sum())
    r1=f(s1,pn); r2=f(s2,pns)
    print('%-12s %7s %7s %6d %7s %7s %6d'%('%s-%s'%(c(a,2),c(min(b,1),2)),*r1,*r2))
print()
print('  fração de probabilidades extremas (< 0,05 ou > 0,95): %s com 6 var., %s sem temp.'%(
      c(np.mean((pn<.05)|(pn>.95)),3),c(np.mean((pns<.05)|(pns>.95)),3)))

print('\n'+'='*70)
print('COMPARAÇÃO (10 dobras x 5 repetições)')
print()
print('%-36s %10s %10s %10s'%('modelo','erro','log-loss','Brier'))
for nm,f,A in [('naive Bayes gaussiano, 6 variáveis',lambda: GaussianNB(),X),
               ('naive Bayes sem a temperatura',lambda: GaussianNB(),Xs),
               ('LDA (covariância comum)',lambda: LinearDiscriminantAnalysis(),X),
               ('QDA (covariância por classe)',lambda: QuadraticDiscriminantAnalysis(),X),
               ('regressão logística',lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1e4,max_iter=3000)),X)]:
    er,ll,br=cv_clf(f,A,y,10,0,5,prob=True)
    print('%-36s %10s %10s %10s'%(nm,c(er),c(ll),c(br)))
print('%-36s %10s'%('erro de Bayes',c(ERRO_BAYES)))

print('\n'+'='*70)
print('NAIVE BAYES COM DENSIDADES POR NÚCLEO (sem supor normalidade)')
from scipy.stats import gaussian_kde
class NBkde:
    def fit(s,A,b):
        s.pi=b.mean(); s.k=[[gaussian_kde(A[b==cl,j]) for j in range(A.shape[1])] for cl in (0,1)]; return s
    def predict_proba(s,A):
        l=[np.log(1-s.pi)+sum(np.log(s.k[0][j](A[:,j])+1e-300) for j in range(A.shape[1])),
           np.log(s.pi)+sum(np.log(s.k[1][j](A[:,j])+1e-300) for j in range(A.shape[1]))]
        p1=1/(1+np.exp(np.clip(l[0]-l[1],-700,700))); return np.column_stack([1-p1,p1])
    def predict(s,A): return (s.predict_proba(A)[:,1]>=.5).astype(int)
er,ll,br=cv_clf(lambda: NBkde(),X,y,10,0,5,prob=True)
print('  erro %s | log-loss %s | Brier %s'%(c(er),c(ll),c(br)))
print('A altitude tem distribuição uniforme, e a nota tem um degrau: dentro')
print('de cada classe a altitude NÃO é normal. O núcleo acomoda isso.')
print('  assimetria da altitude nos finos: %s'%c(float(((X[y==1,0]-X[y==1,0].mean())**3).mean()/X[y==1,0].std()**3),3))
