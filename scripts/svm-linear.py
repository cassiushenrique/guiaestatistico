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
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

y=FINO; ys=2*y-1                        # a SVM usa rótulos -1 e +1
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: fino (+1) ou não (-1). Erro de Bayes = %s.'%c(ERRO_BAYES))
Z,mu,sd=padroniza(X)

print('\n'+'='*70)
print('PRIMEIRO, EM DUAS DIMENSÕES: altitude e pH, padronizadas')
Z2=Z[:,[0,5]]
print()
print('%10s %12s %12s %12s %10s %12s'%('C','w_alt','w_pH','b','n. vet. sup.','margem 2/|w|'))
for C in [0.01,0.1,1,10,100]:
    m=SVC(kernel='linear',C=C).fit(Z2,ys); w=m.coef_[0]; b=m.intercept_[0]
    print('%10s %12s %12s %12s %10d %12s'%(c(C,2),c(w[0]),c(w[1]),c(b),m.n_support_.sum(),c(2/np.linalg.norm(w))))
print()
print('C pequeno: margem LARGA, muitos lotes dentro dela (muitos vetores de')
print('suporte). C grande: margem estreita, a fronteira segue os lotes difíceis.')

print('\n'+'='*70)
print('A FUNÇÃO DE DECISÃO PARA UM LOTE, e a perda hinge')
m=SVC(kernel='linear',C=1).fit(Z2,ys); w=m.coef_[0]; b=m.intercept_[0]
for i in [10,25,40]:
    f=float(Z2[i]@w+b); marg=ys[i]*f
    print('  lote %d: z_alt = %s, z_pH = %s, f(x) = %s*%s + %s*%s + (%s) = %s'%(i+1,c(Z2[i,0],3),c(Z2[i,1],3),c(w[0],3),c(Z2[i,0],3),c(w[1],3),c(Z2[i,1],3),c(b,3),c(f,3)))
    print('           classe real %+d, y*f = %s, perda hinge max(0, 1 - y*f) = %s'%(ys[i],c(marg,3),c(max(0,1-marg),3)))
fx=Z2@w+b; mg=ys*fx
print()
print('  lotes com y*f > 1 (fora da margem, perda zero):     %d'%np.sum(mg>1+1e-6))
print('  lotes com 0 < y*f <= 1 (dentro da margem, certos):  %d'%np.sum((mg>0)&(mg<=1+1e-6)))
print('  lotes com y*f <= 0 (do lado errado):               %d'%np.sum(mg<=0))
print('  vetores de suporte declarados pelo ajuste:          %d'%m.n_support_.sum())
print('Só os vetores de suporte determinam a fronteira: os %d lotes de perda'%np.sum(mg>1+1e-6))
print('zero poderiam ser removidos sem mudar nada.')

print('\n'+'='*70)
print('COM AS SEIS VARIÁVEIS: escolhendo C por validação cruzada')
print()
print('%10s %16s %14s'%('C','erro 10-fold','vet. suporte'))
best=(1,None)
for C in [0.001,0.01,0.1,1,10,100]:
    f=lambda C=C: make_pipeline(StandardScaler(),SVC(kernel='linear',C=C))
    er,_=cv_clf(f,X,y,10,0,5); nsv=f().fit(X,y)[-1].n_support_.sum()
    print('%10s %16s %14d'%(c(C,3),c(er),nsv))
    if er<best[0]: best=(er,C)
Cb=best[1]
m=make_pipeline(StandardScaler(),SVC(kernel='linear',C=Cb)).fit(X,y)
print()
print('Coeficientes (C = %s), na escala padronizada:'%c(Cb,3))
lg=make_pipeline(StandardScaler(),LogisticRegression(C=1e4,max_iter=3000)).fit(X,y)
wl=lg[-1].coef_[0]; ws=m[-1].coef_[0]
print('%-14s %12s %14s %18s'%('variável','SVM','logística','razão SVM/log.'))
for j,k in enumerate(NOMES): print('%-14s %12s %14s %18s'%(ROT[k],c(ws[j]),c(wl[j]),c(ws[j]/wl[j],3)))
cos=float(ws@wl/np.linalg.norm(ws)/np.linalg.norm(wl))
print('Cosseno do ângulo entre as duas direções = %s.'%c(cos))
print('A altitude domina as duas. Mas a temperatura troca de sinal: ela tem')
print('correlação %s com a altitude, e a SVM com C pequeno (regularização'%c(np.corrcoef(X[:,0],X[:,1])[0,1],3))
print('forte) reparte o efeito entre as duas, como faria a regressão ridge;')
print('a logística quase sem penalidade não reparte.')

print('\n'+'='*70)
print('SEM PADRONIZAR')
er1,_=cv_clf(lambda: SVC(kernel='linear',C=Cb,max_iter=200000),X,y,10,0,1)
er2,_=cv_clf(lambda: make_pipeline(StandardScaler(),SVC(kernel='linear',C=Cb)),X,y,10,0,1)
print('  erro 10-fold sem padronizar = %s | padronizado = %s'%(c(er1),c(er2)))

print('\n'+'='*70)
print('PROBABILIDADES: a SVM não as dá. Platt ajusta uma logística ao f(x).')
er,ll,br=cv_clf(lambda: make_pipeline(StandardScaler(),SVC(kernel='linear',C=Cb,probability=True,random_state=0)),X,y,10,0,3,prob=True)
er2,ll2,br2=cv_clf(lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1e4,max_iter=3000)),X,y,10,0,3,prob=True)
print()
print('%-30s %10s %10s %10s'%('modelo','erro','log-loss','Brier'))
print('%-30s %10s %10s %10s'%('SVM linear + Platt',c(er),c(ll),c(br)))
print('%-30s %10s %10s %10s'%('regressão logística',c(er2),c(ll2),c(br2)))
print('%-30s %10s'%('erro de Bayes',c(ERRO_BAYES)))

print('\n'+'='*70)
print('UMA FRONTEIRA LINEAR NÃO VÊ O DEGRAU COMO DEGRAU')
print('Mas pode-se dar a ela uma variável nova: o indicador "altitude >= 1050".')
Xd=np.column_stack([X,(X[:,0]>=1050).astype(float)])
Xi=np.column_stack([Xd,(X[:,1]>20.5)*X[:,3],(X[:,1]<=20.5)*X[:,3]])
for nm,A in [('seis variáveis originais',X),('+ indicador do degrau',Xd),('+ degrau + sombra por clima',Xi)]:
    er,_=cv_clf(lambda: make_pipeline(StandardScaler(),SVC(kernel='linear',C=Cb)),A,y,10,0,5)
    print('  %-32s erro 10-fold = %s'%(nm,c(er)))
print('A SVM linear é exatamente tão boa quanto as variáveis que recebe.')
