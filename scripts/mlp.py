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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

y=FINO
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: fino (1) ou não (0). Erro de Bayes = %s.'%c(ERRO_BAYES))
Z,mu,sd=padroniza(X); n,p=Z.shape
sig=lambda t: 1/(1+np.exp(-t))

print('\n'+'='*70)
print('UMA CAMADA OCULTA, À MÃO: propagação e retropropagação')
print('  h = tanh(W1 z + b1)   (H neurônios ocultos)')
print('  q = sigma(w2 . h + b2)')
rg=np.random.default_rng(0); H=3
W1=rg.normal(0,0.5,(H,p)); b1=np.zeros(H); w2=rg.normal(0,0.5,H); b2=0.0
i=10; z=Z[i]; yi=y[i]
a1=W1@z+b1; h=np.tanh(a1); a2=w2@h+b2; q=sig(a2); L=-(yi*np.log(q)+(1-yi)*np.log(1-q))
print()
print('Lote %d (classe %d), rede com H = 3 e pesos iniciais sorteados:'%(i+1,yi))
print('  entradas ocultas a1 = '+', '.join(c(v) for v in a1))
print('  ativações h = tanh(a1) = '+', '.join(c(v) for v in h))
print('  a2 = %s ; q = sigma(a2) = %s ; perda = %s'%(c(a2),c(q),c(L)))
d2=q-yi; gw2=d2*h; gb2=d2
d1=d2*w2*(1-h**2); gW1=np.outer(d1,z)
print()
print('Retropropagação (regra da cadeia, de trás para a frente):')
print('  delta da saída  = q - y = %s'%c(d2))
print('  dL/dw2 = delta * h = '+', '.join(c(v) for v in gw2))
print('  delta oculto = delta * w2 * (1 - h²) = '+', '.join(c(v) for v in d1))
print('  dL/dW1[1, altitude] = delta_oculto_1 * z_altitude = %s'%c(gW1[0,0]))
def perda(W1_):
    hh=np.tanh(W1_@z+b1); qq=sig(w2@hh+b2); return -(yi*np.log(qq)+(1-yi)*np.log(1-qq))
e=1e-6; Wp=W1.copy(); Wp[0,0]+=e; Wm=W1.copy(); Wm[0,0]-=e
print('  conferência por diferença finita:          %s'%c((perda(Wp)-perda(Wm))/(2*e)))

print('\n'+'='*70)
print('QUANTOS NEURÔNIOS, QUANTO DECAIMENTO (alpha): erro 10-fold')
print('lbfgs, tanh, 3 repetições da validação; cada ajuste parte de pesos novos.')
Hs=[1,2,4,8,16]; As=[0.0001,0.01,0.1,1.0,3.0]
E=np.zeros((len(Hs),len(As)))
for a,H_ in enumerate(Hs):
    for b,al in enumerate(As):
        E[a,b],_=cv_clf(lambda H_=H_,al=al: make_pipeline(StandardScaler(),MLPClassifier((H_,),activation='tanh',solver='lbfgs',alpha=al,max_iter=3000,random_state=1)),X,y,10,0,3)
print()
print('%8s'%'H \\ alfa'+''.join('%10s'%c(al,4) for al in As))
for a,H_ in enumerate(Hs): print('%8d'%H_+''.join('%10s'%c(E[a,b]) for b in range(len(As))))
a,b=np.unravel_index(np.argmin(E),E.shape); Hb,Ab=Hs[a],As[b]
print('  melhor: H = %d, alfa = %s, erro = %s'%(Hb,c(Ab,4),c(E[a,b])))
print('Sem decaimento, mais neurônios = mais sobreajuste. Com decaimento, o')
print('número de neurônios importa pouco: a penalidade controla a flexibilidade.')

print('\n'+'='*70)
print('MÍNIMOS LOCAIS: o mesmo modelo, 10 pontos de partida')
print()
print('%8s %16s %16s'%('semente','perda no treino','erro 10-fold'))
r_=[]
for s_ in range(10):
    m=make_pipeline(StandardScaler(),MLPClassifier((16,),activation='tanh',solver='lbfgs',alpha=0.0001,max_iter=3000,random_state=s_)).fit(X,y)
    q=np.clip(m.predict_proba(X)[:,1],1e-12,1-1e-12)
    er,_=cv_clf(lambda s_=s_: make_pipeline(StandardScaler(),MLPClassifier((16,),activation='tanh',solver='lbfgs',alpha=0.0001,max_iter=3000,random_state=s_)),X,y,10,0,1)
    r_.append(er); print('%8d %16s %16s'%(s_,c(float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))),c(er)))
print('  amplitude do erro entre sementes: %s a %s'%(c(min(r_)),c(max(r_))))
class Media:
    def __init__(s,k,H_,al): s.k,s.H,s.al=k,H_,al
    def fit(s,A,b):
        s.ms=[make_pipeline(StandardScaler(),MLPClassifier((s.H,),activation='tanh',solver='lbfgs',alpha=s.al,max_iter=3000,random_state=r)).fit(A,b) for r in range(s.k)]; return s
    def predict_proba(s,A): return np.mean([m.predict_proba(A) for m in s.ms],0)
    def predict(s,A): return (s.predict_proba(A)[:,1]>=.5).astype(int)
print()
print('Média das probabilidades de 5 redes com pontos de partida diferentes:')
print()
print('%-34s %10s %10s %10s'%('','erro','log-loss','Brier'))
for H_,al in [(16,0.0001),(Hb,Ab)]:
    e1=cv_clf(lambda: make_pipeline(StandardScaler(),MLPClassifier((H_,),activation='tanh',solver='lbfgs',alpha=al,max_iter=3000,random_state=1)),X,y,10,0,3,prob=True)
    e5=cv_clf(lambda: Media(5,H_,al),X,y,10,0,3,prob=True)
    print('%-34s %10s %10s %10s'%('H = %d, alfa = %s: uma rede'%(H_,c(al,4)),*[c(v) for v in e1]))
    print('%-34s %10s %10s %10s'%('H = %d, alfa = %s: média de 5'%(H_,c(al,4)),*[c(v) for v in e5]))
print('%-34s %10s'%('erro de Bayes',c(ERRO_BAYES)))
print('Sem penalidade, cada semente cai num mínimo diferente e a média ajuda')
print('muito. Com penalidade forte, as sementes chegam praticamente à mesma')
print('solução, e a média não tem o que corrigir.')

print('\n'+'='*70)
print('A INTERAÇÃO, agora representável')
print('Efeito na nota de passar a sombra de 10% a 60% (MLP de regressão):')
mr=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(8,),activation='tanh',solver='lbfgs',alpha=1.0,max_iter=5000,random_state=0)).fit(X,NOTA)
print()
print('%-22s %12s %12s'%('','MLP','verdade'))
for T in [18.5,22.5]:
    A=X.copy(); A[:,1]=T; A[:,3]=10; Bm=A.copy(); Bm[:,3]=60
    print('%-22s %12s %12s'%('temperatura %s C'%c(T,1),c(float((mr.predict(Bm)-mr.predict(A)).mean()),3),c(float((verdade(Bm)-verdade(A)).mean()),3)))

print('\n'+'='*70)
print('REGRESSÃO DA NOTA: o MLP contra o piso')
print()
print('%-26s %14s'%('modelo','EQM 10-fold'))
for nm,f in [('MLP H = 4, alfa = 1',lambda: make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(4,),activation='tanh',solver='lbfgs',alpha=1.0,max_iter=5000,random_state=0))),
             ('MLP H = 8, alfa = 1',lambda: make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(8,),activation='tanh',solver='lbfgs',alpha=1.0,max_iter=5000,random_state=0))),
             ('MLP H = 8, alfa = 10',lambda: make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(8,),activation='tanh',solver='lbfgs',alpha=10.0,max_iter=5000,random_state=0))),
             ('MLP H = (16, 8), alfa = 3',lambda: make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,8),activation='tanh',solver='lbfgs',alpha=3.0,max_iter=5000,random_state=0)))]:
    v,_=cv_reg(f,X,NOTA,10,0,3); print('%-26s %14s'%(nm,c(v)))
print('%-26s %14s'%('piso (sigma²)',c(EQM_PISO)))
