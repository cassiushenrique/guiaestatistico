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
from sklearn.linear_model import LogisticRegression

y=FINO
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: fino (1) ou não (0). Erro de Bayes = %s.'%c(ERRO_BAYES))
Z,mu,sd=padroniza(X); n,p=Z.shape
sig=lambda t: 1/(1+np.exp(-t))

print('\n'+'='*70)
print('UM NEURÔNIO: entrada ponderada, soma, função de ativação')
print('  a = b + w1 z1 + ... + w6 z6 ;  saída = sigma(a) = 1/(1 + e^-a)')
print('É exatamente o modelo de regressão logística, escrito como rede.')

print('\n'+'='*70)
print('TREINO POR DESCIDA DO GRADIENTE, à mão')
print('Perda: entropia cruzada média  L = -(1/n) soma[y log q + (1-y) log(1-q)]')
print('Gradiente: dL/dw = (1/n) Z\'(q - y) ;  dL/db = média(q - y)')
def treina(Z,y,eta,iters,lam=0.0,w0=None):
    w=np.zeros(Z.shape[1]) if w0 is None else w0.copy(); b=0.0; hist=[]
    for t in range(iters):
        q=sig(Z@w+b); g=Z.T@(q-y)/len(y)+lam*w; gb=np.mean(q-y)
        w-=eta*g; b-=eta*gb
        hist.append(float(-np.mean(y*np.log(q+1e-12)+(1-y)*np.log(1-q+1e-12))))
    return w,b,hist
print()
print('As três primeiras iterações, com taxa de aprendizado eta = 0,5:')
w=np.zeros(p); b=0.0
for t in range(3):
    q=sig(Z@w+b); g=Z.T@(q-y)/n; gb=np.mean(q-y)
    print('  it. %d: perda = %s | grad. altitude = %s | grad. b = %s'%(t+1,c(float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))),c(g[0]),c(gb)))
    w-=0.5*g; b-=0.5*gb
    print('          novo w_altitude = %s, novo b = %s'%(c(w[0]),c(b)))

print('\n'+'='*70)
print('A TAXA DE APRENDIZADO')
print()
print('%8s %14s %14s %14s %14s'%('eta','perda it. 10','perda it. 100','perda it. 2000','convergiu?'))
ref=LogisticRegression(C=1e10,max_iter=10000,tol=1e-12).fit(Z,y)
Lref=float(-np.mean(y*np.log(ref.predict_proba(Z)[:,1])+(1-y)*np.log(ref.predict_proba(Z)[:,0])))
for eta in [0.01,0.1,1.0,5.0,30.0]:
    _,_,h=treina(Z,y,eta,2000)
    print('%8s %14s %14s %14s %14s'%(c(eta,2),c(h[9]),c(h[99]),c(h[-1]),'sim' if abs(h[-1]-Lref)<1e-4 else 'não'))
print('  perda mínima (máxima verossimilhança) = %s'%c(Lref))

print('\n'+'='*70)
print('O NEURÔNIO TREINADO É A REGRESSÃO LOGÍSTICA')
w,b,h=treina(Z,y,1.0,20000)
print()
print('%-14s %16s %20s'%('peso','descida do grad.','glm (máx. veross.)'))
print('%-14s %16s %20s'%('intercepto',c(b),c(ref.intercept_[0])))
for j,k in enumerate(NOMES): print('%-14s %16s %20s'%(ROT[k],c(w[j]),c(ref.coef_[0,j])))

print('\n'+'='*70)
print('DECAIMENTO DE PESOS (weight decay) = penalidade ridge')
print('Perda + (lambda/2) |w|²; a rede encolhe os pesos em direção a zero.')
print()
print('%10s %12s %12s %16s'%('lambda','|w|','w_altitude','erro 10-fold'))
class Neur:
    def __init__(s,lam): s.lam=lam
    def fit(s,A,b):
        s.m,s.s=A.mean(0),A.std(0,ddof=1); s.w,s.b,_=treina((A-s.m)/s.s,b,1.0,3000,s.lam); return s
    def predict_proba(s,A):
        q=sig(((A-s.m)/s.s)@s.w+s.b); return np.column_stack([1-q,q])
    def predict(s,A): return (s.predict_proba(A)[:,1]>=.5).astype(int)
for lam in [0,0.001,0.01,0.1,1.0]:
    w,_,_=treina(Z,y,1.0,3000,lam); er,_=cv_clf(lambda lam=lam: Neur(lam),X,y,10,0,2)
    print('%10s %12s %12s %16s'%(c(lam,3),c(np.linalg.norm(w)),c(w[0]),c(er)))

print('\n'+'='*70)
print('O PERCEPTRON DE ROSENBLATT: saída em degrau e regra de correção')
print('Se o lote é mal classificado: w <- w + y_i z_i (rótulos -1 e +1).')
ys=2*y-1; w=np.zeros(p); b=0.0; rg=np.random.default_rng(0)
print()
print('%8s %18s'%('época','erros no treino'))
for ep in range(1,51):
    for i in rg.permutation(n):
        if ys[i]*(Z[i]@w+b)<=0: w+=ys[i]*Z[i]; b+=ys[i]
    if ep in (1,2,5,10,20,50): print('%8d %18d'%(ep,int(np.sum(ys*(Z@w+b)<=0))))
print('As classes não são linearmente separáveis: o perceptron nunca para de')
print('corrigir, e o número de erros oscila sem convergir.')

print('\n'+'='*70)
print('MAIS DE DUAS CLASSES: a camada de saída softmax')
cls=np.digitize(NOTA,[76,80])           # 0 = comercial (<76), 1 = bom, 2 = fino
print('Classes: comercial (nota < 76), bom (76 a 80), fino (>= 80).')
print('  contagens: %s'%', '.join('%d'%np.sum(cls==k) for k in range(3)))
Y=np.eye(3)[cls]; W=np.zeros((p,3)); B=np.zeros(3)
for t in range(5000):
    A=Z@W+B; A-=A.max(1,keepdims=True); P=np.exp(A); P/=P.sum(1,keepdims=True)
    W-=0.5*Z.T@(P-Y)/n; B-=0.5*np.mean(P-Y,0)
pred=P.argmax(1)
print()
print('Matriz de confusão (treino):')
print('%12s %10s %10s %10s'%('real \\ prev.','comercial','bom','fino'))
for k,nm in enumerate(['comercial','bom','fino']):
    print('%12s %10d %10d %10d'%(nm,*[np.sum((cls==k)&(pred==j)) for j in range(3)]))
print('  acurácia = %s'%c(acur(cls,pred)))

print('\n'+'='*70)
print('O QUE UM NEURÔNIO NÃO CONSEGUE: a interação')
print('A fronteira é um hiperplano. O efeito do sombreamento tem de ter o')
print('MESMO sinal em qualquer temperatura. Nos dados, comparando a nota média')
print('de lotes com sombra alta e baixa, separadamente por clima:')
for nm,s_ in [('temperatura <= 20,5 C',X[:,1]<=20.5),('temperatura > 20,5 C',X[:,1]>20.5)]:
    hi=s_&(X[:,3]>=35); lo=s_&(X[:,3]<20)
    print('  %-22s nota média, sombra >= 35%%: %s (n=%d) | sombra < 20%%: %s (n=%d)'%(nm,c(NOTA[hi].mean(),2),hi.sum(),c(NOTA[lo].mean(),2),lo.sum()))
w,b,_=treina(Z,y,1.0,20000)
print('  peso do sombreamento no neurônio: %s (um só sinal para os dois climas)'%c(w[3]))
print('A ficha seguinte acrescenta uma camada oculta para resolver isso.')
