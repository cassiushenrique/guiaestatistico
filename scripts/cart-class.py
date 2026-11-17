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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

y=FINO
print('LOTES DE CAFÉ:',DADOS_TXT)
print('Resposta: fino (1) ou não (0). Prevalência = %s (%d de %d).'%(c(y.mean()),int(y.sum()),NC))
print('Como o processo gerador é conhecido, sabe-se o ERRO DE BAYES, o menor')
print('erro que QUALQUER classificador pode ter com essas seis variáveis:')
print('  erro de Bayes = %s'%c(ERRO_BAYES))
print('  erro de quem chuta sempre "não fino" = %s'%c(1-y.mean()))

print('\n'+'='*70)
print('AS TRÊS MEDIDAS DE IMPUREZA, na raiz e no primeiro corte')
def gini(p): return 2*p*(1-p)
def entr(p): return 0.0 if p in (0,1) else -(p*np.log2(p)+(1-p)*np.log2(1-p))
def mcl(p): return min(p,1-p)
p0=y.mean()
print()
print('Na raiz, p = %s:'%c(p0))
print('  Gini        = 2p(1-p)          = %s'%c(gini(p0)))
print('  Entropia    = -p log2 p - ...   = %s'%c(entr(p0)))
print('  Erro        = min(p, 1-p)       = %s'%c(mcl(p0)))
t=1050.9
e=X[:,0]<=t; nE,nD=e.sum(),(~e).sum(); pE,pD=y[e].mean(),y[~e].mean()
print()
print('Corte Altitude <= %s: esquerda n=%d p=%s | direita n=%d p=%s'%(c(t,1),nE,c(pE),nD,c(pD)))
print()
print('%-12s %10s %10s %10s %12s'%('medida','esquerda','direita','média pond.','redução'))
for nm,f in [('Gini',gini),('Entropia',entr),('Erro',mcl)]:
    m=(nE*f(pE)+nD*f(pD))/NC
    print('%-12s %10s %10s %10s %12s'%(nm,c(f(pE)),c(f(pD)),c(m),c(f(p0)-m)))

print('\n'+'='*70)
print('POR QUE NÃO CRESCER PELO ERRO DE CLASSIFICAÇÃO')
print('Um corte que torna os nós mais PUROS sem mudar a classe majoritária')
print('não reduz o erro, mas reduz o Gini. Exemplo nos dados: dentro do ramo')
print('Altitude <= 1050,9 (onde quase todos são "não fino"):')
sub=e
Xs,ys=X[sub],y[sub]
best=None
for j in range(6):
    xs=np.unique(Xs[:,j]); 
    for tt in (xs[:-1]+xs[1:])/2:
        a=Xs[:,j]<=tt
        if a.sum()<10 or (~a).sum()<10: continue
        g=(a.sum()*gini(ys[a].mean())+(~a).sum()*gini(ys[~a].mean()))/len(ys)
        er=(a.sum()*mcl(ys[a].mean())+(~a).sum()*mcl(ys[~a].mean()))/len(ys)
        if best is None or g<best[0]: best=(g,er,j,tt,a.sum(),ys[a].mean(),(~a).sum(),ys[~a].mean())
g,er,j,tt,n1,p1,n2,p2=best
print('  pai: n=%d, p=%s, Gini=%s, erro=%s'%(len(ys),c(ys.mean()),c(gini(ys.mean())),c(mcl(ys.mean()))))
print('  melhor corte por Gini: %s <= %s'%(ROT[NOMES[j]],c(tt,1)))
print('    filhos: n=%d p=%s | n=%d p=%s'%(n1,c(p1),n2,c(p2)))
print('    Gini ponderado = %s (cai %s)'%(c(g),c(gini(ys.mean())-g)))
print('    erro ponderado = %s (cai %s)'%(c(er),c(mcl(ys.mean())-er)))

print('\n'+'='*70)
print('CRESCER E PODAR, com o erro validado')
cam=DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X,y)
al=np.unique(np.concatenate([[0],np.quantile(cam.ccp_alphas[cam.ccp_alphas>0],np.linspace(0,1,35))]))
res=[]
for a in al:
    er,sd=cv_clf(lambda a=a: DecisionTreeClassifier(ccp_alpha=a,random_state=0),X,y,10,0,5)
    nf=DecisionTreeClassifier(ccp_alpha=a,random_state=0).fit(X,y).get_n_leaves()
    res.append((a,er,sd,nf))
res=np.array(res); jm=int(np.argmin(res[:,1]))
print()
print('%12s %8s %14s'%('alpha','folhas','erro 10-fold'))
for i in sorted(set(list(range(0,len(res),4))+[jm,len(res)-1])):
    print('%12s %8d %14s%s'%(c(res[i,0],5),int(res[i,3]),c(res[i,1]),' <- menor erro' if i==jm else ''))
am=res[jm,0]
arv=DecisionTreeClassifier(ccp_alpha=am,random_state=0).fit(X,y)
print()
print('A árvore escolhida (%d folhas):'%arv.get_n_leaves())
print(export_text(arv,feature_names=[ROT[k] for k in NOMES],show_weights=True,decimals=1))

print('='*70)
print('A FOLHA DÁ UMA PROBABILIDADE: a proporção de finos nela')
fo=arv.apply(X); pp=arv.predict_proba(X)[:,1]
print()
print('%6s %6s %14s %20s'%('folha','n','p observada','P(fino) verdadeira'))
for f in np.unique(fo):
    i=fo==f
    print('%6d %6d %14s %20s'%(f,i.sum(),c(y[i].mean()),c(PVER[i].mean())))
print()
print('A probabilidade verdadeira varia DENTRO de cada folha; a árvore dá a')
print('todos os lotes da folha o mesmo valor. Só há %d probabilidades distintas.'%len(np.unique(pp)))

print('\n'+'='*70)
print('COMPARAÇÃO, por validação cruzada (10 dobras, 5 repetições)')
mods=[('árvore sem poda',lambda: DecisionTreeClassifier(random_state=0)),
      ('árvore de profundidade 2',lambda: DecisionTreeClassifier(max_depth=2,random_state=0)),
      ('árvore podada (alpha do menor erro)',lambda: DecisionTreeClassifier(ccp_alpha=am,random_state=0)),
      ('árvore podada, critério entropia',lambda: DecisionTreeClassifier(ccp_alpha=am,criterion='entropy',random_state=0)),
      ('regressão logística nas seis',lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1e4,max_iter=2000)))]
print()
print('%-38s %10s %10s %10s'%('modelo','erro','log-loss','Brier'))
for nm,f in mods:
    er,ll,br=cv_clf(f,X,y,10,0,5,prob=True)
    print('%-38s %10s %10s %10s'%(nm,c(er),c(ll),c(br)))
print('%-38s %10s'%('regra de Bayes (o piso)',c(ERRO_BAYES)))
print()
print('O log-loss da árvore sem poda é enorme: suas folhas são puras, e ela')
print('atribui probabilidade 0 ou 1 — quando erra, erra com certeza total.')

print('\n'+'='*70)
print('CUSTOS DESIGUAIS: perder um lote fino custa 3 vezes mais')
print('Duas formas: pesos de classe no crescimento, ou limiar na probabilidade.')
d=dobras(NC,10,0); pr=np.empty(NC); pw=np.empty(NC)
for j in range(10):
    tr,te=d!=j,d==j
    m=DecisionTreeClassifier(ccp_alpha=am,random_state=0).fit(X[tr],y[tr]); pr[te]=m.predict_proba(X[te])[:,1]
    m=DecisionTreeClassifier(ccp_alpha=am,random_state=0,class_weight={0:1,1:3}).fit(X[tr],y[tr]); pw[te]=m.predict(X[te])
def custo(pred): return float((3*np.sum((pred==0)&(y==1))+np.sum((pred==1)&(y==0)))/NC)
print()
print('%-34s %10s %10s %10s'%('regra','sensib.','especif.','custo médio'))
for nm,pred in [('limiar 0,50 (custos iguais)',(pr>=.5).astype(int)),('limiar 0,25 = 1/(1+3)',(pr>=.25).astype(int)),('pesos 1:3 no crescimento',pw.astype(int)),('regra de Bayes com custos',(PVER>=.25).astype(int))]:
    print('%-34s %10s %10s %10s'%(nm,c(np.mean(pred[y==1]==1)),c(np.mean(pred[y==0]==0)),c(custo(pred))))
