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
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# BART (Chipman, George e McCulloch, 2010), implementado em numpy para
# que cada passo fique visível. No R, use o pacote BART ou dbarts.
# ------------------------------------------------------------------
class Arvore:
    """Uma árvore: nós internos (var, corte, esq, dir) e folhas (mu).
    Guarda, para cada folha, os índices dos lotes de treino e de teste."""
    def __init__(s, ntr, nte):
        s.no = {0: dict(prof=0, folha=True, mu=0.0, itr=np.arange(ntr), ite=np.arange(nte), pai=None)}
        s.prox = 1
    def folhas(s): return [k for k, v in s.no.items() if v['folha']]
    def nogs(s):   # nós internos cujos dois filhos são folhas
        return [k for k, v in s.no.items() if not v['folha'] and s.no[v['e']]['folha'] and s.no[v['d']]['folha']]

def logml(soma, n, s2, s2mu):
    """Log-verossimilhança marginal de uma folha, com mu integrado (a menos de constantes)."""
    return -0.5*np.log(1 + n*s2mu/s2) + s2mu*soma**2/(2*s2*(s2 + n*s2mu))

def bart(Xtr, ytr, Xte, m=50, iters=1500, queima=500, alfa=0.95, beta=2.0, k=2.0, nu=3, q=0.90,
         semente=0, ncortes=100, minfolha=3):
    rg = np.random.default_rng(semente)
    lo, hi = ytr.min(), ytr.max(); ys = (ytr - lo)/(hi - lo) - 0.5          # resposta em [-0,5; 0,5]
    n, p = Xtr.shape; nte = len(Xte)
    s2mu = (0.5/(k*np.sqrt(m)))**2                                            # prior das folhas
    sig_ols = np.std(ys - LinearRegression().fit(Xtr, ys).predict(Xtr), ddof=p+1)
    lam = sig_ols**2*chi2.ppf(1 - q, nu)/nu                                   # prior de sigma²
    grade = [np.unique(np.quantile(Xtr[:, j], np.linspace(0, 1, ncortes + 2)[1:-1])) for j in range(p)]
    psplit = lambda d: alfa*(1 + d)**(-beta)
    arv = [Arvore(n, nte) for _ in range(m)]
    ajtr = np.zeros((m, n)); ajte = np.zeros((m, nte)); tot = ajtr.sum(0)
    s2 = sig_ols**2
    sig_cad = []; pred = []; folhas = []; usos = np.zeros(p); aceite = [0, 0, 0, 0]
    for it in range(iters):
        for t in range(m):
            T = arv[t]; R = ys - (tot - ajtr[t])                               # resíduo parcial
            fl = T.folhas(); ng = T.nogs()
            cresce = (len(fl) == 1) or (rg.random() < 0.5)
            if cresce:
                h = fl[rg.integers(len(fl))]; v = T.no[h]; j = rg.integers(p)
                xs = Xtr[v['itr'], j]; cand = grade[j][(grade[j] >= xs.min()) & (grade[j] < xs.max())]
                if len(cand):
                    c_ = cand[rg.integers(len(cand))]; esq = xs <= c_
                    ie, idd = v['itr'][esq], v['itr'][~esq]
                    if len(ie) >= minfolha and len(idd) >= minfolha:
                        d = v['prof']; aceite[0] += 1
                        lr = (logml(R[ie].sum(), len(ie), s2, s2mu) + logml(R[idd].sum(), len(idd), s2, s2mu)
                              - logml(R[v['itr']].sum(), len(v['itr']), s2, s2mu))
                        pai_nog = (v['pai'] is not None and v['pai'] in ng)
                        w2 = len(ng) + 1 - (1 if pai_nog else 0)
                        lt = np.log((0.5)/(1.0 if len(fl) == 1 else 0.5)) + np.log(len(fl)/w2)
                        lp = np.log(psplit(d)) + 2*np.log(1 - psplit(d + 1)) - np.log(1 - psplit(d))
                        if np.log(rg.random()) < lr + lt + lp:
                            aceite[1] += 1
                            ee = Xte[v['ite'], j] <= c_
                            a, b = T.prox, T.prox + 1; T.prox += 2
                            T.no[a] = dict(prof=d+1, folha=True, mu=0.0, itr=ie, ite=v['ite'][ee], pai=h)
                            T.no[b] = dict(prof=d+1, folha=True, mu=0.0, itr=idd, ite=v['ite'][~ee], pai=h)
                            v.update(folha=False, var=j, corte=c_, e=a, d=b)
            elif ng:
                h = ng[rg.integers(len(ng))]; v = T.no[h]; a, b = T.no[v['e']], T.no[v['d']]; d = v['prof']
                aceite[2] += 1
                lr = (logml(R[v['itr']].sum(), len(v['itr']), s2, s2mu)
                      - logml(R[a['itr']].sum(), len(a['itr']), s2, s2mu) - logml(R[b['itr']].sum(), len(b['itr']), s2, s2mu))
                nfl_dep = len(fl) - 1
                pai_vira_nog = v['pai'] is not None and all(T.no[x]['folha'] or x == h for x in (T.no[v['pai']]['e'], T.no[v['pai']]['d']))
                w2_dep = len(ng) - 1 + (1 if pai_vira_nog else 0)
                lt = np.log((1.0 if nfl_dep == 1 else 0.5)/0.5) + np.log(len(ng)/nfl_dep)
                lp = np.log(1 - psplit(d)) - np.log(psplit(d)) - 2*np.log(1 - psplit(d + 1))
                if np.log(rg.random()) < lr + lt + lp:
                    aceite[3] += 1
                    del T.no[v['e']], T.no[v['d']]; v.update(folha=True, mu=0.0)
            # sorteio das médias das folhas, dada a estrutura
            ftr = np.empty(n); fte = np.empty(nte)
            for h in T.folhas():
                v = T.no[h]; nn = len(v['itr']); S = R[v['itr']].sum()
                prec = nn/s2 + 1/s2mu
                v['mu'] = rg.normal((S/s2)/prec, np.sqrt(1/prec))
                ftr[v['itr']] = v['mu']; fte[v['ite']] = v['mu']
            tot += ftr - ajtr[t]; ajtr[t] = ftr; ajte[t] = fte
        # sorteio de sigma²
        sse = np.sum((ys - tot)**2)
        s2 = 1/rg.gamma((nu + n)/2, 2/(nu*lam + sse))
        sig_cad.append(np.sqrt(s2)*(hi - lo))
        if it >= queima:
            pred.append((ajte.sum(0) + 0.5)*(hi - lo) + lo)
            folhas.append(np.mean([len(T.folhas()) for T in arv]))
            for T in arv:
                for v in T.no.values():
                    if not v['folha']: usos[v['var']] += 1
    return dict(pred=np.array(pred), sigma=np.array(sig_cad), folhas=np.array(folhas),
                usos=usos/usos.sum(), aceite=aceite, s2mu=s2mu, lam=lam, sig_ols=sig_ols*(hi - lo))

y = NOTA
rg = np.random.default_rng(5); idx = rg.permutation(NC); te, tr = idx[:200], idx[200:]
print('LOTES DE CAFÉ:', DADOS_TXT)
print('Treino: 400 lotes. Teste: 200. Verdade conhecida: sigma = %s.' % c(SIGN, 2))

print('\n' + '='*70)
print('AS PRIORIS, como ficam nestes dados (m = 50 árvores)')
r = bart(X[tr], y[tr], X[te], m=50, iters=1500, queima=500, semente=1)
print()
print('  profundidade: P(dividir um nó de profundidade d) = 0,95 (1 + d)^-2')
for d in range(4): print('     d = %d: %s' % (d, c(0.95*(1 + d)**-2.0)))
print('  folhas: mu ~ N(0, sigma_mu²), sigma_mu = 0,5/(k raiz(m)) = %s (escala [-0,5; 0,5])' % c(np.sqrt(r['s2mu'])))
print('  sigma²: qui-quadrado inversa escalada, nu = 3, ancorada no sigma do modelo')
print('          linear (%s), com P(sigma < sigma_linear) = 0,90' % c(r['sig_ols'], 3))

print('\n' + '='*70)
print('A CADEIA: 1.500 iterações, 500 de aquecimento descartadas')
s = r['sigma']
print()
print('%12s %16s' % ('iterações', 'sigma médio'))
for a, b in [(0, 10), (10, 50), (50, 200), (200, 500), (500, 1000), (1000, 1500)]:
    print('%12s %16s' % ('%d-%d' % (a + 1, b), c(float(s[a:b].mean()), 3)))
print()
print('  propostas de crescer: %d, aceitas %d (%s)' % (r['aceite'][0], r['aceite'][1], c(r['aceite'][1]/r['aceite'][0], 3)))
print('  propostas de podar:   %d, aceitas %d (%s)' % (r['aceite'][2], r['aceite'][3], c(r['aceite'][3]/r['aceite'][2], 3)))
print('  número médio de folhas por árvore (pós-aquecimento): %s' % c(float(r['folhas'].mean()), 2))

print('\n' + '='*70)
print('O QUE O BART ENTREGA: predição E incerteza')
P = r['pred']; mp = P.mean(0)
li, ls = np.quantile(P, 0.05, 0), np.quantile(P, 0.95, 0)
pos = s[500:]
print()
print('  EQM no teste (média a posteriori)     = %s' % c(eqm(y[te], mp)))
print('  EQM da função verdadeira no teste     = %s' % c(eqm(y[te], MU[te])))
print('  sigma a posteriori: média %s, intervalo 90%% [%s; %s]; verdade %s' % (
    c(float(pos.mean()), 3), c(float(np.quantile(pos, .05)), 3), c(float(np.quantile(pos, .95)), 3), c(SIGN, 3)))
cob = np.mean((MU[te] >= li) & (MU[te] <= ls))
print('  intervalos de 90%% para f(x): cobrem a f VERDADEIRA em %s dos lotes de teste' % c(cob, 3))
print('  largura média desses intervalos       = %s pontos' % c(float(np.mean(ls - li)), 3))
sp = P + rg.normal(0, 1, P.shape)*pos[:, None]
lpi, lps = np.quantile(sp, .05, 0), np.quantile(sp, .95, 0)
print('  intervalos PREDITIVOS de 90%% para y: cobrem a nota observada em %s' % c(np.mean((y[te] >= lpi) & (y[te] <= lps)), 3))

print('\n' + '='*70)
print('ONDE A INCERTEZA É MAIOR')
dist = np.abs(X[te, 0] - 1050)
print()
print('%-28s %6s %16s %14s' % ('lotes de teste', 'n', 'largura 90%', 'cobertura'))
for nm, sel in [('altitude a menos de 25 m do degrau', dist < 25), ('altitude a 25-100 m do degrau', (dist >= 25) & (dist < 100)),
                ('altitude a mais de 100 m', dist >= 100)]:
    print('%-28s %6d %16s %14s' % (nm, sel.sum(), c(float(np.mean((ls - li)[sel])), 3), c(float(np.mean(((MU[te] >= li) & (MU[te] <= ls))[sel])), 3)))

print('\n' + '='*70)
print('QUANTAS VEZES CADA VARIÁVEL É USADA NOS CORTES')
print()
for j in np.argsort(-r['usos']): print('  %-14s %s' % (ROT[NOMES[j]], c(100*r['usos'][j], 1) + '%'))

print('\n' + '='*70)
print('O NÚMERO DE ÁRVORES m')
print()
print('%6s %14s %16s %14s' % ('m', 'EQM teste', 'sigma a post.', 'cobertura 90%'))
for mm in [1, 10, 50, 200]:
    rr = r if mm == 50 else bart(X[tr], y[tr], X[te], m=mm, iters=1200 if mm < 200 else 700, queima=400 if mm < 200 else 300, semente=2)
    PP = rr['pred']; a1, a2 = np.quantile(PP, .05, 0), np.quantile(PP, .95, 0)
    q0 = len(rr['sigma']) - len(PP)
    print('%6d %14s %16s %14s' % (mm, c(eqm(y[te], PP.mean(0))), c(float(rr['sigma'][q0:].mean()), 3), c(np.mean((MU[te] >= a1) & (MU[te] <= a2)), 3)))
print()
print('Com m = 1 o BART é uma árvore bayesiana única. Com m grande, cada árvore')
print('é forçada pela prior a contribuir pouco: é a soma que modela.')
