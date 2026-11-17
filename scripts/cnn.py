import numpy as np
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (dados16cnn)
# ------------------------------------------------------------------
import numpy as np
# Imagens fictícias de 16 x 16 pixels de folhas de café. Classe 1: manchas
# de ferrugem (redondas); classe 0: arranhões mecânicos (traços alongados).
# As duas marcas escurecem a folha em quantidade parecida: só a FORMA as
# distingue. 'lado' restringe as marcas à metade esquerda ou direita.
def folha(rg, classe, lado=None):
    L = 16; yy, xx = np.mgrid[0:L, 0:L]
    img = 0.62 + 0.06*np.sin(xx/5 + rg.uniform(0, 6)) + 0.04*(yy - 7.5)/7.5*rg.uniform(-1, 1)
    v = rg.integers(3, 13); img[:, v] -= 0.05                      # nervura
    lo, hi = (1, 14) if lado is None else ((1, 6) if lado == 'esq' else (9, 14))
    for _ in range(rg.integers(1, 4)):
        cy, cx = rg.uniform(2, 13), rg.uniform(lo, hi + 1)
        if classe == 1:                                          # mancha redonda
            r = rg.uniform(1.1, 1.6); d = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            img -= 0.34*np.clip(1.4 - d/r, 0, 1)
        else:                                                    # traço alongado
            ang = rg.uniform(0, np.pi); comp = rg.uniform(2.4, 3.4)
            u = (xx - cx)*np.cos(ang) + (yy - cy)*np.sin(ang)
            w = -(xx - cx)*np.sin(ang) + (yy - cy)*np.cos(ang)
            img -= 0.34*np.clip(1.4 - np.sqrt((u/comp)**2 + (w/0.55)**2), 0, 1)
    return np.clip(img + rg.normal(0, 0.07, (L, L)), 0, 1)

def conjunto(n, semente, lado=None):
    rg = np.random.default_rng(semente); y = rg.integers(0, 2, n)
    return np.array([folha(rg, k, lado) for k in y]), y

XTR, YTR = conjunto(800, 21, 'esq')      # treino: marcas só na metade ESQUERDA
XTE, YTE = conjunto(400, 22, 'esq')      # teste do mesmo tipo
XDS, YDS = conjunto(400, 23, 'dir')      # teste DESLOCADO: marcas na metade direita
# ------------------------------------------------------------------ fim do bloco comum
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
def c(v, d=4): return ('%.*f' % (d, v)).replace('.', ',')

print('IMAGENS DE FOLHAS DE CAFÉ (fictícias), 16 x 16 pixels em tons de cinza.')
print('Classe 1: manchas de ferrugem, REDONDAS. Classe 0: arranhões, ALONGADOS.')
print('As duas escurecem a folha em quantidade parecida:')
for k,nm in [(0,'arranhão'),(1,'ferrugem')]:
    print('  %-9s brilho médio %s | fração de pixels escuros (< 0,45) %s'%(nm,c(XTR[YTR==k].mean()),c((XTR[YTR==k]<.45).mean())))
print('Treino: 800 imagens com as marcas SÓ NA METADE ESQUERDA da folha.')
print('Teste: 400 do mesmo tipo, e 400 DESLOCADAS (marcas só na metade direita).')

# ------------------------------------------------------------ a convolução
def patches(I):                       # I: (n,16,16) -> (n,196,9), janelas 3x3
    n=I.shape[0]; P=np.empty((n,14,14,9))
    for a in range(3):
        for b in range(3): P[:,:,:,3*a+b]=I[:,a:a+14,b:b+14]
    return P.reshape(n,196,9)

print('\n'+'='*70)
print('UMA CONVOLUÇÃO À MÃO: um filtro 3 x 3 percorre a imagem')
F=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])/8.0
I=XTR[np.where(YTR==1)[0][0]]
r,s_=np.unravel_index(np.argmin(I[1:15,1:15]),(14,14)); r+=1; s_+=1
J=I[r-1:r+2,s_-1:s_+2]
print('Janela 3 x 3 em torno do pixel mais escuro (linha %d, coluna %d):'%(r,s_))
for lin in J: print('   '+'  '.join(c(v,3) for v in lin))
print('Filtro (detector de ponto: centro contra vizinhança), dividido por 8:')
for lin in F*8: print('   '+'  '.join('%3d'%v for v in lin))
print('Saída = soma dos produtos elemento a elemento = %s'%c(float((J*F).sum())))
print('  (centro %s menos a média dos 8 vizinhos %s)'%(c(J[1,1],3),c((J.sum()-J[1,1])/8,3)))
print('O mesmo filtro, com os mesmos 9 pesos, é aplicado em todas as 196')
print('posições: é o COMPARTILHAMENTO DE PESOS.')

# ------------------------------------------------------------ a rede
def adam_passos(params, grads, est, t, eta=0.01, b1=0.9, b2=0.999):
    for k in params:
        m,v=est.setdefault(k,(np.zeros_like(params[k]),np.zeros_like(params[k])))
        m=b1*m+(1-b1)*grads[k]; v=b2*v+(1-b2)*grads[k]**2; est[k]=(m,v)
        params[k]-=eta*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+1e-8)

def cnn(I, prm, modo, guarda=False):
    P=patches(I); A=P@prm['W']+prm['b']; R=np.maximum(A,0)          # (n,196,K)
    n,_,K=R.shape
    if modo=='global':                                               # máximo em toda a imagem
        idx=R.argmax(1); M=np.take_along_axis(R,idx[:,None,:],1)[:,0,:]
    else:                                                            # máximo em blocos 2x2 -> 7x7xK
        R4=R.reshape(n,7,2,7,2,K).transpose(0,1,3,5,2,4).reshape(n,7,7,K,4)
        idx=R4.argmax(-1); M=np.take_along_axis(R4,idx[...,None],-1)[...,0].reshape(n,-1)
    z=M@prm['v']+prm['c']; q=1/(1+np.exp(-z))
    return (q,(P,A,R,idx,M)) if guarda else q

def treina_cnn(modo, K=8, epocas=40, semente=0, lote=32):
    rg=np.random.default_rng(semente)
    d=K if modo=='global' else 49*K
    prm=dict(W=rg.normal(0,0.3,(9,K)), b=np.zeros(K), v=rg.normal(0,0.1,d), c=np.zeros(1))
    est={}; t=0; n=len(YTR)
    for ep in range(epocas):
        for ii in np.array_split(rg.permutation(n), n//lote):
            I=XTR[ii]; y=YTR[ii]; q,(P,A,R,idx,M)=cnn(I,prm,modo,True)
            g=(q-y)/len(y)                                        # dL/dz
            gr=dict(v=M.T@g, c=np.array([g.sum()]))
            gM=np.outer(g,prm['v'])
            gR=np.zeros_like(R); m_=len(y)
            if modo=='global':
                np.put_along_axis(gR,idx[:,None,:],gM[:,None,:],1)
            else:
                K_=R.shape[2]; g4=np.zeros((m_,7,7,K_,4))
                np.put_along_axis(g4,idx[...,None],gM.reshape(m_,7,7,K_)[...,None],-1)
                gR=g4.reshape(m_,7,7,K_,2,2).transpose(0,1,4,2,5,3).reshape(m_,196,K_)
            gA=gR*(A>0)
            gr['W']=np.einsum('npf,npk->fk',P,gA); gr['b']=gA.sum((0,1))
            t+=1; adam_passos(prm,gr,est,t)
    return prm

def aval(q,y): 
    qq=np.clip(q,1e-7,1-1e-7)
    return np.mean((q>=.5)!=y), float(-np.mean(y*np.log(qq)+(1-y)*np.log(1-qq)))

print('\n'+'='*70)
print('QUATRO MODELOS, e a contagem de parâmetros')
Xa,Xt,Xd=XTR.reshape(800,-1),XTE.reshape(400,-1),XDS.reshape(400,-1)
res=[]
lg=LogisticRegression(C=1.0,max_iter=5000).fit(Xa,YTR)
res.append(('regressão logística nos 256 pixels',257,lg.predict_proba(Xt)[:,1],lg.predict_proba(Xd)[:,1],1-np.mean(lg.predict(Xa)==YTR)))
ml=MLPClassifier((64,),alpha=1e-3,max_iter=2000,random_state=0).fit(Xa,YTR)
res.append(('MLP, 64 neurônios ocultos',256*64+64+65,ml.predict_proba(Xt)[:,1],ml.predict_proba(Xd)[:,1],1-np.mean(ml.predict(Xa)==YTR)))
p2=treina_cnn('blocos'); 
res.append(('CNN: 8 filtros, máx. 2x2, densa',9*8+8+49*8+1,cnn(XTE,p2,'blocos'),cnn(XDS,p2,'blocos'),aval(cnn(XTR,p2,'blocos'),YTR)[0]))
p1=treina_cnn('global')
res.append(('CNN: 8 filtros, máximo GLOBAL',9*8+8+8+1,cnn(XTE,p1,'global'),cnn(XDS,p1,'global'),aval(cnn(XTR,p1,'global'),YTR)[0]))
print()
print('%-36s %10s %10s %12s %14s'%('modelo','parâmetros','erro treino','erro teste','erro DESLOCADO'))
for nm,k,qt,qd,et in res:
    print('%-36s %10d %10s %12s %14s'%(nm,k,c(et),c(aval(qt,YTE)[0]),c(aval(qd,YDS)[0])))
print()
print('Nas imagens do mesmo tipo do treino, a CNN de máximo global é a melhor')
print('com o menor número de parâmetros. Nas DESLOCADAS, os modelos que ligam')
print('cada peso a um PIXEL não reconhecem a marca no lugar novo; o máximo')
print('global não pergunta ONDE o filtro respondeu, só SE respondeu.')

print('\n'+'='*70)
print('O QUE OS FILTROS APRENDERAM (CNN de máximo global)')
W=p1['W']; v=p1['v']
print('Peso de cada filtro na saída (positivo = evidência de ferrugem):')
print('   '+'  '.join('%7s'%c(x,2) for x in v))
for k in np.argsort(-v)[:1].tolist()+np.argsort(v)[:1].tolist():
    print('Filtro %d (peso %s):'%(k+1,c(v[k],2)))
    for lin in W[:,k].reshape(3,3): print('   '+'  '.join('%7s'%c(x,3) for x in lin))
Mt=cnn(XTE,p1,'global',True)[1][4]
print()
print('Resposta máxima média de cada filtro, por classe (teste):')
print('%8s %12s %12s'%('filtro','arranhão','ferrugem'))
for k in range(8): print('%8d %12s %12s'%(k+1,c(Mt[YTE==0,k].mean(),3),c(Mt[YTE==1,k].mean(),3)))

print('\n'+'='*70)
print('VARIABILIDADE ENTRE TREINOS: 5 sementes, CNN de máximo global')
print()
print('%8s %12s %14s'%('semente','erro teste','erro deslocado'))
for s_ in range(1,6):
    pp=treina_cnn('global',semente=s_)
    print('%8d %12s %14s'%(s_,c(aval(cnn(XTE,pp,'global'),YTE)[0]),c(aval(cnn(XDS,pp,'global'),YDS)[0])))
