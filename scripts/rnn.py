import numpy as np
# ------------------------------------------------------------------
# Bloco comum às fichas deste grupo: funções auxiliares e dados (dados16rnn)
# ------------------------------------------------------------------
import numpy as np
# Sequências fictícias de 30 dias de clima em talhões de café, e a ocorrência
# de SURTO de ferrugem na semana seguinte. A infecção exige vários dias
# SEGUIDOS de folha molhada (>= 8 h) com temperatura entre 19 e 25 C:
# o que importa é a MAIOR SEQUÊNCIA de dias favoráveis, não a quantidade.
T = 30
def clima(rg, n):
    chuva = np.zeros((n, T), bool); st = rg.random(n) < 0.3
    for t in range(T):                         # períodos chuvosos persistentes
        st = np.where(st, rg.random(n) < 0.72, rg.random(n) < 0.20); chuva[:, t] = st
    molh = np.clip(np.where(chuva, rg.normal(10.5, 2.5, (n, T)), rg.normal(4.0, 2.5, (n, T))), 0, 24)
    base = rg.normal(22, 2.2, n)[:, None]
    temp = base + np.cumsum(rg.normal(0, 0.6, (n, T)), 1)*0.5 - 1.2*chuva + rg.normal(0, 0.8, (n, T))
    return temp, molh

def favoravel(temp, molh): return (molh >= 8) & (temp >= 19) & (temp <= 25)
def maior_seq(f):
    r = np.zeros(f.shape[0], int); cur = np.zeros(f.shape[0], int)
    for t in range(f.shape[1]):
        cur = np.where(f[:, t], cur + 1, 0); r = np.maximum(r, cur)
    return r
def p_surto(temp, molh): return 1/(1 + np.exp(-(-4.2 + 1.05*maior_seq(favoravel(temp, molh)))))

def amostra(n, semente):
    rg = np.random.default_rng(semente); temp, molh = clima(rg, n)
    p = p_surto(temp, molh); y = (rg.random(n) < p).astype(int)
    return temp, molh, y, p
TTR, MTR, YTR, PTR = amostra(2000, 31)
TTE, MTE, YTE, PTE = amostra(1000, 32)
# ------------------------------------------------------------------ fim do bloco comum
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
def c(v, d=4): return ('%.*f' % (d, v)).replace('.', ',')
def aval(q, y):
    qq = np.clip(q, 1e-7, 1 - 1e-7)
    return float(np.mean((q >= .5) != y)), float(-np.mean(y*np.log(qq) + (1 - y)*np.log(1 - qq)))

print('CLIMA E FERRUGEM (dados fictícios): 30 dias de temperatura média (C) e')
print('horas de molhamento foliar por dia, em 3.000 talhões; resposta: surto de')
print('ferrugem na semana seguinte (1) ou não (0).')
print('Processo gerador CONHECIDO: o risco depende da MAIOR SEQUÊNCIA de dias')
print('favoráveis seguidos (molhamento >= 8 h e temperatura de 19 a 25 C).')
erro_bayes = float(np.mean(np.minimum(PTE, 1 - PTE)))
print('  treino: 2.000 talhões; teste: 1.000. Prevalência no treino = %s' % c(YTR.mean()))
print('  erro de Bayes no teste = %s' % c(erro_bayes))
f = favoravel(TTR, MTR)
print('  correlação entre o NÚMERO de dias favoráveis e a MAIOR SEQUÊNCIA = %s' % c(np.corrcoef(f.sum(1), maior_seq(f))[0, 1]))

print('\n' + '='*70)
print('DOIS TALHÕES COM O MESMO NÚMERO DE DIAS FAVORÁVEIS')
fte = favoravel(TTE, MTE); cnt = fte.sum(1); seq = maior_seq(fte)
for k in range(4, 12):
    i = np.where(cnt == k)[0]
    if len(i) > 1 and seq[i].max() - seq[i].min() >= 4: break
a, b = i[np.argmin(seq[i])], i[np.argmax(seq[i])]
for nm, j in [('A', a), ('B', b)]:
    print('  talhão %s: %d dias favoráveis, maior sequência %d, P(surto) = %s' % (nm, cnt[j], seq[j], c(PTE[j])))
    print('     ' + ''.join('#' if v else '.' for v in fte[j]))
print('Um modelo que só conta os dias não distingue A de B. A ORDEM importa.')

# ------------------------------------------------------------ preparação
mt, st_ = TTR.mean(), TTR.std(); mm, sm = MTR.mean(), MTR.std()
def entrada(Tm, Mo): return np.stack([(Tm - mt)/st_, (Mo - mm)/sm], -1)      # (n, T, 2)
Xtr, Xte = entrada(TTR, MTR), entrada(TTE, MTE)

# ------------------------------------------------------------ a rede recorrente
def rnn_frente(X, p):
    n = X.shape[0]; H = p['Wh'].shape[0]; h = np.zeros((n, H)); hs = [h]
    for t in range(X.shape[1]):
        h = np.tanh(X[:, t] @ p['Wx'] + h @ p['Wh'] + p['b']); hs.append(h)
    q = 1/(1 + np.exp(-(h @ p['v'] + p['c'])))
    return q, hs

def rnn_gradiente(X, y, p):
    q, hs = rnn_frente(X, p); n = len(y)
    g = (q - y)/n
    gr = {k: np.zeros_like(v) for k, v in p.items()}
    gr['v'] = hs[-1].T @ g; gr['c'] = np.array([g.sum()])
    dh = np.outer(g, p['v'])
    for t in range(X.shape[1] - 1, -1, -1):                  # retropropagação no tempo
        da = dh*(1 - hs[t + 1]**2)
        gr['Wx'] += X[:, t].T @ da; gr['Wh'] += hs[t].T @ da; gr['b'] += da.sum(0)
        dh = da @ p['Wh'].T
    return gr

def treina_rnn(A, H=16, epocas=150, semente=0, lote=64, eta=0.001, corte=1.0, embaralha=False, B=None, marcas=()):
    """Adam com corte da norma do gradiente (corte=None desliga)."""
    rg = np.random.default_rng(semente); d = A.shape[2]
    p = dict(Wx=rg.normal(0, 0.5, (d, H)), Wh=rg.normal(0, 0.9/np.sqrt(H), (H, H)), b=np.zeros(H),
             v=rg.normal(0, 0.3, H), c=np.zeros(1))
    est = {}; t = 0; n = len(YTR); hist = []
    for ep in range(epocas):
        for ii in np.array_split(rg.permutation(n), n//lote):
            Xb = A[ii]
            if embaralha: Xb = Xb[:, rg.permutation(T)]
            gr = rnn_gradiente(Xb, YTR[ii], p); t += 1
            if corte:
                nrm = np.sqrt(sum((g**2).sum() for g in gr.values()))
                if nrm > corte: gr = {k: g*corte/nrm for k, g in gr.items()}
            for k in p:
                m, v = est.get(k, (0, 0)); m = 0.9*m + 0.1*gr[k]; v = 0.999*v + 0.001*gr[k]**2; est[k] = (m, v)
                p[k] -= eta*(m/(1 - 0.9**t))/(np.sqrt(v/(1 - 0.999**t)) + 1e-8)
        if ep + 1 in marcas: hist.append((ep + 1, aval(rnn_frente(A, p)[0], YTR), aval(rnn_frente(B, p)[0], YTE)))
    return p, hist

print('\n' + '='*70)
print('A RECORRÊNCIA:  h_t = tanh(W_x x_t + W_h h_(t-1) + b),  q = sigma(v . h_30 + c)')
print('Os MESMOS pesos são usados em todos os 30 dias. Exemplo com H = 2 e pesos')
print('escolhidos à mão, para um talhão, nos 4 primeiros dias:')
pe = dict(Wx=np.array([[0.2, -0.1], [0.9, 0.4]]), Wh=np.array([[0.5, 0.0], [0.3, 0.6]]), b=np.zeros(2))
h = np.zeros(2)
for t in range(4):
    x = Xte[a, t]; a_ = x @ pe['Wx'] + h @ pe['Wh'] + pe['b']; h = np.tanh(a_)
    print('  dia %d: x = (%s, %s)  ->  h = (%s, %s)' % (t + 1, c(x[0], 3), c(x[1], 3), c(h[0], 3), c(h[1], 3)))
print('O estado h carrega um resumo de TUDO o que veio antes.')

print('\n' + '='*70)
print('TREINO DA RNN NOS DADOS BRUTOS (temperatura e molhamento de cada dia)')
print('H = 16, Adam, lotes de 64, retropropagação no tempo. Duas configurações:')
mk = (5, 20, 50, 100, 150)
pa, ha = treina_rnn(Xtr, eta=0.01, corte=None, B=Xte, marcas=mk)
pb, hb = treina_rnn(Xtr, eta=0.003, corte=1.0, B=Xte, marcas=mk)
print()
print('%7s %26s %26s' % ('', 'eta = 0,01, sem corte', 'eta = 0,003, corte em 1'))
print('%7s %13s %12s %13s %12s' % ('época', 'erro teste', 'log-loss', 'erro teste', 'log-loss'))
for (ep, _, ta), (_, _, tb) in zip(ha, hb): print('%7d %13s %12s %13s %12s' % (ep, c(ta[0]), c(ta[1]), c(tb[0]), c(tb[1])))
print('Passo grande sem corte: o erro oscila e não melhora. O CORTE DA NORMA')
print('do gradiente impede os saltos que a recorrência produz.')
p = pb

print('\n' + '='*70)
print('A RNN COM UM PRÉ-PROCESSAMENTO AGRONÔMICO')
print('Em vez dos valores brutos, cada dia entra como +1 (favorável) ou -1.')
print('A rede não precisa mais descobrir a faixa de 19 a 25 C: só a parte TEMPORAL.')
Ftr = favoravel(TTR, MTR)[..., None]*2.0 - 1; Fte = fte[..., None]*2.0 - 1
pf, hf = treina_rnn(Ftr, eta=0.001, corte=1.0, B=Fte, marcas=mk)
print()
print('%7s %14s %14s %14s' % ('época', 'erro treino', 'erro teste', 'log-loss teste'))
for ep, tr_, te_ in hf: print('%7d %14s %14s %14s' % (ep, c(tr_[0]), c(te_[0]), c(te_[1])))
print('  parâmetros: dados brutos %d; dia favorável %d' % (sum(v.size for v in pb.values()), sum(v.size for v in pf.values())))

print('\n' + '='*70)
print('COMPARAÇÃO NO TESTE')
res = []
def lg(A, B):
    m = LogisticRegression(C=1e4, max_iter=5000).fit(A, YTR); return m.predict_proba(B)[:, 1]
res.append(('logística nas MÉDIAS (temp., molhamento)', lg(np.column_stack([TTR.mean(1), MTR.mean(1)]), np.column_stack([TTE.mean(1), MTE.mean(1)]))))
res.append(('logística no NÚMERO de dias favoráveis', lg(f.sum(1)[:, None], fte.sum(1)[:, None])))
m = MLPClassifier((32,), alpha=1e-2, max_iter=3000, random_state=0).fit(Xtr.reshape(2000, -1), YTR)
res.append(('MLP nos 60 valores, dia a dia', m.predict_proba(Xte.reshape(1000, -1))[:, 1]))
res.append(('RNN nos dados brutos', rnn_frente(Xte, pb)[0]))
res.append(('RNN no indicador de dia favorável', rnn_frente(Fte, pf)[0]))
res.append(('logística na MAIOR SEQUÊNCIA (oráculo)', lg(maior_seq(f)[:, None], seq[:, None])))
print()
print('%-44s %10s %10s' % ('modelo', 'erro', 'log-loss'))
for nm, q in res: print('%-44s %10s %10s' % (nm, *[c(v) for v in aval(q, YTE)]))
print('%-44s %10s' % ('erro de Bayes', c(erro_bayes)))
print('O "oráculo" recebe a variável que gerou os dados; nenhum método real a')
print('teria. Ele mostra o quanto ainda é possível ganhar.')

print('\n' + '='*70)
print('A RNN USA A ORDEM? Teste com os dias EMBARALHADOS (RNN do dia favorável)')
rg = np.random.default_rng(9); ordem = rg.permutation(T)
print('  erro no teste, ordem verdadeira       = %s' % c(aval(rnn_frente(Fte, pf)[0], YTE)[0]))
print('  erro no teste, dias embaralhados      = %s' % c(aval(rnn_frente(Fte[:, ordem], pf)[0], YTE)[0]))
print('E se a rede for TREINADA com os dias embaralhados? Só pode aprender a contar:')
p2, _ = treina_rnn(Ftr, eta=0.001, corte=1.0, embaralha=True)
print('  erro no teste = %s | log-loss = %s' % tuple(c(v) for v in aval(rnn_frente(Fte, p2)[0], YTE)))
print('  (compare com a logística no NÚMERO de dias favoráveis, acima)')

print('\n' + '='*70)
print('O GRADIENTE QUE SE DESVANECE')
print('Sensibilidade da saída a uma pequena mudança no molhamento de cada dia,')
print('média sobre os talhões de teste (diferenças finitas), na rede dos dados')
print('brutos ANTES e DEPOIS do treino:')
p0, _ = treina_rnn(Xtr, epocas=0, semente=0)            # pesos iniciais, sem treino
def sensib(pp):
    q0 = rnn_frente(Xte, pp)[0]; out = []
    for t in range(T):
        Z = Xte.copy(); Z[:, t, 1] += 0.01; out.append(float(np.mean(np.abs(rnn_frente(Z, pp)[0] - q0))/0.01))
    return np.array(out)
s0, s1 = sensib(p0), sensib(pb)
print()
print('%6s %16s %16s' % ('dia', 'antes do treino', 'depois'))
for t in [0, 4, 9, 14, 19, 24, 27, 29]: print('%6d %16s %16s' % (t + 1, c(s0[t], 5), c(s1[t], 5)))
print('  razão dia 30 / dia 1: antes %s ; depois %s' % (c(s0[-1]/s0[0], 1), c(s1[-1]/s1[0], 2)))
rho = lambda W: float(np.max(np.abs(np.linalg.eigvals(W))))
print('  raio espectral de W_h: antes %s ; depois %s' % (c(rho(p0['Wh']), 3), c(rho(pb['Wh']), 3)))
print('Na partida, a influência de um dia cai por um fator a cada passo, e o')
print('dia 1 quase não chega à saída: é o gradiente que se desvanece. O treino')
print('aumentou W_h e aprendeu a reter a memória, o que é possível em 30 passos.')
print('Em sequências de centenas de passos a RNN simples não consegue; as')
print('variantes LSTM e GRU usam portões para preservar a memória.')

print('\n' + '='*70)
print('VARIABILIDADE ENTRE TREINOS: 4 sementes, RNN do dia favorável')
for s_ in range(1, 5):
    pp, _ = treina_rnn(Ftr, eta=0.001, corte=1.0, semente=s_)
    print('  semente %d: erro teste %s | log-loss %s' % (s_, *[c(v) for v in aval(rnn_frente(Fte, pp)[0], YTE)]))
