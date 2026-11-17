# Bibliotecas: numpy, scipy e matplotlib (todas já no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# Os dados de Freireich et al. (1963): 42 pacientes com leucemia,
# pareados, metade recebendo 6-mercaptopurina e metade placebo.
# O desfecho é o tempo até a recaída, em semanas. O sinal + marca
# observações CENSURADAS (o paciente ainda estava em remissão no fim
# do acompanhamento). Os dados acompanham o pacote survival do R.
# ===================================================================
t6  = np.array([6,6,6,6,7,9,10,10,11,13,16,17,19,20,22,23,25,32,32,34,35])
e6  = np.array([0,1,1,1,1,0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
tpl = np.array([1,1,2,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23])
epl = np.ones(21, int)

def km(t, e):
    """Estimador de Kaplan-Meier com erro-padrão de Greenwood."""
    t = np.asarray(t, float); e = np.asarray(e, int)
    tempos = np.unique(t[e == 1])                    # só os tempos de evento
    n_risco = np.array([np.sum(t >= x) for x in tempos])
    d = np.array([np.sum((t == x) & (e == 1)) for x in tempos])
    S = np.cumprod(1 - d / n_risco)
    # Greenwood: Var(S) = S² * soma d / (n (n - d))
    soma = np.cumsum(d / (n_risco * (n_risco - d)))
    ep = S * np.sqrt(soma)
    return tempos, n_risco, d, S, ep, soma

def ic_log(S, soma, alfa=0.05):
    """IC pela transformação log, que é o padrão de survfit() no R."""
    z = stats.norm.ppf(1 - alfa/2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.sqrt(soma)
        lo = np.clip(S*np.exp(-z*sig), 0, 1); hi = np.clip(S*np.exp(+z*sig), 0, 1)
    return np.nan_to_num(lo), np.nan_to_num(hi)

def ic_loglog(S, soma, alfa=0.05):
    """IC pela transformação log(-log S); é a opção conf.type = 'log-log'."""
    z = stats.norm.ppf(1 - alfa/2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ep_ll = np.sqrt(soma) / np.abs(np.log(S))
        lo = S ** np.exp(+z * ep_ll); hi = S ** np.exp(-z * ep_ll)
    return np.nan_to_num(lo), np.nan_to_num(hi)

def mediana(tempos, S):
    idx = np.where(S <= 0.5)[0]
    return float(tempos[idx[0]]) if len(idx) else np.nan

def ic_mediana(tempos, lo, hi):
    """Limites do IC da mediana: o primeiro tempo em que cada banda cruza 0,5."""
    a = np.where(lo <= 0.5)[0]; b = np.where(hi <= 0.5)[0]
    return (float(tempos[a[0]]) if len(a) else np.nan,
            float(tempos[b[0]]) if len(b) else np.nan)

def rmst(t, e, tau):
    """Tempo médio restrito de sobrevivência: área sob a curva até tau."""
    tempos, nr, d, S, ep, soma = km(t, e)
    x = np.r_[0.0, tempos[tempos <= tau], tau]
    y = np.r_[1.0, S[tempos <= tau]]
    return float(np.sum(np.diff(x) * y))

print("Freireich et al. (1963): tempo até a recaída, em semanas")
print(f"  6-MP:    n = {len(t6)}, eventos = {e6.sum()}, "
      f"censuras = {len(t6)-e6.sum()} ({100*(len(t6)-e6.sum())/len(t6):.1f}%)")
print(f"  placebo: n = {len(tpl)}, eventos = {epl.sum()}, censuras = 0")

# ===================================================================
# PARTE 1 - a tabela de vida do grupo 6-MP, linha a linha
# ===================================================================
tempos, nr, d, S, ep, soma = km(t6, e6)
lo, hi = ic_log(S, soma)
print("\nTabela de Kaplan-Meier, grupo 6-MP")
print("(a coluna 'censuras' conta as saídas ocorridas DESDE o tempo da linha")
print(" anterior, inclusive as do próprio tempo da linha)")
print(f"{'tempo':>7}{'em risco':>10}{'eventos':>9}{'censuras':>10}"
      f"{'1 - d/n':>10}{'S(t)':>10}{'EP':>9}{'IC 95%':>20}")
ant = -1.0
for j, x in enumerate(tempos):
    cj = int(np.sum((t6 > ant) & (t6 <= x) & (e6 == 0)))
    print(f"{x:>7.0f}{nr[j]:>10d}{d[j]:>9d}{cj:>10d}"
          f"{1-d[j]/nr[j]:>10.4f}{S[j]:>10.4f}{ep[j]:>9.4f}"
          f"   [{lo[j]:.4f}; {hi[j]:.4f}]")
    ant = x
resto = int(np.sum((t6 > tempos[-1]) & (e6 == 0)))
print(f"Depois do último evento (t = {tempos[-1]:.0f}) restam {resto} censuras, que não")
print("geram linha: a curva permanece em S = {:.4f} e simplesmente termina ali.".format(S[-1]))
print("As censuras não produzem degrau na curva, mas REDUZEM o número em risco")
print("das linhas seguintes, e é assim que elas entram na conta.")

tp, nrp, dp, Sp_, epp, somap = km(tpl, epl)
lop, hip = ic_log(Sp_, somap)
print("\nTabela de Kaplan-Meier, grupo placebo (sem censuras)")
print(f"{'tempo':>7}{'em risco':>10}{'eventos':>9}{'S(t)':>10}{'EP':>9}{'IC 95%':>20}")
for j, x in enumerate(tp):
    e_ = "-" if not np.isfinite(epp[j]) else f"{epp[j]:.4f}"
    print(f"{x:>7.0f}{nrp[j]:>10d}{dp[j]:>9d}{Sp_[j]:>10.4f}{e_:>9}"
          f"   [{lop[j]:.4f}; {hip[j]:.4f}]")
print("Sem censuras, o estimador coincide com a proporção simples de sobreviventes:")
print(f"em t = 8, por exemplo, S = {Sp_[tp==8][0]:.4f} = 8/21 = {8/21:.4f}.")

# ===================================================================
# PARTE 2 - as quantidades que se reportam
# ===================================================================
print("\n" + "=" * 70)
m6 = mediana(tempos, S); mpl = mediana(tp, Sp_)
i6 = ic_mediana(tempos, lo, hi); ipl = ic_mediana(tp, lop, hip)
l6b, h6b = ic_loglog(S, soma); lpb, hpb = ic_loglog(Sp_, somap)
i6b = ic_mediana(tempos, l6b, h6b); iplb = ic_mediana(tp, lpb, hpb)
print("Mediana de sobrevivência (primeiro tempo em que S(t) <= 0,5)")
print(f"  6-MP:    {m6:.0f} semanas   IC 95% [{i6[0]:.0f}; "
      f"{'não atingido' if np.isnan(i6[1]) else format(i6[1],'.0f')}]")
print(f"  placebo: {mpl:.0f} semanas    IC 95% [{ipl[0]:.0f}; {ipl[1]:.0f}]")
print(f"  Razão das medianas = {m6/mpl:.2f}")
print("  (IC pela transformação log, que é o padrão de survfit() no R; com a")
print(f"   transformação log-log sairia 6-MP [{i6b[0]:.0f}; "
      f"{'não atingido' if np.isnan(i6b[1]) else format(i6b[1],'.0f')}] e placebo "
      f"[{iplb[0]:.0f}; {iplb[1]:.0f}])")

for q in [0.75, 0.50, 0.25]:
    a = tempos[S <= q]; b = tp[Sp_ <= q]
    print(f"  tempo em que S(t) <= {q:.2f}:  6-MP "
          f"{(str(int(a[0])) + ' sem') if len(a) else 'não atingido':>14}"
          f"   | placebo {int(b[0]):>3d} sem")

print("\nProbabilidade de ainda estar em remissão")
for x in [5, 10, 15, 20, 23]:
    s6 = S[tempos <= x][-1] if np.any(tempos <= x) else 1.0
    sp = Sp_[tp <= x][-1] if np.any(tp <= x) else 1.0
    print(f"  em {x:>2d} semanas:  6-MP {100*s6:6.2f}%   placebo {100*sp:6.2f}%")

tau = 23.0
print(f"\nTempo médio restrito de sobrevivência até tau = {tau:.0f} semanas")
print("(a área sob a curva; é a medida a usar quando a mediana não é atingida)")
r6 = rmst(t6, e6, tau); rpl = rmst(tpl, epl, tau)
print(f"  6-MP:    {r6:.3f} semanas")
print(f"  placebo: {rpl:.3f} semanas")
print(f"  diferença = {r6-rpl:.3f} semanas em favor do 6-MP")
print("  tau = 23 porque é o último tempo observado no placebo: comparar além")
print("  dele exigiria extrapolar a curva que termina primeiro.")

# ===================================================================
# PARTE 3 - o que acontece quando a censura é tratada errado
# ===================================================================
print("\n" + "=" * 70)
print("Três maneiras ERRADAS de lidar com a censura, no grupo 6-MP")
print(f"\n{'procedimento':<42}{'mediana':>12}{'média':>12}")
print(f"{'Kaplan-Meier (correto)':<42}{m6:>10.0f} sem{'não estimável':>14}")
print(f"{'descartar os 12 censurados':<42}"
      f"{np.median(t6[e6==1]):>10.1f} sem{np.mean(t6[e6==1]):>10.2f} sem")
print(f"{'tratar censura como recaída':<42}"
      f"{np.median(t6):>10.1f} sem{np.mean(t6):>10.2f} sem")
print(f"{'tratar censura como acompanhamento completo':<42}"
      f"{'nunca recai':>14}{'-':>12}")
print("\nDescartar os 12 censurados dá mediana de 10 semanas, menos da metade das")
print("23 corretas: os pacientes descartados são justamente os que foram melhor.")
print("Tratar a censura como recaída dá 16 semanas e subestima da mesma forma,")
print("porque supõe que todos recaíram no dia em que saíram do estudo. A média")
print("aritmética não é estimável pelo Kaplan-Meier aqui, porque a curva termina")
print("em S = 0,4482 e não chega a zero; é para isso que serve o tempo médio")
print("restrito até um tau declarado. O Kaplan-Meier")
print("usa a informação parcial de cada censurado - 'sobreviveu ao menos até t' -")
print("sem inventar o que veio depois.")

# ===================================================================
# PARTE 4 - a censura precisa ser NÃO INFORMATIVA
# ===================================================================
print("\n" + "=" * 70)
print("O pressuposto central: a censura deve ser NÃO INFORMATIVA, isto é,")
print("quem sai do estudo em t deve ter o mesmo risco de quem fica.")
print("Simulação: n = 300, tempos exponenciais com média 10, acompanhamento")
print("até 15. Quatro mecanismos de censura, 2.000 repetições cada.")
rng = np.random.default_rng(17)
verdade = 1 - np.exp(-10/10)        # P(T <= 10) com média 10
print(f"\nValor verdadeiro de S(10) = {np.exp(-1):.4f}")
print(f"{'mecanismo de censura':<46}{'S(10) estimado':>16}{'viés':>10}")
for nome, modo in [("administrativa (fim do estudo em t = 15)", "adm"),
                   ("aleatória, independente do risco", "ind"),
                   ("INFORMATIVA: abandona cedo quem iria bem", "bom"),
                   ("INFORMATIVA: é retirado quem iria mal", "mal")]:
    est = []
    for _ in range(2000):
        T = rng.exponential(10, 300)
        if modo == "adm":
            C = np.full(300, 15.0)
        elif modo == "ind":
            C = rng.exponential(15, 300)
        elif modo == "bom":
            # metade dos que teriam T grande abandona cedo o estudo
            C = np.where((rng.random(300) < 0.5) & (T > 8),
                         rng.uniform(0.5, 5.0, 300), 99.0)
        else:
            # metade dos que teriam T pequeno é retirada antes do evento
            C = np.where((rng.random(300) < 0.5) & (T < 8),
                         T * rng.uniform(0.3, 0.9, 300), 99.0)
        X = np.minimum(T, C); E = (T <= C).astype(int)
        tt, nn_, dd, SS, _, _ = km(X, E)
        est.append(SS[tt <= 10][-1] if np.any(tt <= 10) else 1.0)
    m = np.mean(est)
    print(f"{nome:<46}{m:>16.4f}{m-np.exp(-1):>10.4f}")
print("\nOs dois primeiros mecanismos são não informativos e o estimador é")
print("praticamente não viesado: o viés fica em um milésimo, que é ruído de")
print("simulação. Os dois últimos ligam a saída ao prognóstico, e o viés aparece")
print("com sinal oposto em cada um - para BAIXO quando abandonam cedo os que")
print("iriam bem (0,2651 contra 0,3679), para CIMA quando são retirados os que")
print("iriam mal (0,5339). A magnitude chega a 45% do valor verdadeiro.")
print("Nenhum teste detecta isso nos dados, porque o que teria acontecido depois")
print("da censura não foi observado: é uma suposição sobre o DELINEAMENTO, que")
print("precisa ser defendida com argumento substantivo, não com estatística.")
