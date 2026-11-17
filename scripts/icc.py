# Bibliotecas: numpy e scipy (ambas já instaladas no Google Colab)
import numpy as np
from scipy import stats

# ===================================================================
# Dados fictícios, criados pelo autor. As mesmas 40 folhas de cafeeiro
# das demais fichas de concordância, agora com a severidade medida em
# ESCALA CONTÍNUA: porcentagem de área foliar lesionada, estimada por
# quatro avaliadores.
# ===================================================================
AREA = np.array([
 [ 1.8, 8.3,26.3,11.1],[32.4,20.9,32.8,26.8],[37.8,34.4,39.3,36.4],
 [34.1,34.5,40.7,38.5],[31.6,32.0,41.3,31.0],[ 2.2, 0.1,11.7, 0.2],
 [20.8,10.2,32.3,28.5],[ 0.3, 8.2,23.8, 6.2],[13.1, 8.0,28.8,33.8],
 [24.0,22.3,29.6,17.4],[49.6,38.6,56.5,52.8],[20.3,20.0,27.4,25.8],
 [ 3.3, 0.1, 5.0, 0.1],[20.3,15.7,28.8,19.8],[10.0, 9.1,13.5, 0.1],
 [33.0,25.5,39.9,29.9],[27.9,22.2,49.3,42.6],[28.1,21.6,35.1,22.1],
 [13.3, 7.8,19.7,15.9],[31.9,10.0,37.0,29.6],[ 6.2,11.0,22.9,10.6],
 [ 3.6, 0.1,26.4,24.1],[22.3, 9.8,26.1,18.6],[ 4.7, 4.3,22.0, 7.3],
 [ 0.1, 0.1,17.1, 4.7],[ 2.5, 8.5,14.5, 0.5],[ 9.2,18.4,12.8,23.6],
 [12.6, 1.5,13.4,16.8],[37.4,30.3,57.1,38.9],[ 3.5, 0.1,10.0, 4.2],
 [21.5,23.1,39.7,32.5],[10.8,10.1,19.4, 2.1],[ 9.5, 0.1,10.4, 8.2],
 [ 0.1, 6.7, 5.5, 0.1],[14.7,11.3,28.4,18.9],[19.2,10.2,25.1, 8.6],
 [19.7,10.0,15.5,27.9],[ 1.3, 0.1,11.3,12.7],[36.4,38.8,57.4,58.7],
 [13.9,12.6,20.1,30.7]])
AV = ["A", "B", "C", "D"]

def quadrados_medios(Y):
    """Decomposição da ANOVA de duas vias sem repetição."""
    Y = np.asarray(Y, float); n, k = Y.shape; gm = Y.mean()
    MSR = k*((Y.mean(1)-gm)**2).sum()/(n-1)                       # entre unidades
    MSC = n*((Y.mean(0)-gm)**2).sum()/(k-1)                       # entre avaliadores
    MSE = ((Y - Y.mean(1)[:,None] - Y.mean(0)[None,:] + gm)**2).sum()/((n-1)*(k-1))
    MSW = ((Y - Y.mean(1)[:,None])**2).sum()/(n*(k-1))            # dentro (uma via)
    return n, k, MSR, MSC, MSE, MSW

def todos_icc(Y):
    n, k, MSR, MSC, MSE, MSW = quadrados_medios(Y)
    return {
      "ICC(1,1)": (MSR-MSW)/(MSR+(k-1)*MSW),
      "ICC(2,1)": (MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n),
      "ICC(3,1)": (MSR-MSE)/(MSR+(k-1)*MSE),
      "ICC(1,k)": (MSR-MSW)/MSR,
      "ICC(2,k)": (MSR-MSE)/(MSR+(MSC-MSE)/n),
      "ICC(3,k)": (MSR-MSE)/MSR}

def ic_icc31(Y, alfa=0.05):
    """IC exato de ICC(3,1), baseado na razão F = MSR/MSE."""
    n, k, MSR, MSC, MSE, MSW = quadrados_medios(Y)
    F = MSR/MSE
    fl = F/stats.f.ppf(1-alfa/2, n-1, (n-1)*(k-1))
    fu = F*stats.f.ppf(1-alfa/2, (n-1)*(k-1), n-1)
    return (fl-1)/(fl+k-1), (fu-1)/(fu+k-1)

n, k, MSR, MSC, MSE, MSW = quadrados_medios(AREA)
print(f"{n} folhas, {k} avaliadores, resposta contínua (% de área lesionada)")
print(f"\nMédia por avaliador: " + ", ".join(
      f"{AV[j]} {AREA[:,j].mean():.2f}%" for j in range(k)))
print(f"Desvio-padrão:       " + ", ".join(
      f"{AV[j]} {AREA[:,j].std(ddof=1):.2f}" for j in range(k)))
print(f"Média geral = {AREA.mean():.3f}%")
print("O avaliador C superestima sistematicamente: a média dele é quase o")
print("dobro da do B. Essa diferença é exatamente o que separa as formas de")
print("ICC umas das outras.")

# ===================================================================
# PARTE 1 - a tabela de ANOVA que está por trás
# ===================================================================
print("\n" + "=" * 70)
print("ANOVA de duas vias sem repetição: folhas (linhas) x avaliadores (colunas)")
SQR = MSR*(n-1); SQC = MSC*(k-1); SQE = MSE*(n-1)*(k-1)
print(f"\n{'fonte':<22}{'gl':>6}{'soma de quadrados':>20}{'quadrado médio':>18}")
print(f"{'entre folhas':<22}{n-1:>6}{SQR:>20.2f}{MSR:>18.4f}")
print(f"{'entre avaliadores':<22}{k-1:>6}{SQC:>20.2f}{MSC:>18.4f}")
print(f"{'resíduo':<22}{(n-1)*(k-1):>6}{SQE:>20.2f}{MSE:>18.4f}")
print(f"{'total':<22}{n*k-1:>6}{SQR+SQC+SQE:>20.2f}")
print(f"\nDentro das folhas, sem separar avaliadores: MSW = {MSW:.4f}")
print(f"  (MSW junta o efeito de avaliador e o resíduo: "
      f"{(SQC+SQE)/(n*(k-1)):.4f})")
F = MSC/MSE
print(f"\nTeste do efeito de avaliador: F = MSC/MSE = {MSC:.2f}/{MSE:.2f} = "
      f"{F:.3f}")
print(f"  gl = {k-1} e {(n-1)*(k-1)} | p = {stats.f.sf(F, k-1, (n-1)*(k-1)):.3e}")
print("  Há diferença sistemática entre avaliadores, e é isso que faz")
print("  ICC(2,1) e ICC(3,1) divergirem.")

# ===================================================================
# PARTE 2 - as seis formas
# ===================================================================
print("\n" + "=" * 70)
print("As seis formas de ICC de Shrout e Fleiss (1979)")
icc = todos_icc(AREA)
print(f"\n{'forma':<11}{'modelo':<24}{'o que mede':<36}{'valor':>8}")
desc = [("ICC(1,1)","uma via, aleatório","avaliadores diferentes por unidade"),
        ("ICC(2,1)","duas vias, aleatório","concordância ABSOLUTA, 1 avaliador"),
        ("ICC(3,1)","duas vias, misto","CONSISTÊNCIA, 1 avaliador"),
        ("ICC(1,k)","uma via, aleatório","média de k avaliadores diferentes"),
        ("ICC(2,k)","duas vias, aleatório","concordância absoluta, média de k"),
        ("ICC(3,k)","duas vias, misto","consistência, média de k")]
for nome, mod, oq in desc:
    print(f"{nome:<11}{mod:<24}{oq:<36}{icc[nome]:>8.4f}")
lo, hi = ic_icc31(AREA)
print(f"\nIC 95% exato de ICC(3,1): [{lo:.4f}; {hi:.4f}]")
print(f"\nDiferença entre ICC(3,1) e ICC(2,1) = "
      f"{icc['ICC(3,1)']-icc['ICC(2,1)']:.4f}")
print("Essa diferença é o preço do viés dos avaliadores. ICC(3,1) pergunta se")
print("eles ORDENAM as folhas do mesmo jeito; ICC(2,1) pergunta se dão o MESMO")
print("NÚMERO. Como o avaliador C soma cerca de 10 pontos percentuais a tudo,")
print("ele ordena bem mas não acerta o valor, e só o ICC(2,1) o penaliza.")

print("\nDemonstração: somando 10 a TODAS as medidas do avaliador C")
Y2 = AREA.copy(); Y2[:,2] = AREA[:,2] + 10
i2 = todos_icc(Y2)
print(f"{'forma':<12}{'original':>12}{'com +10 em C':>16}{'variação':>12}")
for nome in ["ICC(1,1)","ICC(2,1)","ICC(3,1)"]:
    print(f"{nome:<12}{icc[nome]:>12.4f}{i2[nome]:>16.4f}{i2[nome]-icc[nome]:>+12.4f}")
print("ICC(3,1) não se move uma casa decimal: um deslocamento constante não")
print("altera a ordenação. ICC(2,1) e ICC(1,1) caem, e devem cair mesmo.")

# ===================================================================
# PARTE 3 - a escolha da forma, em três perguntas
# ===================================================================
print("\n" + "=" * 70)
print("A escolha da forma responde a três perguntas, nesta ordem:")
print("\n  1. Todas as unidades foram avaliadas pelos MESMOS avaliadores?")
print("     Não -> modelo de uma via, ICC(1,*).")
print("     Sim -> pergunta 2.")
print("  2. Os avaliadores são uma amostra de uma população maior, à qual")
print("     se quer generalizar?")
print("     Sim -> modelo de duas vias ALEATÓRIO, ICC(2,*).")
print("     Não, são estes e só estes -> modelo MISTO, ICC(3,*).")
print("  3. A medida que se vai usar na prática é de UM avaliador ou a MÉDIA")
print("     de k avaliadores?")
print("     Um -> ICC(*,1).   Média de k -> ICC(*,k).")
print("\nE uma quarta, transversal: interessa a concordância ABSOLUTA ou")
print("apenas a CONSISTÊNCIA? Absoluta -> ICC(2,*). Consistência -> ICC(3,*).")

print("\nPor que a média de k avaliadores é sempre mais confiável:")
print(f"{'k':>4}{'ICC(2,k) previsto por Spearman-Brown':>42}")
r1 = icc["ICC(2,1)"]
for kk in [1, 2, 3, 4, 6, 10]:
    print(f"{kk:>4}{kk*r1/(1+(kk-1)*r1):>42.4f}")
print(f"O valor observado com k = 4 é {icc['ICC(2,k)']:.4f} e a previsão da fórmula")
print(f"de Spearman-Brown é {4*r1/(1+3*r1):.4f}: são a mesma conta, escrita de dois modos.")
print("Médias de avaliadores são mais confiáveis que avaliadores isolados, e a")
print("fórmula diz exatamente quanto se ganha com cada avaliador a mais.")

# ===================================================================
# PARTE 4 - erro-padrão da medida e diferença mínima detectável
# ===================================================================
print("\n" + "=" * 70)
print("O ICC é adimensional, e por isso não basta. Duas quantidades na")
print("unidade original completam o quadro.")
EPM = np.sqrt(MSE)
DMD = 1.96*np.sqrt(2)*EPM
print(f"\n  Erro-padrão da medida: EPM = raiz(MSE) = {EPM:.3f} pontos percentuais")
print(f"  Diferença mínima detectável: DMD = 1,96 x raiz(2) x EPM = "
      f"{DMD:.3f} p.p.")
print(f"  Amplitude observada das médias por folha: "
      f"{AREA.mean(1).min():.1f}% a {AREA.mean(1).max():.1f}%")
print("Duas leituras da mesma folha por avaliadores diferentes podem diferir")
print(f"em até {DMD:.1f} pontos percentuais só por erro de medida. Em um ensaio que")
print("compare tratamentos separados por menos que isso, o método de avaliação")
print("não serve - por mais alto que seja o ICC.")

print("\nLimites de concordância de Bland-Altman, para os pares de avaliadores")
print(f"{'par':>6}{'viés médio':>13}{'DP das diferenças':>20}{'limites de 95%':>26}")
for i in range(k):
    for j in range(i+1, k):
        d = AREA[:,i] - AREA[:,j]
        print(f"{AV[i]+'-'+AV[j]:>6}{d.mean():>13.3f}{d.std(ddof=1):>20.3f}"
              f"   [{d.mean()-1.96*d.std(ddof=1):7.3f}; "
              f"{d.mean()+1.96*d.std(ddof=1):7.3f}]")
print("O par B-C tem viés de −12,9 pontos: o C mede sistematicamente mais.")
print("O ICC sozinho esconde isso; o gráfico de Bland-Altman o expõe.")

# ===================================================================
# PARTE 5 - o ICC depende da HETEROGENEIDADE das unidades
# ===================================================================
print("\n" + "=" * 70)
print("O ICC não é uma propriedade só do instrumento: depende também de quão")
print("diferentes são as unidades medidas. Restringindo a amostra a folhas")
print("cada vez mais parecidas entre si, com o MESMO erro de medida:")
med = AREA.mean(1)
print(f"\n{'subconjunto':<34}{'n':>4}{'DP das médias':>16}{'ICC(2,1)':>11}{'EPM':>9}")
for rot, mask in [("todas as 40 folhas", np.ones(n, bool)),
                  ("severidade média abaixo de 35%", med < 35),
                  ("severidade média abaixo de 25%", med < 25),
                  ("severidade média abaixo de 15%", med < 15),
                  ("severidade média entre 5% e 20%", (med >= 5) & (med < 20))]:
    Ys = AREA[mask]
    if len(Ys) < 8: continue
    r = todos_icc(Ys); _, _, _, _, mse_, _ = quadrados_medios(Ys)
    print(f"{rot:<34}{len(Ys):>4}{Ys.mean(1).std(ddof=1):>16.3f}"
          f"{r['ICC(2,1)']:>11.4f}{np.sqrt(mse_):>9.3f}")
print("\nO erro-padrão da medida mal se move - o instrumento é o mesmo - mas o")
print("ICC despenca de 0,71 para 0,13. É o mesmo fenômeno da")
print("correlação restrita a uma faixa estreita: o numerador encolhe e o")
print("denominador não. Por isso um ICC obtido em uma população heterogênea")
print("não se transfere para uma população homogênea, e por isso o EPM, que")
print("não depende da heterogeneidade, deve ser reportado junto.")
