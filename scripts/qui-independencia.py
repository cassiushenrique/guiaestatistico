# Bibliotecas: numpy, scipy e matplotlib (já instaladas no Google Colab)
import numpy as np
from scipy import stats

# Dados: UCBAdmissions (R). Admissões à pós-graduação da Universidade da Califórnia
# em Berkeley, outono de 1973, nos seis maiores departamentos.
# Fonte: Bickel, Hammel e O'Connell (1975), Science 187:398-404.
dep = ["A", "B", "C", "D", "E", "F"]
# tabela[d] = [[homens admitidos, homens rejeitados], [mulheres admitidas, mulheres rejeitadas]]
tabela = {"A": [[512, 313], [ 89,  19]], "B": [[353, 207], [ 17,   8]],
          "C": [[120, 205], [202, 391]], "D": [[138, 279], [131, 244]],
          "E": [[ 53, 138], [ 94, 299]], "F": [[ 22, 351], [ 24, 317]]}
T = sum(np.array(tabela[d], float) for d in dep)
sexo = ["homens", "mulheres"]; sit = ["admitidos", "rejeitados"]

print("Tabela agregada dos seis departamentos")
print(f"{'':<10}{'admitidos':>11}{'rejeitados':>12}{'total':>9}{'taxa':>9}")
for i in range(2):
    print(f"{sexo[i]:<10}{int(T[i,0]):11d}{int(T[i,1]):12d}{int(T[i].sum()):9d}"
          f"{100*T[i,0]/T[i].sum():8.2f}%")
print(f"{'total':<10}{int(T[:,0].sum()):11d}{int(T[:,1].sum()):12d}{int(T.sum()):9d}"
      f"{100*T[:,0].sum()/T.sum():8.2f}%")

# 1. O teste
qui, p, gl, esp = stats.chi2_contingency(T, correction=False)
quiy, py, _, _ = stats.chi2_contingency(T, correction=True)
print("\nFrequências esperadas sob independência (linha x coluna / total)")
for i in range(2):
    print(f"{sexo[i]:<10}" + "".join(f"{esp[i,j]:11.2f}" for j in range(2)))
print(f"\nqui-quadrado sem correção = {qui:.4f} | gl = {gl} | p-valor = {p:.3e}")
print(f"qui-quadrado com correção de Yates = {quiy:.4f} | p-valor = {py:.3e}")
print(f"(o R aplica Yates por padrão em tabelas 2x2; use correct = FALSE para desligar)")
print(f"Menor frequência esperada = {esp.min():.2f}")

print("\nContribuição de cada célula e resíduos padronizados")
print(f"{'célula':<24}{'O':>7}{'E':>10}{'(O-E)²/E':>11}{'resíduo':>10}")
for i in range(2):
    for j in range(2):
        r = (T[i,j]-esp[i,j])/np.sqrt(esp[i,j])
        print(f"{sexo[i]+' '+sit[j]:<24}{int(T[i,j]):7d}{esp[i,j]:10.2f}"
              f"{(T[i,j]-esp[i,j])**2/esp[i,j]:11.4f}{r:10.3f}")

# 2. Medidas de associação
n = T.sum()
V = np.sqrt(qui / (n * (min(T.shape) - 1)))
orr = T[0,0]*T[1,1] / (T[0,1]*T[1,0])
ep_lor = np.sqrt(sum(1/T[i,j] for i in range(2) for j in range(2)))
print(f"\nV de Cramér = {V:.4f}  (0,1 pequeno; 0,3 médio; 0,5 grande)")
print(f"phi = {np.sqrt(qui/n):.4f} (em tabelas 2x2 coincide com V)")
print(f"Razão de chances = {orr:.4f} | IC 95% = "
      f"[{np.exp(np.log(orr)-1.96*ep_lor):.4f}; {np.exp(np.log(orr)+1.96*ep_lor):.4f}]")
print(f"Diferença de proporções = {100*(T[0,0]/T[0].sum() - T[1,0]/T[1].sum()):.2f} pontos percentuais")
print(f"Risco relativo = {(T[0,0]/T[0].sum())/(T[1,0]/T[1].sum()):.4f}")
print("\nTeste exato de Fisher, para comparação:")
od, pf = stats.fisher_exact(T.astype(int))
print(f"  razão de chances condicional = {od:.4f} | p-valor = {pf:.3e}")
print("Com n = 4526 o exato e a aproximação coincidem; o exato é para amostras pequenas.")

# 3. A mesma análise, departamento por departamento
print("\n" + "=" * 70)
print("A MESMA pergunta, dentro de cada departamento")
print(f"{'dep':<5}{'H adm.':>8}{'H rej.':>8}{'M adm.':>8}{'M rej.':>8}"
      f"{'taxa H':>9}{'taxa M':>9}{'RC':>8}{'qui²':>9}{'p':>9}")
ors = []
for d in dep:
    t = np.array(tabela[d], float)
    q, pp, g, e = stats.chi2_contingency(t, correction=False)
    o = t[0,0]*t[1,1]/(t[0,1]*t[1,0]); ors.append(o)
    print(f"{d:<5}{int(t[0,0]):8d}{int(t[0,1]):8d}{int(t[1,0]):8d}{int(t[1,1]):8d}"
          f"{100*t[0,0]/t[0].sum():8.2f}%{100*t[1,0]/t[1].sum():8.2f}%{o:8.3f}{q:9.3f}{pp:9.4f}")
print("\nSó o departamento A é significativo - e a FAVOR das mulheres (RC = 0,349:")
print("a chance de admissão de um homem é 0,35 vez a de uma mulher).")

# 4. Mantel-Haenszel: a associação comum, controlando o departamento
num = sum(np.array(tabela[d],float)[0,0]*np.array(tabela[d],float)[1,1]/np.array(tabela[d],float).sum() for d in dep)
den = sum(np.array(tabela[d],float)[0,1]*np.array(tabela[d],float)[1,0]/np.array(tabela[d],float).sum() for d in dep)
print(f"\nRazão de chances de Mantel-Haenszel = {num/den:.4f}")
print(f"contra {orr:.4f} na tabela agregada. O sinal da associação se INVERTE.")
# estatística de Mantel-Haenszel
a = np.array([np.array(tabela[d],float)[0,0] for d in dep])
Ea = np.array([np.array(tabela[d],float)[0].sum()*np.array(tabela[d],float)[:,0].sum()/np.array(tabela[d],float).sum() for d in dep])
Va = np.array([np.prod([np.array(tabela[d],float)[0].sum(), np.array(tabela[d],float)[1].sum(),
                        np.array(tabela[d],float)[:,0].sum(), np.array(tabela[d],float)[:,1].sum()]) /
               (np.array(tabela[d],float).sum()**2*(np.array(tabela[d],float).sum()-1)) for d in dep])
mh = (abs(a.sum()-Ea.sum())-0.5)**2/Va.sum()
print(f"Estatística de Mantel-Haenszel = {mh:.4f} | gl = 1 | p-valor = {stats.chi2.sf(mh,1):.4f}")
print("Controlando o departamento, não há evidência de associação entre sexo e admissão.")

# 5. Por que a agregação engana
print("\n" + "=" * 70)
print("A explicação do paradoxo: as mulheres se candidataram aos departamentos")
print("mais concorridos.")
print(f"{'dep':<5}{'taxa global de admissão':>26}{'% de candidatas mulheres':>27}")
for d in dep:
    t = np.array(tabela[d], float)
    print(f"{d:<5}{100*(t[0,0]+t[1,0])/t.sum():25.2f}%{100*t[1].sum()/t.sum():26.2f}%")
taxa = np.array([(np.array(tabela[d],float)[0,0]+np.array(tabela[d],float)[1,0])/np.array(tabela[d],float).sum() for d in dep])
pmul = np.array([np.array(tabela[d],float)[1].sum()/np.array(tabela[d],float).sum() for d in dep])
print(f"\nCorrelação entre a taxa de admissão do departamento e a proporção de")
print(f"candidatas mulheres: r = {np.corrcoef(taxa, pmul)[0,1]:.4f}")
print("Os departamentos fáceis (A e B, com mais de 63% de admissão) receberam")
print("4% e 12% de candidatas; os difíceis (C e E, com 25% a 35%), cerca de 65%.")

print("\nO que aconteceria se as mulheres tivessem a MESMA distribuição de candidaturas")
print("que os homens, mantendo as taxas de admissão observadas em cada departamento:")
nh = np.array([np.array(tabela[d],float)[0].sum() for d in dep])
tm = np.array([np.array(tabela[d],float)[1,0]/np.array(tabela[d],float)[1].sum() for d in dep])
nm_total = T[1].sum()
esperadas = (nh/nh.sum()*nm_total*tm).sum()
print(f"  mulheres admitidas: {esperadas:.1f} de {nm_total:.0f} = {100*esperadas/nm_total:.2f}%")
print(f"  contra os {100*T[1,0]/T[1].sum():.2f}% observados e os {100*T[0,0]/T[0].sum():.2f}% dos homens.")
print("A diferença agregada não só desaparece como se inverte: padronizada pela")
print("distribuição de candidaturas dos homens, a taxa feminina passa a ser a maior.")
