# Mann-Whitney / Wilcoxon rank-sum: exemplo completo ----
# Pacote: apenas o stats (já vem com o R).

# 1. Dados: tempo até o reparo (h) de duas equipes de manutenção (dados fictícios)
equipeA <- c(1.6, 4.1, 2.8, 3.4, 5.2, 2.2, 2.4, 3.0, 1.2, 3.2, 1.8, 4.5, 3.7, 2.6, 2.0, 6.6)
equipeB <- c(60.0, 2.6, 2.0, 7.4, 4.5, 3.0, 4.2, 8.5, 6.7, 4.9, 3.9, 3.3, 3.6, 6.1, 5.6, 5.2)
dados <- data.frame(tempo = c(equipeA, equipeB),
                    equipe = factor(rep(c("A", "B"), each = 16)))
rbind(media = tapply(dados$tempo, dados$equipe, mean),
      mediana = tapply(dados$tempo, dados$equipe, median),
      dp = tapply(dados$tempo, dados$equipe, sd))

# 2. Por que não usar o teste t: assimetria forte e um valor extremo
par(mfrow = c(1, 2))
boxplot(tempo ~ equipe, data = dados, col = "gray85", ylab = "Tempo até o reparo (h)")
qqnorm(equipeB, pch = 19); qqline(equipeB, col = "red")
par(mfrow = c(1, 1))
tapply(dados$tempo, dados$equipe, function(v) shapiro.test(v)$p.value)
t.test(tempo ~ equipe, data = dados)        # Welch: não detecta

# 3. Teste de Mann-Whitney (o R chama de Wilcoxon rank sum)
wilcox.test(tempo ~ equipe, data = dados)

# 4. Com estimativa de Hodges-Lehmann e intervalo de confiança
wilcox.test(tempo ~ equipe, data = dados, conf.int = TRUE)

# 5. Cálculo manual: postos, somas e U
postos <- rank(dados$tempo)
R1 <- sum(postos[dados$equipe == "A"]); R2 <- sum(postos[dados$equipe == "B"])
n1 <- 16; n2 <- 16
c(R1 = R1, R2 = R2, total = R1 + R2, esperado = (n1 + n2) * (n1 + n2 + 1) / 2,
  U1 = R1 - n1 * (n1 + 1) / 2, U2 = R2 - n2 * (n2 + 1) / 2)

# 6. Tamanho do efeito: probabilidade de superioridade e delta de Cliff
U2 <- R2 - n2 * (n2 + 1) / 2
c(prob_superioridade = U2 / (n1 * n2), delta_de_Cliff = 2 * U2 / (n1 * n2) - 1)

# 7. O efeito do valor extremo e da transformação
semext <- equipeB[equipeB < 50]
c(t_sem_extremo = t.test(equipeA, semext)$p.value,
  MW_sem_extremo = wilcox.test(equipeA, semext)$p.value,
  t_com_log = t.test(log(equipeA), log(equipeB))$p.value)

# 8. Gráfico dos postos
plot(postos, as.numeric(dados$equipe), pch = 19, yaxt = "n",
     xlab = "Posto", ylab = "", ylim = c(0.5, 2.5))
axis(2, at = 1:2, labels = c("equipe A", "equipe B"))
