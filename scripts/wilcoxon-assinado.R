# Wilcoxon de postos sinalizados (pareado): exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: sleep.

# 1. Dados: horas extras de sono de 10 pacientes com dois medicamentos
#    Fonte original: Cushny e Peebles (1905); usados por Student (1908).
medicamento1 <- sleep$extra[sleep$group == 1]
medicamento2 <- sleep$extra[sleep$group == 2]
d <- medicamento2 - medicamento1
data.frame(paciente = 1:10, medicamento1, medicamento2, diferenca = d)
c(media = mean(d), mediana = median(d), dp = sd(d))

# 2. Por que não usar o teste t pareado aqui
shapiro.test(d)
qqnorm(d, pch = 19); qqline(d, col = "red", lwd = 2)

# 3. Construção dos postos (zeros são descartados; os postos vêm de |d|)
nz <- d[d != 0]
data.frame(diferenca = nz, absoluto = abs(nz), posto = rank(abs(nz)), sinal = sign(nz))
c(W_positivo = sum(rank(abs(nz))[nz > 0]), W_negativo = sum(rank(abs(nz))[nz < 0]))

# 4. Teste
wilcox.test(d, mu = 0)                       # ou wilcox.test(medicamento2, medicamento1, paired = TRUE)

# 5. Com estimativa de Hodges-Lehmann e intervalo de confiança
wilcox.test(d, mu = 0, conf.int = TRUE)

# 6. Valor-p exato: só vale sem empates e sem zeros
wilcox.test(d, mu = 0, exact = TRUE)         # aqui o R avisa e usa a aproximação
# Para comparar, o valor exato com 9 diferenças, todas positivas:
2 / 2^9

# 7. Comparação com o teste t pareado e o teste do sinal
t.test(d, mu = 0)
binom.test(sum(d > 0), sum(d != 0))

# 8. Gráficos
par(mfrow = c(1, 2))
matplot(rbind(medicamento1, medicamento2), type = "b", pch = 19, lty = 1, col = "gray50",
        xaxt = "n", xlab = "", ylab = "Horas extras de sono")
axis(1, at = 1:2, labels = c("medicamento 1", "medicamento 2"))
hist(d, col = "gray85", xlab = "Diferença (h)", main = ""); abline(v = 0, col = "red", lwd = 2)
par(mfrow = c(1, 1))
