# Teste de Yuen pareado: exemplo completo ----
# Pacote: WRS2. Instale uma única vez: install.packages("WRS2")
library(WRS2)

# 1. Dados: dois métodos de medição aplicados às MESMAS 20 peças (dados fictícios)
#    Registrou-se a diferença entre as leituras de cada peça.
diferenca <- c(2.0, 3.8, -6.8, 1.3, 16.5, -9.4, 3.5, 4.0, 1.7, 3.3,
               2.2, 2.9, 2.7, 5.4, 3.1, 4.3, 0.6, 13.2, 2.5, 4.7)
c(media = mean(diferenca), aparada20 = mean(diferenca, trim = 0.2),
  mediana = median(diferenca), dp = sd(diferenca))

# 2. Diagnóstico: as diferenças têm caudas pesadas
par(mfrow = c(1, 2))
hist(diferenca, col = "gray85", xlab = "Diferença", main = ""); abline(v = 0, col = "red", lwd = 2)
qqnorm(diferenca, pch = 19); qqline(diferenca, col = "red", lwd = 2)
par(mfrow = c(1, 1))
shapiro.test(diferenca)

# 3. Teste t pareado comum
t.test(diferenca, mu = 0)

# 4. Teste de Yuen pareado
#    Com as duas medidas em mãos, a função é yuend(x, y). Como aqui só temos as
#    diferenças, o teste equivale a aplicar o intervalo aparado de uma amostra a elas.
trimci(diferenca, tr = 0.2)
# yuend(metodoB, metodoA, tr = 0.2)   # equivalente, quando os dois vetores existem

# 6. Efeito da proporção de aparagem
for (tr in c(0.05, 0.10, 0.20, 0.25)) {
  r <- trimci(diferenca, tr = tr, pr = FALSE)
  cat("aparagem", tr, ": media aparada =", round(mean(diferenca, trim = tr), 4),
      " IC = [", round(r$ci[1], 4), ";", round(r$ci[2], 4), "]\n")
}

# 7. Alternativas não paramétricas
wilcox.test(diferenca, mu = 0)
binom.test(sum(diferenca > 0), length(diferenca))
