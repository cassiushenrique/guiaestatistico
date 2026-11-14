# Shapiro–Wilk: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars (já vem com o R).

# 1. Dados
x <- mtcars$disp            # cilindrada de 32 automóveis (pol³)

# 2. Estatísticas descritivas
summary(x)
sd(x)
length(x)

# 3. Diagnóstico gráfico (sempre antes do teste)
par(mfrow = c(1, 3))
hist(x, breaks = "FD", freq = FALSE, col = "gray85", border = "white",
     main = "Histograma", xlab = "Cilindrada (pol³)")
curve(dnorm(x, mean(x), sd(x)), add = TRUE, col = "red", lwd = 2)
qqnorm(x, pch = 19, main = "Gráfico Q-Q"); qqline(x, col = "red", lwd = 2)
boxplot(x, main = "Boxplot", ylab = "Cilindrada (pol³)")
par(mfrow = c(1, 1))

# 4. Teste
resultado <- shapiro.test(x)
print(resultado)

# 5. Decisão (nível de significância de 5%)
cat(ifelse(resultado$p.value < 0.05,
           "Evidência contra normalidade",
           "Sem evidência suficiente contra normalidade"), "\n")

# 6. Efeito de uma transformação logarítmica
shapiro.test(log(x))

# 7. Assimetria e curtose amostrais (para julgar a magnitude do desvio)
m <- mean(x); s <- sd(x); n <- length(x)
assimetria <- sum((x - m)^3) / n / (sum((x - m)^2) / n)^(3/2)
assimetria
