# Regressão linear simples: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: mtcars.

# 1. Gráfico de dispersão
plot(mpg ~ wt, data = mtcars, pch = 19, xlab = "Peso (1000 lb)", ylab = "Consumo (mpg)")

# 2. Ajuste do modelo mpg = b0 + b1 * wt
modelo <- lm(mpg ~ wt, data = mtcars)
summary(modelo)
confint(modelo)                 # intervalos de confiança de 95% dos coeficientes
anova(modelo)                   # tabela de análise de variância
abline(modelo, col = "red", lwd = 2)

# 3. Previsão para um carro de 3000 libras (wt = 3)
novo <- data.frame(wt = 3)
predict(modelo, novo, interval = "confidence")    # IC para a média
predict(modelo, novo, interval = "prediction")    # intervalo de predição

# 4. Diagnóstico dos resíduos
shapiro.test(residuals(modelo))
head(sort(cooks.distance(modelo), decreasing = TRUE), 3)
par(mfrow = c(2, 2)); plot(modelo); par(mfrow = c(1, 1))

# 5. Teste de Breusch-Pagan (pacote lmtest): install.packages("lmtest")
# lmtest::bptest(modelo)

# 6. Faixas de confiança e de predição
grade <- data.frame(wt = seq(1.5, 5.5, length.out = 100))
ic <- predict(modelo, grade, interval = "confidence")
ip <- predict(modelo, grade, interval = "prediction")
plot(mpg ~ wt, data = mtcars, pch = 19, xlab = "Peso (1000 lb)", ylab = "Consumo (mpg)")
matlines(grade$wt, cbind(ic, ip[, -1]), lty = c(1, 2, 2, 3, 3), col = c("red", "gray40", "gray40", "gray60", "gray60"))
