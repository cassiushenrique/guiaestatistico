# Regressão quantílica: exemplo completo ----
# Pacote: quantreg. Instale uma única vez: install.packages("quantreg")
library(quantreg)

# 1. Dados: 60 domicílios; renda mensal e gasto com alimentação, em R$ mil (dados fictícios)
dados <- data.frame(
  renda = c(1.2, 1.4, 1.4, 1.7, 1.7, 1.8, 1.9, 1.9, 2.1, 2.8, 3.4, 3.7, 3.7, 3.7, 3.8,
            3.8, 4.1, 4.3, 4.6, 4.7, 4.7, 5.2, 5.3, 5.4, 5.5, 5.6, 5.9, 5.9, 6.0, 6.5,
            6.6, 6.7, 6.7, 6.8, 7.2, 7.2, 8.2, 8.5, 8.5, 8.5, 8.7, 8.8, 9.2, 9.8, 9.8,
            9.9, 9.9, 9.9, 10.3, 10.3, 10.6, 10.6, 10.7, 10.9, 10.9, 11.3, 11.3, 11.5, 11.7, 12.0),
  gasto = c(0.74, 0.73, 0.57, 0.56, 0.66, 0.87, 0.95, 0.89, 1.01, 1.27, 1.07, 0.92, 1.76, 0.95, 0.62,
            1.16, 1.25, 1.36, 1.75, 0.94, 1.21, 1.63, 1.32, 1.66, 1.24, 1.58, 1.68, 2.35, 2.28, 2.24,
            0.83, 1.46, 1.24, 2.19, 1.72, 0.65, 0.41, 2.29, 1.92, 1.56, 1.32, 1.69, 2.47, 1.80, 2.46,
            1.19, 2.88, 1.91, 2.04, 2.30, 2.12, 2.22, 1.33, 3.18, 1.64, 1.52, 2.37, 1.76, 2.66, 2.76))

# 2. Mínimos quadrados (referência): modela a média
mq <- lm(gasto ~ renda, data = dados)
summary(mq)

# 3. Heterocedasticidade: a variabilidade cresce com a renda
metade <- dados$renda <= median(dados$renda)
c(dp_rendas_baixas = sd(residuals(mq)[metade]), dp_rendas_altas = sd(residuals(mq)[!metade]))

# 4. Regressão quantílica em cinco quantis
modelo <- rq(gasto ~ renda, tau = c(0.10, 0.25, 0.50, 0.75, 0.90), data = dados)
modelo
summary(modelo, se = "boot", R = 1000)

# 5. As inclinações diferem entre os quantis?
anova(rq(gasto ~ renda, tau = 0.10, data = dados),
      rq(gasto ~ renda, tau = 0.90, data = dados))

# 6. Robustez: efeito de um valor atípico
dados2 <- dados; dados2$gasto[60] <- 9.00
c(mq_antes = coef(mq)[2], mq_depois = coef(lm(gasto ~ renda, data = dados2))[2],
  mediana_antes = coef(rq(gasto ~ renda, tau = 0.5, data = dados))[2],
  mediana_depois = coef(rq(gasto ~ renda, tau = 0.5, data = dados2))[2])

# 7. Gráficos
plot(gasto ~ renda, data = dados, pch = 19, col = "gray50",
     xlab = "Renda mensal (R$ mil)", ylab = "Gasto com alimentação (R$ mil)")
for (t in c(0.10, 0.25, 0.50, 0.75, 0.90)) abline(rq(gasto ~ renda, tau = t, data = dados), col = "darkgreen")
abline(mq, col = "red", lwd = 2, lty = 2)

# Como a inclinação varia ao longo dos quantis
plot(summary(rq(gasto ~ renda, tau = seq(0.05, 0.95, 0.05), data = dados)))
