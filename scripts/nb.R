# Regressão binomial negativa: exemplo completo ----
# Pacote: MASS (acompanha a instalação padrão do R)
library(MASS)

# 1. Dados: warpbreaks (54 teares; rupturas do fio por tipo de lã e tensão)
c(media = mean(warpbreaks$breaks), variancia = var(warpbreaks$breaks),
  razao = var(warpbreaks$breaks) / mean(warpbreaks$breaks))

# 2. Modelo de Poisson (referência) e diagnóstico de superdispersão
mp <- glm(breaks ~ wool + tension, data = warpbreaks, family = poisson)
phi <- sum(residuals(mp, type = "pearson")^2) / df.residual(mp)
c(phi = phi, AIC = AIC(mp))

# 3. Modelo binomial negativa
mn <- glm.nb(breaks ~ wool + tension, data = warpbreaks)
summary(mn)
c(theta = mn$theta, erro_padrao_theta = mn$SE.theta, AIC = AIC(mn))

# 4. Razões de taxas e intervalos de confiança de 95%
exp(cbind(razao_de_taxas = coef(mn), confint(mn)))

# 5. Comparação dos dois modelos
# theta = 0 está na fronteira do espaço de parâmetros: o valor-p é metade do usual
G <- 2 * (logLik(mn) - logLik(mp))
c(G = as.numeric(G), valor_p = 0.5 * pchisq(as.numeric(G), 1, lower.tail = FALSE))

# 6. Efeito sobre os erros-padrão
cbind(poisson = summary(mp)$coefficients[, 2],
      binomial_negativa = summary(mn)$coefficients[, 2])

# 7. Variância implícita por cada modelo
mu <- c(15, 25, 35, 45)
cbind(mu, var_poisson = mu, var_bn = mu + mu^2 / mn$theta)

# 8. Gráficos
par(mfrow = c(1, 2))
hist(warpbreaks$breaks, breaks = 12, col = "gray85", main = "", xlab = "Rupturas")
curve(x, 5, 60, xlab = "Média prevista", ylab = "Variância", col = "darkgreen", lwd = 2)
curve(x + x^2 / mn$theta, add = TRUE, col = "red", lwd = 2)
legend("topleft", c("Poisson", "Binomial negativa"), col = c("darkgreen", "red"), lwd = 2, bty = "n")
par(mfrow = c(1, 1))
