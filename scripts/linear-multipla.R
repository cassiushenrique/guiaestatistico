# Regressão linear múltipla: exemplo completo ----
# Pacotes: stats (já vem com o R) e car (para o VIF): install.packages("car")
library(car)

# 1. Dados e dispersões entre as variáveis
dados <- mtcars[, c("mpg", "wt", "hp")]
pairs(dados, pch = 19)
cor(dados)

# 2. Modelo simples (referência) e modelo múltiplo
m1 <- lm(mpg ~ wt, data = dados)
m2 <- lm(mpg ~ wt + hp, data = dados)
summary(m2)
confint(m2)

# 3. Teste F parcial: acrescentar hp melhora o modelo?
anova(m1, m2)

# 4. Multicolinearidade: fator de inflação da variância
vif(m2)

# 5. Coeficientes padronizados
coef(lm(scale(mpg) ~ scale(wt) + scale(hp), data = dados))

# 6. Previsão: carro de 3000 libras e 150 hp
novo <- data.frame(wt = 3, hp = 150)
predict(m2, novo, interval = "confidence")
predict(m2, novo, interval = "prediction")

# 7. Diagnóstico dos resíduos
shapiro.test(residuals(m2))
head(sort(cooks.distance(m2), decreasing = TRUE), 3)
par(mfrow = c(2, 2)); plot(m2); par(mfrow = c(1, 1))

# 8. Teste de Breusch-Pagan (pacote lmtest): install.packages("lmtest")
# lmtest::bptest(m2)
