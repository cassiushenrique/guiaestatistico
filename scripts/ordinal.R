# Regressão logística ordinal (chances proporcionais): exemplo completo ----
# Pacote: MASS (acompanha a instalação padrão do R)
library(MASS)

# 1. Dados: mtcars; resposta ordinal = número de cilindros (4 < 6 < 8), preditor = peso
dados <- mtcars
dados$cilindros <- factor(dados$cyl, levels = c(4, 6, 8), ordered = TRUE)
table(dados$cilindros)
aggregate(wt ~ cilindros, data = dados, FUN = mean)

# 2. Ajuste do modelo de chances proporcionais
modelo <- polr(cilindros ~ wt, data = dados, Hess = TRUE)
summary(modelo)

# 3. Valores-p de Wald (o polr não os imprime)
ct <- coef(summary(modelo))
cbind(ct, "valor_p" = 2 * pnorm(abs(ct[, "t value"]), lower.tail = FALSE))

# 4. Razão de chances e intervalo de confiança de 95%
exp(coef(modelo))
exp(confint(modelo))        # por verossimilhança perfilada

# 5. Teste da razão de verossimilhanças contra o modelo nulo
nulo <- polr(cilindros ~ 1, data = dados)
anova(nulo, modelo)

# 6. Probabilidades estimadas
novo <- data.frame(wt = c(2.0, 2.5, 3.0, 3.5, 4.0, 5.0))
cbind(novo, round(predict(modelo, novo, type = "probs"), 3))

# 7. Verificação das chances proporcionais
#    (a) duas logísticas binárias com cortes diferentes: as inclinações devem ser parecidas
coef(glm(I(cyl >= 6) ~ wt, data = mtcars, family = binomial))
coef(glm(I(cyl >= 8) ~ wt, data = mtcars, family = binomial))
#    (b) teste formal: install.packages("brant"); brant::brant(modelo)

# 8. Gráfico: probabilidades estimadas por categoria
grade <- data.frame(wt = seq(1.4, 5.6, length.out = 200))
P <- predict(modelo, grade, type = "probs")
matplot(grade$wt, P, type = "l", lty = 1, lwd = 2,
        xlab = "Peso (1000 lb)", ylab = "Probabilidade estimada")
legend("right", legend = levels(dados$cilindros), col = 1:3, lty = 1, lwd = 2, bty = "n")
