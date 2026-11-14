# Regressão logística binária: exemplo completo ----
# Pacote: apenas o stats (já vem com o R).

# 1. Dados: 23 lançamentos do ônibus espacial antes do acidente da Challenger (Dalal et al., 1989)
dados <- data.frame(
  temp  = c(66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58),
  falha = c( 0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1))
table(dados$falha)
tapply(dados$temp, dados$falha, mean)

# 2. Ajuste do modelo logit(p) = b0 + b1 * temperatura
modelo <- glm(falha ~ temp, data = dados, family = binomial)
summary(modelo)

# 3. Razão de chances (odds ratio) e intervalos de confiança de 95%
exp(coef(modelo))
exp(confint.default(modelo))     # IC de Wald
exp(confint(modelo))             # IC por verossimilhança perfilada

# 4. Teste da razão de verossimilhanças
anova(modelo, test = "Chisq")

# 5. Probabilidades estimadas
novo <- data.frame(temp = c(81, 70, 65, 53, 31))
cbind(novo, prob = predict(modelo, novo, type = "response"))
-coef(modelo)[1] / coef(modelo)[2]          # temperatura em que p = 0,5

# 6. Classificação com ponto de corte 0,5
prob <- fitted(modelo)
table(observado = dados$falha, previsto = as.numeric(prob >= 0.5))

# 7. Área sob a curva ROC (pacote pROC): install.packages("pROC")
# pROC::auc(dados$falha, prob)

# 8. Gráfico: dados e curva ajustada
plot(falha ~ temp, data = dados, pch = 19, xlim = c(30, 85),
     xlab = "Temperatura (°F)", ylab = "Probabilidade de dano")
curve(predict(modelo, data.frame(temp = x), type = "response"), add = TRUE, col = "red", lwd = 2)
abline(v = 31, lty = 2)
