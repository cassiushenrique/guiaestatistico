# Regressão de Poisson: exemplo completo ----
# Pacote: apenas o stats (já vem com o R). Dados: warpbreaks.

# 1. Dados: 54 teares; número de rupturas do fio por tipo de lã e tensão
str(warpbreaks)
aggregate(breaks ~ wool + tension, data = warpbreaks,
          FUN = function(v) c(media = mean(v), variancia = var(v)))

# 2. Ajuste do modelo com ligação logarítmica
modelo <- glm(breaks ~ wool + tension, data = warpbreaks, family = poisson)
summary(modelo)

# 3. Razões de taxas (exp dos coeficientes) e intervalos de confiança de 95%
exp(cbind(razao_de_taxas = coef(modelo), confint(modelo)))

# 4. Testes da razão de verossimilhanças para cada termo
anova(modelo, test = "Chisq")

# 5. Diagnóstico de superdispersão
X2 <- sum(residuals(modelo, type = "pearson")^2)
phi <- X2 / df.residual(modelo)
c(qui2_Pearson = X2, gl = df.residual(modelo), phi = phi,
  deviance_por_gl = deviance(modelo) / df.residual(modelo))
pchisq(deviance(modelo), df.residual(modelo), lower.tail = FALSE)

# 6. Correção quase-Poisson: mesmos coeficientes, erros-padrão maiores
modelo_qp <- glm(breaks ~ wool + tension, data = warpbreaks, family = quasipoisson)
summary(modelo_qp)

# 7. Contagens previstas por combinação
novo <- expand.grid(wool = c("A", "B"), tension = c("L", "M", "H"))
cbind(novo, previsto = round(predict(modelo, novo, type = "response"), 3))

# 8. Gráficos
par(mfrow = c(1, 2))
boxplot(breaks ~ tension + wool, data = warpbreaks, col = "gray85", las = 2, xlab = "", ylab = "Rupturas")
plot(fitted(modelo), residuals(modelo, type = "pearson"), pch = 19,
     xlab = "Contagem prevista", ylab = "Resíduo de Pearson"); abline(h = 0, col = "red")
par(mfrow = c(1, 1))
