# MANCOVA: exemplo completo ----
# Pacotes: stats (base), car (testes multivariados), emmeans (médias ajustadas), biotools
# install.packages(c("car", "emmeans", "biotools"))

# 1. Dados fictícios: as MESMAS 36 parcelas da ficha da ANCOVA, com uma segunda resposta.
mo <- c(3.82,2.17,3.92,3.88,2.48,3.17,4.13,3.79,3.75,3.39,3.23,2.69,
        3.29,1.37,3.83,3.21,3.28,3.35,2.98,2.49,2.84,2.39,3.49,2.75,
        2.49,3.90,1.74,2.29,1.76,3.19,1.38,2.34,2.43,3.44,1.70,2.49)
produtividade <- c(5309,4323,5577,5282,4288,4849,5513,5078,5120,4772,4946,4752,
                   5146,3691,5311,5377,5342,4838,5047,4578,5242,4517,5305,4803,
                   5092,5845,4461,4713,4182,5490,4199,4609,5057,5600,4502,4987)
proteina <- c(10.75,10.39,10.69,11.18,10.50,10.41,11.10,10.89,11.02,11.15,10.72,10.29,
              10.16, 9.86,10.61,10.33,10.28,10.92,10.38,10.21, 9.73,10.07,10.40,10.03,
               9.49, 9.99, 9.35, 9.59, 9.63,10.26, 9.14, 9.97, 9.20,10.02, 9.42, 9.62)
dados <- data.frame(cultivar = factor(rep(c("A", "B", "C"), each = 12)),
                    mo = mo, produtividade = produtividade, proteina = proteina)
aggregate(cbind(produtividade, proteina, mo) ~ cultivar, data = dados, FUN = mean)
cor(dados$produtividade, dados$proteina)              # correlação BRUTA

# 2. Pressuposto central: paralelismo multivariado (homogeneidade das inclinações)
library(car)
m_int <- lm(cbind(produtividade, proteina) ~ mo * cultivar, data = dados)
Anova(m_int, test.statistic = "Pillai")               # olhe a linha mo:cultivar

# 3. O modelo de MANCOVA: planos paralelos
modelo <- lm(cbind(produtividade, proteina) ~ mo + cultivar, data = dados)
Anova(modelo, type = "II", test.statistic = "Pillai")
Anova(modelo, type = "II", test.statistic = "Wilks")
summary(Anova(modelo, type = "II"))                   # as quatro estatísticas

# A correlação que interessa é a dos RESÍDUOS, não a bruta
cor(residuals(modelo))

# 4. Para comparar: a MANOVA sem a covariável
m_sem <- lm(cbind(produtividade, proteina) ~ cultivar, data = dados)
Anova(m_sem, test.statistic = "Pillai")
det(crossprod(residuals(modelo)))                     # volume do erro, com covariável
det(crossprod(residuals(m_sem)))                      # sem covariável

# 5. Médias ajustadas, resposta por resposta
library(emmeans)
for (r in c("produtividade", "proteina")) {
  mj <- lm(as.formula(paste(r, "~ mo + cultivar")), data = dados)
  print(r); print(emmeans(mj, ~cultivar)); print(pairs(emmeans(mj, ~cultivar), adjust = "tukey"))
}

# 6. ANCOVAs univariadas de acompanhamento, com ajuste
p <- sapply(c("produtividade", "proteina"), function(r)
  Anova(lm(as.formula(paste(r, "~ mo + cultivar")), data = dados), type = "II")["cultivar", "Pr(>F)"])
p.adjust(p, method = "bonferroni")

# 7. Pressupostos restantes
library(biotools)
boxM(residuals(modelo), dados$cultivar)
apply(residuals(modelo), 2, shapiro.test)
par(mfrow = c(1, 2)); for (j in 1:2) qqnorm(residuals(modelo)[, j]); par(mfrow = c(1, 1))

# 8. Gráficos
par(mfrow = c(1, 2))
plot(dados$produtividade, dados$proteina, col = dados$cultivar, pch = 19,
     xlab = "Produtividade (kg/ha)", ylab = "Proteína (%)", main = "Respostas brutas")
plot(residuals(modelo), col = dados$cultivar, pch = 19,
     xlab = "Resíduo de produtividade", ylab = "Resíduo de proteína", main = "Resíduos do modelo")
abline(h = 0, v = 0, col = "gray60"); abline(lm(residuals(modelo)[,2] ~ residuals(modelo)[,1]), col = "red", lwd = 2)
par(mfrow = c(1, 1))
