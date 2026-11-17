# ANCOVA: exemplo completo ----
# Pacotes: stats (base), car (tipo II/III), emmeans (médias ajustadas)
# install.packages(c("car", "emmeans"))

# 1. Dados fictícios: produtividade (kg/ha) de três cultivares de milho em 36 parcelas
#    e o teor de matéria orgânica do solo (%) medido ANTES da semeadura.
mo <- c(3.82,2.17,3.92,3.88,2.48,3.17,4.13,3.79,3.75,3.39,3.23,2.69,
        3.29,1.37,3.83,3.21,3.28,3.35,2.98,2.49,2.84,2.39,3.49,2.75,
        2.49,3.90,1.74,2.29,1.76,3.19,1.38,2.34,2.43,3.44,1.70,2.49)
prod <- c(5309,4323,5577,5282,4288,4849,5513,5078,5120,4772,4946,4752,
          5146,3691,5311,5377,5342,4838,5047,4578,5242,4517,5305,4803,
          5092,5845,4461,4713,4182,5490,4199,4609,5057,5600,4502,4987)
dados <- data.frame(cultivar = factor(rep(c("A", "B", "C"), each = 12)),
                    mo = mo, produtividade = prod)
aggregate(cbind(produtividade, mo) ~ cultivar, data = dados, FUN = mean)
cor(dados$mo, dados$produtividade)

# 2. O que a ANOVA sem covariável responderia
summary(aov(produtividade ~ cultivar, data = dados))
# e o alerta: a covariável NÃO está equilibrada entre as cultivares
summary(aov(mo ~ cultivar, data = dados))

# 3. Pressuposto central: as retas são paralelas?
modelo_int <- lm(produtividade ~ mo * cultivar, data = dados)
anova(modelo_int)                       # olhe a linha mo:cultivar
# equivalente: anova(lm(produtividade ~ mo + cultivar, data = dados), modelo_int)
coef(modelo_int)                        # inclinação de cada cultivar

# 4. O modelo de ANCOVA: retas paralelas
modelo <- lm(produtividade ~ mo + cultivar, data = dados)
summary(modelo)
library(car)
Anova(modelo, type = "II")              # efeito do cultivar AJUSTADO pela covariável
anova(modelo)                           # sequencial: covariável primeiro

# 5. Médias ajustadas (médias marginais estimadas)
library(emmeans)
ma <- emmeans(modelo, ~cultivar)        # avaliadas na média da covariável
ma
pairs(ma, adjust = "tukey")
confint(pairs(ma, adjust = "tukey"))
# comparação com as médias observadas:
aggregate(produtividade ~ cultivar, data = dados, FUN = mean)

# 6. Pressupostos restantes
par(mfrow = c(2, 2)); plot(modelo); par(mfrow = c(1, 1))
shapiro.test(residuals(modelo))
leveneTest(residuals(modelo) ~ dados$cultivar, center = median)
# sobreposição das faixas da covariável entre os grupos:
tapply(dados$mo, dados$cultivar, range)

# 7. Gráficos
library(ggplot2)
ggplot(dados, aes(mo, produtividade, colour = cultivar)) +
  geom_point(size = 2.5) +
  geom_line(aes(y = fitted(modelo)), linewidth = 1) +
  geom_vline(xintercept = mean(dados$mo), linetype = "dotted") +
  labs(x = "Matéria orgânica do solo (%)", y = "Produtividade (kg/ha)") +
  theme_minimal()

# 8. Quando o paralelismo FALHA: o caso de mtcars
data(mtcars)
mt <- transform(mtcars, am = factor(am, labels = c("automatica", "manual")))
t.test(mpg ~ am, data = mt)                       # diferença bruta: 7,24 mpg
anova(lm(mpg ~ wt * am, data = mt))               # a interação wt:am É significativa
# nesse caso NÃO se reporta uma diferença ajustada única; descreva-a em função do peso:
library(emmeans)
mi <- lm(mpg ~ wt * am, data = mt)
emmeans(mi, ~am | wt, at = list(wt = c(1.6, 2.0, 2.4, 2.8, 3.2, 3.6))) |> pairs()
# ou procure a faixa em que a diferença é significativa (Johnson-Neyman):
# emmeans::emtrends(mi, ~am, var = "wt")
# interactions::johnson_neyman(mi, pred = am, modx = wt)
