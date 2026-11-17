# ANOVA fatorial (dois fatores com interação): exemplo completo ----
# Pacotes: stats (base), car (tipos de soma de quadrados), emmeans (efeitos simples)
# install.packages(c("car", "emmeans"))

# 1. Dados: warpbreaks, rupturas do fio em teares, por tipo de lã e tensão
data(warpbreaks)
str(warpbreaks)
table(warpbreaks$wool, warpbreaks$tension)         # 9 observações por célula: balanceado
with(warpbreaks, tapply(breaks, list(wool, tension), mean))
with(warpbreaks, tapply(breaks, list(wool, tension), sd))

# 2. O gráfico de interação vem ANTES da tabela
with(warpbreaks, interaction.plot(tension, wool, breaks,
                                  ylab = "Média de rupturas", xlab = "Tensão",
                                  col = c("darkgreen", "orange"), lwd = 3, pch = 19, type = "b"))

# 3. O modelo completo
modelo <- aov(breaks ~ wool * tension, data = warpbreaks)   # o mesmo que wool + tension + wool:tension
summary(modelo)

# 4. Pressupostos
par(mfrow = c(2, 2)); plot(modelo); par(mfrow = c(1, 1))
shapiro.test(residuals(modelo))
library(car)
leveneTest(breaks ~ wool * tension, data = warpbreaks, center = median)
with(warpbreaks, tapply(breaks, list(wool, tension), var))

# 5. Tamanho de efeito
library(effectsize)
eta_squared(modelo, partial = TRUE)
omega_squared(modelo, partial = TRUE)

# 6. Interação significativa: a leitura passa a ser por EFEITOS SIMPLES
library(emmeans)
md <- emmeans(modelo, ~tension | wool)        # efeito da tensão dentro de cada lã
md
pairs(md, adjust = "tukey")
joint_tests(modelo, by = "wool")              # teste F da tensão dentro de cada lã

me <- emmeans(modelo, ~wool | tension)        # efeito da lã dentro de cada tensão
pairs(me)

# 7. Tukey entre as seis células
TukeyHSD(modelo, which = "wool:tension")

# 8. Se as variâncias forem heterogêneas: a escala da raiz quadrada
modelo_raiz <- aov(sqrt(breaks) ~ wool * tension, data = warpbreaks)
summary(modelo_raiz)
leveneTest(sqrt(breaks) ~ wool * tension, data = warpbreaks, center = median)

# 9. Para delineamentos DESBALANCEADOS, declare o tipo de soma de quadrados
options(contrasts = c("contr.sum", "contr.poly"))
Anova(lm(breaks ~ wool * tension, data = warpbreaks), type = "III")
options(contrasts = c("contr.treatment", "contr.poly"))   # volta ao padrão
