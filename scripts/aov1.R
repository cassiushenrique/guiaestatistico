# ANOVA de um fator: exemplo completo ----
# Pacotes: stats (base), car (Levene/Brown-Forsythe), multcomp (comparações múltiplas)
# install.packages(c("car", "multcomp"))
library(car)

# 1. Dados: PlantGrowth, peso seco de plantas sob controle e dois tratamentos
data(PlantGrowth)
str(PlantGrowth)
aggregate(weight ~ group, data = PlantGrowth,
          FUN = function(x) c(n = length(x), media = mean(x), dp = sd(x), var = var(x)))

# 2. Ajuste do modelo
modelo <- aov(weight ~ group, data = PlantGrowth)
summary(modelo)

# 3. Pressupostos, examinados nos RESÍDUOS do modelo
par(mfrow = c(2, 2)); plot(modelo); par(mfrow = c(1, 1))
shapiro.test(residuals(modelo))
leveneTest(weight ~ group, data = PlantGrowth, center = mean)     # Levene
leveneTest(weight ~ group, data = PlantGrowth, center = median)   # Brown-Forsythe
bartlett.test(weight ~ group, data = PlantGrowth)

# 4. Tamanho de efeito
tab <- summary(modelo)[[1]]
SQ_entre <- tab[1, "Sum Sq"]; SQ_dentro <- tab[2, "Sum Sq"]
QM_dentro <- tab[2, "Mean Sq"]; gl_entre <- tab[1, "Df"]
SQ_entre / (SQ_entre + SQ_dentro)                                  # eta ao quadrado
(SQ_entre - gl_entre * QM_dentro) / (SQ_entre + SQ_dentro + QM_dentro)   # omega ao quadrado
# equivalente: effectsize::eta_squared(modelo); effectsize::omega_squared(modelo)

# 5. Comparações múltiplas de Tukey
TukeyHSD(modelo)
plot(TukeyHSD(modelo))
# equivalente, com mais recursos:
# library(multcomp); summary(glht(modelo, linfct = mcp(group = "Tukey")))

# 6. Contraste planejado: controle contra a média dos dois tratamentos
contrastes <- matrix(c(1, -0.5, -0.5), nrow = 1,
                     dimnames = list("ctrl vs trts", levels(PlantGrowth$group)))
summary(glht(modelo, linfct = mcp(group = contrastes)))

# 7. Alternativas, para comparar
oneway.test(weight ~ group, data = PlantGrowth, var.equal = FALSE)   # ANOVA de Welch
kruskal.test(weight ~ group, data = PlantGrowth)                     # Kruskal-Wallis

# 8. Gráficos
par(mfrow = c(1, 2))
boxplot(weight ~ group, data = PlantGrowth, col = "gray85",
        xlab = "Tratamento", ylab = "Peso seco (g)")
points(jitter(as.numeric(PlantGrowth$group), 0.5), PlantGrowth$weight, pch = 19, col = "gray40")
plot(TukeyHSD(modelo))
par(mfrow = c(1, 1))
