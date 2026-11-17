# Tukey HSD / Tukey-Kramer: exemplo completo ----
# Pacotes: stats (base), multcomp e emmeans (versões com mais recursos), multcompView (letras)
# install.packages(c("multcomp", "emmeans", "multcompView", "agricolae"))

# 1. Dados: chickwts, peso de pintinhos por tipo de ração
data(chickwts)
table(chickwts$feed)                 # n desiguais: 10 a 14
modelo <- aov(weight ~ feed, data = chickwts)
summary(modelo)
tapply(chickwts$weight, chickwts$feed, function(x) c(n = length(x), media = mean(x), dp = sd(x)))

# 2. Pressupostos (os mesmos da ANOVA)
par(mfrow = c(2, 2)); plot(modelo); par(mfrow = c(1, 1))
shapiro.test(residuals(modelo))
car::leveneTest(weight ~ feed, data = chickwts, center = median)

# 3. O procedimento
tk <- TukeyHSD(modelo, conf.level = 0.95)
tk
par(mar = c(5, 11, 4, 2)); plot(tk, las = 1, cex.axis = 0.7); par(mar = c(5, 4, 4, 2))

# 4. De onde vem o valor crítico: a distribuição da amplitude estudentizada
k <- nlevels(chickwts$feed); gl <- df.residual(modelo)
qtukey(0.95, k, gl)                  # q crítico
qtukey(0.95, k, gl) / sqrt(2)        # o mesmo na escala do teste t
qt(0.975, gl)                        # para comparar: t sem ajuste
qtukey(0.95, 2, gl) / sqrt(2)        # com k = 2, Tukey vira o teste t

# 5. Diferença mínima significativa, se os grupos fossem balanceados
QM <- sum(residuals(modelo)^2) / gl
n_medio <- mean(table(chickwts$feed))
qtukey(0.95, k, gl) * sqrt(QM / n_medio)    # DMS aproximada

# 6. Versões com mais recursos
library(multcomp)
summary(glht(modelo, linfct = mcp(feed = "Tukey")))
confint(glht(modelo, linfct = mcp(feed = "Tukey")))

library(emmeans)
em <- emmeans(modelo, ~feed)
pairs(em, adjust = "tukey")
plot(em, comparisons = TRUE)         # setas que se sobrepõem indicam ausência de diferença

# 7. Apresentação em letras
library(multcompView)
letras <- multcompLetters4(modelo, tk)
letras
# alternativa, com a tabela pronta:
library(agricolae)
HSD.test(modelo, "feed", console = TRUE)

# 8. Gráfico com médias, intervalos e letras
medias <- tapply(chickwts$weight, chickwts$feed, mean)
ns <- table(chickwts$feed)
eps <- sqrt(QM / ns)
ordem <- order(medias)
plot(medias[ordem], seq_along(medias), pch = 19, cex = 1.4, yaxt = "n", xlim = c(100, 450),
     xlab = "Peso aos 6 dias (g)", ylab = "")
axis(2, at = seq_along(medias), labels = names(medias)[ordem], las = 1)
arrows(medias[ordem] - qt(0.975, gl) * eps[ordem], seq_along(medias),
       medias[ordem] + qt(0.975, gl) * eps[ordem], seq_along(medias),
       angle = 90, code = 3, length = 0.06)
