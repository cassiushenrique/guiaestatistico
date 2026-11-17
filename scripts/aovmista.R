# ANOVA mista (um fator entre e um fator dentro): exemplo completo ----
# Pacotes: stats (base), ez ou rstatix (esfericidade), emmeans (efeitos simples)
# install.packages(c("ez", "rstatix", "emmeans", "lme4"))

# 1. Dados fictícios: distância no teste de caminhada de seis minutos (m)
#    20 participantes alocados ao acaso em dois protocolos, medidos em 3 momentos.
dist <- c(430,441,473, 457,420,442, 427,438,457, 387,428,393, 440,489,479,
          442,440,434, 407,421,413, 427,450,448, 429,447,456, 421,452,466,
          423,451,507, 426,490,496, 410,441,458, 409,458,495, 391,433,485,
          428,482,523, 396,467,514, 406,442,498, 400,453,469, 390,453,483)
longo <- data.frame(
  participante = factor(rep(paste0("p", 1:20), each = 3)),
  protocolo    = factor(rep(c("A", "B"), each = 30)),      # ENTRE participantes
  momento      = factor(rep(c("inicio", "sem4", "sem8"), times = 20),
                        levels = c("inicio", "sem4", "sem8")),   # DENTRO
  distancia    = dist)
str(longo)
with(longo, table(protocolo, momento))
with(longo, tapply(distancia, list(protocolo, momento), mean))
with(longo, tapply(distancia, list(protocolo, momento), sd))

# 2. O modelo: Error() declara o participante como unidade do fator ENTRE
modelo <- aov(distancia ~ protocolo * momento + Error(participante / momento), data = longo)
summary(modelo)

# 3. Esfericidade e correções, com o pacote ez
library(ez)
ezANOVA(data = longo, dv = distancia, wid = participante,
        within = momento, between = protocolo, detailed = TRUE)
# alternativa: rstatix::anova_test(longo, dv = distancia, wid = participante,
#                                  within = momento, between = protocolo)

# 4. Pressupostos
#    4a. normalidade dos resíduos
res <- residuals(lm(distancia ~ protocolo * momento + participante, data = longo))
shapiro.test(res)
#    4b. homogeneidade entre os grupos ENTRE participantes
library(car)
medias_suj <- aggregate(distancia ~ participante + protocolo, data = longo, FUN = mean)
leveneTest(distancia ~ protocolo, data = medias_suj, center = median)
#    4c. homogeneidade das matrizes de covariância: teste M de Box
largo <- reshape(longo[, c("participante", "protocolo", "momento", "distancia")],
                 idvar = c("participante", "protocolo"), timevar = "momento",
                 direction = "wide")
library(biotools)
boxM(largo[, 3:5], largo$protocolo)

# 5. Interação significativa: efeitos simples
library(emmeans)
mm <- emmeans(modelo, ~momento | protocolo)
mm
pairs(mm, adjust = "holm")                      # efeito do momento DENTRO de cada protocolo
pairs(emmeans(modelo, ~protocolo | momento))    # efeito do protocolo EM CADA momento

# 6. Alternativa recomendada: modelo misto, que aceita dados faltantes
library(lme4); library(lmerTest)
mmisto <- lmer(distancia ~ protocolo * momento + (1 | participante), data = longo)
anova(mmisto)

# 7. Gráficos
par(mfrow = c(1, 2))
cores <- ifelse(longo$protocolo == "A", "darkgreen", "orange")
matplot(matrix(longo$distancia, nrow = 3), type = "b", pch = 19, lty = 1,
        col = rep(c("darkgreen", "orange"), each = 10), xaxt = "n",
        xlab = "Momento", ylab = "Distância (m)")
axis(1, at = 1:3, labels = c("início", "4 sem", "8 sem"))
with(longo, interaction.plot(momento, protocolo, distancia, lwd = 3, pch = 19, type = "b",
                             col = c("darkgreen", "orange"), ylab = "Distância média (m)"))
par(mfrow = c(1, 1))
